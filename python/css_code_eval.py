import numpy as np
import numpy.random as npr
import networkx as nx
import networkx.algorithms.bipartite as bpt
import scipy.sparse as sp

import pym4ri as m4ri
import numba
from tqdm import tqdm

from experiments_settings import gosper_next, long_gosper_next
from numba_bitset import Bitset

# TODO: implement some strategy for importance sampling

def CSS_HGP_code_from_state(state: nx.MultiGraph) -> tuple[sp.sparray, int, int]:
    # Convert multigraph state to simple graph
    G = nx.Graph(state)
    
    # Extract biadjacency matrix from the Tanner graph of the classical code
    c = [n for n, b in G.nodes(data='bipartite') if b == 0]
    v = [n for n, b in G.nodes(data='bipartite') if b == 1]
    H = bpt.biadjacency_matrix(G, row_order=sorted(c), column_order=sorted(v), dtype=np.bool_)
    m, n = H.shape
    
    # Compute HGP code from classical code
    Hx = sp.hstack([sp.kron(sp.eye_array(m), H.T), # Im x H2
                    sp.kron(H, sp.eye_array(n)),   # H1 x In  
                   ], dtype=np.bool_).tocsc() # [ Im x H2 | H1 x In ] } m*n rows

    Hz = sp.hstack([sp.kron(H.T, sp.eye_array(m)), # H1'x Im
                    sp.kron(sp.eye_array(n), H),   # In x H2'
                   ], dtype=np.bool_).tocsc() # [ H1'x Im | In x H2'] } m*n rows
    
    H = sp.vstack([Hx, Hz])

    N = m*m + n*n
    K = N - m4ri.rank(Hx.todense()) - m4ri.rank(Hz.todense())

    # Return stacked CSS matrices H = [Hx \\ Hz], N = number of qubits and K = code dimension
    return H, N, K

def MC_erasure_plog_fixed_p(num_trials: int, state: nx.MultiGraph, p: float, 
                    rank_method: bool=False, only_X: bool=False) -> float:
    return MC_erasure_plog(num_trials, state, [p], rank_method, only_X)['mean'][0]

def MC_erasure_plog(num_trials: int, state: nx.MultiGraph, p_vals: list[float], 
                    rank_method: bool=False, only_X: bool=False) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    shape = (len(c), len(v))
    edgelist = list(nx.Graph(state).edges(data=False))
    return m4ri.MC_erasure_plog(shape, edgelist, num_trials, p_vals, rank_method, only_X)

def _peel(erasure: np.array, H: sp.csr_array) -> bool:
    while np.any(erasure):
        erased_cols = np.nonzero(erasure)[0]
        H_E = H[:, erasure]
        dangling_checks = np.nonzero(np.diff(H_E.indptr) == 1)[0]
        
        if len(dangling_checks) > 0:    
            erasure[erased_cols[H_E.indices[H_E.indptr[dangling_checks[0]]]]] = 0
        else:
            return False
    
    return True

# Idea to improve peeling implementation (same algo, just more efficient implementation):
# construct the erased graph structure via adjacency lists:
# iterate over all checks (rows of sparse pcm) to count their degree towards the erasure, 
# and add them to the adjacency list of each (erased) bit in their neighborhood
# and add every degree-1 check to a stack for fast retrieval;
# keep track of the number of remaining erasures;
# while num_erasures > 0:
# look for a dangling check in the stack (if None -> failure immediately), 
# if any, pick it, find its dangling bit (only remaining erased bit in its adj list),
# unerase that bit (decrease num erasures), and while at it, decrease the degree of 
# all the checks in the adj list of this bit, also while at it, add any new dangling 
# check to the stack. 
# 

@numba.njit
def _fast_peel(erasure: np.ndarray, Hindptr: np.ndarray, Hindices: np.ndarray) -> bool:
    # Build adjacency list structure for erased bits, and keep track of their count
    erased_bits = np.nonzero(erasure)[0]
    num_erased_bits = len(erased_bits)
    reversed_erased_bits = np.zeros((len(erasure),), dtype=np.int32)
    for i, e in enumerate(erased_bits):
        reversed_erased_bits[e] = i
    bit_adj_lists = np.zeros((num_erased_bits, len(Hindptr)-1), dtype=np.int32)
    erased_bit_degrees = np.zeros((num_erased_bits,), dtype=np.int32)

    # Keep track of the degree of the checks towards the erasure and stack the dangling checks
    check_erasure_degrees = np.zeros((len(Hindptr)-1,), dtype=np.int32)
    dangling_check_stack, stack_size = np.zeros((len(Hindptr)-1,), dtype=np.int32), 0
    
    # Populate the adjacency lists of the erased bits and add the first dangling checks
    for check_idx in range(len(Hindptr)-1):
        for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
            if erasure[bit_idx]:
                check_erasure_degrees[check_idx] += 1
                rev_bit_idx = reversed_erased_bits[bit_idx]
                bit_adj_lists[rev_bit_idx][erased_bit_degrees[rev_bit_idx]] = check_idx
                erased_bit_degrees[rev_bit_idx] += 1

        if check_erasure_degrees[check_idx] == 1:
            dangling_check_stack[stack_size] = check_idx
            stack_size += 1

    # Main peeling loop
    while num_erased_bits > 0:
        # Find a dangling check
        dangling_check_idx = -1
        # Pop checks whose dangling bit has already been unerased
        while stack_size > 0 and dangling_check_idx == -1:
            dangling_check_idx = dangling_check_stack[stack_size-1]
            stack_size -= 1
            if check_erasure_degrees[dangling_check_idx] == 0:
                dangling_check_idx = -1

        # If no dangling checks remain, failure
        if dangling_check_idx == -1:
            return False

        # Find the corresponding dangling bit
        for bit_idx in Hindices[Hindptr[dangling_check_idx]:Hindptr[dangling_check_idx+1]]:
            if erasure[bit_idx]:
                dangling_bit_idx = bit_idx
                break
        # Unerase it and decrease the erasure count
        erasure[dangling_bit_idx] = False
        num_erased_bits -= 1
        # Decrease the degrees of each neighboring check
        rev_bit_idx = reversed_erased_bits[dangling_bit_idx]
        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
            check_erasure_degrees[check_idx] -= 1
            # If a check becomes dangling, add it to the stack
            if check_erasure_degrees[check_idx] == 1:
                dangling_check_stack[stack_size] = check_idx
                stack_size += 1

    # If there are no erasures left, success
    return True

@numba.njit
def numba_unpack(packed):
    count = packed.size * 8
    out = np.zeros(count, dtype=np.uint8)
    for i in range(count):
        byte = packed[i // 8]
        out[i] = (byte >> (i % 8)) & 1  # LSB first
    return out

@numba.njit
def numba_pack(arr):
    n = arr.size
    out_len = (n + 7) // 8
    out = np.zeros(out_len, dtype=np.uint8)
    for i in range(n):
        if arr[i]:
            out[i // 8] |= 1 << (i % 8)  # LSB first
    return out

@numba.njit
def rearrange(arr: np.ndarray, mask: np.uint64) -> np.ndarray:
    # spreaded_mask = np.unpackbits(mask.reshape(1).view(np.uint8), bitorder='little')
    spreaded_mask = numba_unpack(mask.view(np.uint8))
    # spreaded_branches = np.unpackbits(arr.view(np.uint8).reshape(-1, 8), bitorder='little').reshape(-1, 64)
    spreaded_branches = numba_unpack(arr.flatten().view(np.uint8)).reshape(-1, spreaded_mask.size)
    
    return numba_pack(np.hstack((spreaded_branches[:, spreaded_mask == 0], 
                                 spreaded_branches[:, spreaded_mask == 1]&0)
                                 ).flatten()).view(np.uint64).reshape(arr.shape)
    

@numba.njit
def peel_guess(erasure: np.ndarray, Hindptr: np.ndarray, Hindices: np.ndarray) -> tuple[np.ndarray, int, int, tuple[int]]:   
    # Build adjacency list structure for erased bits, and keep track of their count
    erased_bits = np.nonzero(erasure)[0]
    num_erased_bits = len(erased_bits)
    reversed_erased_bits = np.zeros((len(erasure),), dtype=np.int32)
    for i, e in enumerate(erased_bits):
        reversed_erased_bits[e] = i
    bit_adj_lists = np.zeros((num_erased_bits, len(Hindptr)-1), dtype=np.int32)
    erased_bit_degrees = np.zeros((num_erased_bits,), dtype=np.int32)

    # Keep track of the degree of the checks towards the erasure and stack the dangling checks
    check_erasure_degrees = np.zeros((len(Hindptr)-1,), dtype=np.int32)
    dangling_check_stack, stack_size = np.zeros((len(Hindptr)-1,), dtype=np.int32), 0
    
    # Populate the adjacency lists of the erased bits and add the first dangling checks
    for check_idx in range(len(Hindptr)-1):
        for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
            if erasure[bit_idx]:
                check_erasure_degrees[check_idx] += 1
                rev_bit_idx = reversed_erased_bits[bit_idx]
                bit_adj_lists[rev_bit_idx][erased_bit_degrees[rev_bit_idx]] = check_idx
                erased_bit_degrees[rev_bit_idx] += 1

        if check_erasure_degrees[check_idx] == 1:
            dangling_check_stack[stack_size] = check_idx
            stack_size += 1

    # Guessing parameters
    guessed_bit_initialization = lambda ng: np.uint64(((1<<(1<<ng)) - 1) << (1<<ng))
    max_guesses = 6 # max uint type has size 64 = 2**6 bits
    num_guesses = 0
    max_used_guesses = 0
    erasure_sizes = np.zeros(7, dtype=np.uint32)
    stopping_size = 0
    
    # Initialize bit correction and cumulative syndrome branches
    bit_correction = np.zeros((len(erasure),), dtype=np.uint64)
    check_syndrome = np.zeros((len(Hindptr)-1,), dtype=np.uint64)
    
    # Main peeling loop
    while num_erased_bits > 0:
        # Find a dangling check
        dangling_check_idx = -1
        # Pop checks whose dangling bit has already been unerased
        while stack_size > 0 and dangling_check_idx == -1:
            dangling_check_idx = dangling_check_stack[stack_size-1]
            stack_size -= 1
            if check_erasure_degrees[dangling_check_idx] == 0:
                # Verify for consistency. 
                # Alternatively, defer consistency verifications to the end, by adding these checks to another stack/queue. 
                # Another idea, in the opposite direction, whenever you decrease the degrees, check if the equation became empty, 
                # and verify its consistency right away
                if check_syndrome[dangling_check_idx] != 0:
                    assert num_guesses > 0
                    # Select inconsistent branches
                    branch_mask = check_syndrome[dangling_check_idx]
                    # Rearrange branches to free up space
                    bit_correction = rearrange(bit_correction, branch_mask)
                    check_syndrome = rearrange(check_syndrome, branch_mask)
                    num_guesses -= 1

                    assert check_syndrome[dangling_check_idx] == 0                    
                
                dangling_check_idx = -1

        # If no dangling checks remain, peeling is blocked -> resort to guessing=
        if dangling_check_idx == -1:
            if num_guesses == 0:
                stopping_size = np.count_nonzero(erasure)
            # If guesses are available, use one
            if num_guesses < max_guesses:
                # Guess score to choose the guessed bit
                guess_score = np.zeros((len(np.nonzero(erasure)[0]),), dtype=np.int32)
                for i, b in enumerate(np.nonzero(erasure)[0]):
                    rev_bit_idx = reversed_erased_bits[b]
                    for check_idx in bit_adj_lists[rev_bit_idx]:
                        if check_erasure_degrees[check_idx] == 2:
                            guess_score[i] += 1
                guess_bit = np.nonzero(erasure)[0][np.argmax(guess_score)]

                # Guess the correction at guess_bit as being 1 and duplicate other values
                # TODO: think of a more efficient way to do this duplication, i.e., duplicate only the nonempty bits/checks (sparse arrays)
                bit_correction |= (bit_correction << np.uint64(1<<num_guesses))
                bit_correction[guess_bit] = guessed_bit_initialization(num_guesses)
                check_syndrome |= (check_syndrome << np.uint64(1<<num_guesses))
                
                # Update guessing budget and record stats
                num_guesses += 1
                if num_guesses > max_used_guesses:
                    erasure_sizes[max_used_guesses] = stopping_size
                    max_used_guesses = num_guesses
                
                # Unerase the guessed bit
                assert erasure[guess_bit]
                erasure[guess_bit] = False
                num_erased_bits -= 1
                # Decrease the degrees of each neighboring check
                rev_bit_idx = reversed_erased_bits[guess_bit]
                for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
                    check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
                    check_erasure_degrees[check_idx] -= 1
                    # Propagate branches to check syndromes
                    check_syndrome[check_idx] ^= bit_correction[guess_bit]
                    # If a check becomes dangling, add it to the stack
                    if check_erasure_degrees[check_idx] == 1:
                        dangling_check_stack[stack_size] = check_idx
                        stack_size += 1
                
                # Restart the search for dangling checks
                continue
                
            else:
                # Guessing budget exhausted: signal failure
                erasure_sizes[max_used_guesses] = stopping_size
                return np.ones_like(erasure, dtype=np.uint64)*((1<<64) - 1), num_guesses, max_guesses+1, erasure_sizes

        # Find the corresponding dangling bit
        for bit_idx in Hindices[Hindptr[dangling_check_idx]:Hindptr[dangling_check_idx+1]]:
            if erasure[bit_idx]:
                dangling_bit_idx = bit_idx
                break
        # Unerase it and decrease the erasure count
        erasure[dangling_bit_idx] = False
        # Solve dangling equation for the dangling variable in all branches
        bit_correction[dangling_bit_idx] = check_syndrome[dangling_check_idx]
        num_erased_bits -= 1
        # Decrease the degrees of each neighboring check
        rev_bit_idx = reversed_erased_bits[dangling_bit_idx]
        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
            check_erasure_degrees[check_idx] -= 1
            # Propagate branches to syndromes
            check_syndrome[check_idx] ^= bit_correction[dangling_bit_idx]
            # If a check becomes dangling, add it to the stack
            if check_erasure_degrees[check_idx] == 1:
                dangling_check_stack[stack_size] = check_idx
                stack_size += 1

    
    # When no erasures are left, check one last time for inconsistencies
    while stack_size > 0:
        dangling_check_idx = dangling_check_stack[stack_size-1]
        stack_size -= 1
        assert check_erasure_degrees[dangling_check_idx] == 0
        
        # Verify for consistency
        if check_syndrome[dangling_check_idx] != 0:
            assert num_guesses > 0
            # Select inconsistent branches
            branch_mask = check_syndrome[dangling_check_idx]
            # Rearrange branches to free up space
            bit_correction = rearrange(bit_correction, branch_mask)
            check_syndrome = rearrange(check_syndrome, branch_mask)
            num_guesses -= 1

            assert check_syndrome[dangling_check_idx] == 0                    
    
    # If no erasures left, nor inconsistent equations, check if the alternative corrections are nontrivial (up to a stabilizer)
    assert not np.any(erasure)
    return bit_correction, num_guesses, max_used_guesses, erasure_sizes

@numba.njit
def smallest_connected_component(erasure: np.ndarray, 
                                 reversed_erased_bits: np.ndarray, 
                                 bit_adj_lists: np.ndarray, 
                                 erased_bit_degrees: np.ndarray, 
                                 Hindices: np.ndarray, 
                                 Hindptr: np.ndarray) -> np.ndarray:
    # Keep track of the size of the smallest component, and the current component
    erased_bits = np.nonzero(erasure)[0]
    smallest_component = erased_bits.copy() # initialize with the whole erasure as the CC
    current_component = np.zeros(len(erased_bits), dtype=np.int32)
    current_component_size, smallest_component_size = 0, len(erased_bits)

    # Start decomposing the erasure. Need to go all the way, and complete the description 
    # of a component even if I already know it is larger than the minimum
    unassigned_bits = erasure.copy() # to know of when the procedure is done, keep track of the bits still unassigned to a CC
    unassigned_checks = np.ones(len(Hindptr)-1, dtype=np.int32) # keep track of the assignment of the checks as well to avoid traversing the same check twice
    ub_ptr = 0
    while np.count_nonzero(unassigned_bits) > 0:
        # Pick some unassigned bit and start computing its CC
        while unassigned_bits[erased_bits[ub_ptr]] == 0:
            ub_ptr += 1
        
        # Have my unassigned bit, erased_bits[ub_ptr], now lets find its connected component
        current_component[current_component_size] = erased_bits[ub_ptr]
        current_component_size += 1
        unassigned_bits[erased_bits[ub_ptr]] = 0 # mark this bit as assigned to a CC

        cc_ptr = 0

        # For each bit in the CC, we iterate over its neighbors and add them to the CC. Once you reach the last bit already added to the CC, you stop
        while cc_ptr < current_component_size:
            # Add everyone in the neighborhood of current_component[cc_ptr]
            rev_bit_idx = reversed_erased_bits[current_component[cc_ptr]]
            for check_idx in bit_adj_lists[rev_bit_idx][:erased_bit_degrees[rev_bit_idx]]:
                if unassigned_checks[check_idx]: # skip checks that are already assigned
                    unassigned_checks[check_idx] = 0 # mark check as assigned to a CC
                    # Now loop over the neighboring bits
                    for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
                        if unassigned_bits[bit_idx]: # bit must be previously unassigned and inside the erasure
                            current_component[current_component_size] = bit_idx
                            current_component_size += 1
                            unassigned_bits[bit_idx] = 0 # add bit to the CC and mark it as assigned

            cc_ptr += 1 # Move on to the next bit already added in the component

        # If the loop ends, no one left to add in the CC, move to the next CC
        # Check if it is the smallest so far
        if current_component_size < smallest_component_size:
            smallest_component_size = current_component_size
            smallest_component[:smallest_component_size] = current_component[:current_component_size].copy()
        current_component_size = 0
    
    return smallest_component[:smallest_component_size]

@numba.njit
def peel_prune_guess(erasure: np.ndarray, Hindptr: np.ndarray, Hindices: np.ndarray, 
                     Hconjindptr: np.ndarray, Hconjindices: np.ndarray, 
                     max_pruning_depth: int=1, max_guesses: int=6) -> tuple[np.ndarray, int, int, tuple[int]]:   
    # Build adjacency list structure for erased bits, and keep track of their count
    erased_bits = np.nonzero(erasure)[0]
    num_erased_bits = len(erased_bits)
    reversed_erased_bits = np.zeros((len(erasure),), dtype=np.int32)
    for i, e in enumerate(erased_bits):
        reversed_erased_bits[e] = i
    bit_adj_lists = np.zeros((num_erased_bits, len(Hindptr)-1), dtype=np.int32)
    erased_bit_degrees = np.zeros((num_erased_bits,), dtype=np.int32)

    # Keep track of the degree of the checks towards the erasure and stack the dangling checks
    check_erasure_degrees = np.zeros((len(Hindptr)-1,), dtype=np.int32)
    dangling_check_stack, stack_size = np.zeros((len(Hindptr)-1,), dtype=np.int32), 0
    
    # Populate the adjacency lists of the erased bits and add the first dangling checks
    for check_idx in range(len(Hindptr)-1):
        for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
            if erasure[bit_idx]:
                check_erasure_degrees[check_idx] += 1
                rev_bit_idx = reversed_erased_bits[bit_idx]
                bit_adj_lists[rev_bit_idx][erased_bit_degrees[rev_bit_idx]] = check_idx
                erased_bit_degrees[rev_bit_idx] += 1

        if check_erasure_degrees[check_idx] == 1:
            dangling_check_stack[stack_size] = check_idx
            stack_size += 1

    # Pruning parameters
    stab_idx = np.zeros((max_pruning_depth,), dtype=np.int32)
    pruning_depth = 1
    
    # Guessing parameters
    guessed_bit_initialization = lambda ng: np.array([(((1<<(1<<ng)) - 1) << (1<<ng))], dtype=np.uint64) if ng < 6 else np.array([0]*(1 << (ng-6)) + [-1]*(1 << (ng-6))).astype(np.uint64)
    num_guesses = 0
    max_used_guesses = 0
    erasure_sizes = np.zeros(max_guesses+1, dtype=np.uint32)
    stopping_sets = np.zeros((max_guesses+1, len(erasure)), dtype=np.uint32)
    stopping_size = 0
    
    # Initialize bit correction and cumulative syndrome branches
    bit_correction = np.zeros((len(erasure), (1 << max(0, max_guesses-6))), dtype=np.uint64)
    check_syndrome = np.zeros((len(Hindptr)-1, (1 << max(0, max_guesses-6))), dtype=np.uint64)
    
    # Main peeling loop
    while num_erased_bits > 0:
        # Find a dangling check
        dangling_check_idx = -1
        # Pop checks whose dangling bit has already been unerased
        while stack_size > 0 and dangling_check_idx == -1:
            dangling_check_idx = dangling_check_stack[stack_size-1]
            stack_size -= 1
            if check_erasure_degrees[dangling_check_idx] == 0:
                # Verify for consistency. 
                # Alternatively, defer consistency verifications to the end, by adding these checks to another stack/queue. 
                # Another idea, in the opposite direction, whenever you decrease the degrees, check if the equation became empty, 
                # and verify its consistency right away
                if np.any(check_syndrome[dangling_check_idx] != 0):
                    assert num_guesses > 0
                    # Select inconsistent branches
                    branch_mask = check_syndrome[dangling_check_idx, :1 << max(0, num_guesses - 6)]
                    # Rearrange branches to free up space
                    bit_correction[:, :1 << max(0, num_guesses - 6)] = rearrange(bit_correction[:, :1 << max(0, num_guesses - 6)], branch_mask)
                    check_syndrome[:, :1 << max(0, num_guesses - 6)] = rearrange(check_syndrome[:, :1 << max(0, num_guesses - 6)], branch_mask)
                    num_guesses -= 1
                    
                    assert np.all(check_syndrome[dangling_check_idx] == 0)
                
                dangling_check_idx = -1

        
        # If no dangling check remains, resort to pruning and guessing
        if dangling_check_idx == -1:
            # TODO: implement pruning for any depth
            found_stabilizer = False
            while pruning_depth <= max_pruning_depth and not found_stabilizer:
                while stab_idx[0] < len(Hconjindptr)-1 and not found_stabilizer:
                    # Verify if this stabilizer is supported within the erasure
                    if erasure[Hconjindices[Hconjindptr[stab_idx[0]]:Hconjindptr[stab_idx[0]+1]]].all():
                        # Choose a bit as a gauge and unerase it
                        # Pick first bit of the support
                        gauge_bit = Hconjindices[Hconjindptr[stab_idx[0]]:Hconjindptr[stab_idx[0]+1]][0]
                        erasure[gauge_bit] = False
                        num_erased_bits -= 1
                        # Decrease the degrees of each neighboring check
                        rev_bit_idx = reversed_erased_bits[gauge_bit]
                        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
                            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
                            check_erasure_degrees[check_idx] -= 1
                            # If a check becomes dangling, add it to the stack
                            if check_erasure_degrees[check_idx] == 1:
                                dangling_check_stack[stack_size] = check_idx
                                stack_size += 1
                        
                        found_stabilizer = True
                    stab_idx[0] += 1

                pruning_depth += stab_idx[0] // (len(Hconjindptr)-1)
                stab_idx[0] %= len(Hconjindptr)-1
                
            if found_stabilizer:
                continue


            if num_guesses == 0:
                stopping_size = np.count_nonzero(erasure)
            # If pruning is exhausted, resort to guessing (if available)
            if num_guesses < max_guesses:
                # Guess score to choose the guessed bit;                 
                # guess_candidates = np.nonzero(erasure)[0]
                guess_candidates = smallest_connected_component(erasure, reversed_erased_bits, bit_adj_lists, erased_bit_degrees, Hindices, Hindptr)
                guess_score = np.zeros((len(guess_candidates),), dtype=np.int32)
                for i, b in enumerate(guess_candidates):
                    rev_bit_idx = reversed_erased_bits[b]
                    for check_idx in bit_adj_lists[rev_bit_idx]:
                        if check_erasure_degrees[check_idx] == 2:
                            guess_score[i] += 1
                guess_bit = guess_candidates[np.argmax(guess_score)]
                # guess_bit = npr.choice(guess_candidates)

                # Guess the correction at guess_bit as being 1 and duplicate other values
                # TODO: think of a more efficient way to do this duplication, i.e., duplicate only the nonempty bits/checks (sparse arrays)
                if num_guesses < 6:
                    bit_correction[:, 0] |= (bit_correction[:, 0] << np.uint64(1<<num_guesses))
                    check_syndrome[:, 0] |= (check_syndrome[:, 0] << np.uint64(1<<num_guesses))
                else:
                    bit_correction[:, (1<<(num_guesses-6)):(1<<(num_guesses+1-6))] = bit_correction[:, :(1<<(num_guesses-6))]
                    check_syndrome[:, (1<<(num_guesses-6)):(1<<(num_guesses+1-6))] = check_syndrome[:, :(1<<(num_guesses-6))]
                
                bit_correction[guess_bit, :(1<<(num_guesses+1-6))] = guessed_bit_initialization(num_guesses)

                # Update guessing budget and record stats
                num_guesses += 1
                if num_guesses > max_used_guesses:
                    erasure_sizes[max_used_guesses] = stopping_size
                    stopping_sets[max_used_guesses] = erasure.copy()
                    max_used_guesses = num_guesses
                
                # Unerase the guessed bit
                assert erasure[guess_bit]
                erasure[guess_bit] = False
                num_erased_bits -= 1
                # Decrease the degrees of each neighboring check
                rev_bit_idx = reversed_erased_bits[guess_bit]
                for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
                    check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
                    check_erasure_degrees[check_idx] -= 1
                    # Propagate branches to check syndromes
                    check_syndrome[check_idx, 0:1 << max(0, num_guesses-6)] ^= bit_correction[guess_bit, 0:1 << max(0, num_guesses-6)]
                    # If a check becomes dangling, add it to the stack
                    if check_erasure_degrees[check_idx] == 1:
                        dangling_check_stack[stack_size] = check_idx
                        stack_size += 1
                
                # Restart the search for dangling checks
                continue
                
            else:
                # Guessing budget exhausted: signal failure
                erasure_sizes[max_used_guesses] = stopping_size
                stopping_sets[max_used_guesses] = erasure.copy()
                return np.ones_like(bit_correction)*np.uint64(-1), num_guesses, max_guesses+1, erasure_sizes, stopping_sets

        # Find the corresponding dangling bit
        for bit_idx in Hindices[Hindptr[dangling_check_idx]:Hindptr[dangling_check_idx+1]]:
            if erasure[bit_idx]:
                dangling_bit_idx = bit_idx
                break
        # Unerase it and decrease the erasure count
        erasure[dangling_bit_idx] = False
        # Solve dangling equation for the dangling variable in all branches
        bit_correction[dangling_bit_idx, 0:1 << max(0, num_guesses-6)] = check_syndrome[dangling_check_idx, 0:1 << max(0, num_guesses-6)]
        num_erased_bits -= 1
        # Decrease the degrees of each neighboring check
        rev_bit_idx = reversed_erased_bits[dangling_bit_idx]
        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
            check_erasure_degrees[check_idx] -= 1
            # Propagate branches to syndromes
            check_syndrome[check_idx, 0:1 << max(0, num_guesses-6)] ^= bit_correction[dangling_bit_idx, 0:1 << max(0, num_guesses-6)]
            # If a check becomes dangling, add it to the stack
            if check_erasure_degrees[check_idx] == 1:
                dangling_check_stack[stack_size] = check_idx
                stack_size += 1

    
    # When no erasures are left, check one last time for inconsistencies
    while stack_size > 0:
        dangling_check_idx = dangling_check_stack[stack_size-1]
        stack_size -= 1
        assert check_erasure_degrees[dangling_check_idx] == 0
        
        # Verify for consistency
        if np.any(check_syndrome[dangling_check_idx] != 0):
            assert num_guesses > 0
            # Select inconsistent branches
            branch_mask = check_syndrome[dangling_check_idx, :1 << max(0, num_guesses - 6)]
            # Rearrange branches to free up space
            bit_correction[:, :1 << max(0, num_guesses - 6)] = rearrange(bit_correction[:, :1 << max(0, num_guesses - 6)], branch_mask)
            check_syndrome[:, :1 << max(0, num_guesses - 6)] = rearrange(check_syndrome[:, :1 << max(0, num_guesses - 6)], branch_mask)
            num_guesses -= 1
            
            assert np.all(check_syndrome[dangling_check_idx] == 0)
    
    # If no erasures left, nor inconsistent equations, check if the alternative corrections are nontrivial (up to a stabilizer)
    assert not np.any(erasure)
    return bit_correction, num_guesses, max_used_guesses, erasure_sizes, stopping_sets

@numba.njit
def symbolic_quantum_maxwell(erasure: np.ndarray, Hindptr: np.ndarray, Hindices: np.ndarray, 
                     Hconjindptr: np.ndarray, Hconjindices: np.ndarray, 
                     max_guesses: int=6, max_pruning_depth: int=1) -> tuple[np.ndarray, int, int, tuple[int]]:   
    # Build adjacency list structure for erased bits, and keep track of their count
    erased_bits = np.nonzero(erasure)[0]
    num_erased_bits = len(erased_bits)
    reversed_erased_bits = np.zeros((len(erasure),), dtype=np.int32)
    for i, e in enumerate(erased_bits):
        reversed_erased_bits[e] = i
    bit_adj_lists = np.zeros((num_erased_bits, len(Hindptr)-1), dtype=np.int32)
    erased_bit_degrees = np.zeros((num_erased_bits,), dtype=np.int32)

    # Keep track of the degree of the checks towards the erasure and stack the dangling checks
    check_erasure_degrees = np.zeros((len(Hindptr)-1,), dtype=np.int32)
    dangling_check_stack, stack_size = np.zeros((len(Hindptr)-1,), dtype=np.int32), 0
    
    # Populate the adjacency lists of the erased bits and add the first dangling checks
    for check_idx in range(len(Hindptr)-1):
        for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
            if erasure[bit_idx]:
                check_erasure_degrees[check_idx] += 1
                rev_bit_idx = reversed_erased_bits[bit_idx]
                bit_adj_lists[rev_bit_idx][erased_bit_degrees[rev_bit_idx]] = check_idx
                erased_bit_degrees[rev_bit_idx] += 1

        if check_erasure_degrees[check_idx] == 1:
            dangling_check_stack[stack_size] = check_idx
            stack_size += 1

    # Pruning parameters
    stab_idx = np.zeros((max_pruning_depth,), dtype=np.int32)
    pruning_depth = 1
    
    # Guessing parameters
    num_guesses, max_used_guesses = 0, 0
    erasure_sizes = np.zeros(max_guesses+1, dtype=np.uint32)
    stopping_size = 0
    stopping_sets = np.zeros((max_guesses+1, len(erasure)), dtype=np.uint32)
    
    # Initialize bit correction and cumulative syndrome branches
    bit_correction = np.zeros((len(erasure), (1 << max(0, max_guesses-6))), dtype=np.uint64)
    check_syndrome = np.zeros((len(Hindptr)-1, (1 << max(0, max_guesses-6))), dtype=np.uint64)
    
    # Main peeling loop
    while num_erased_bits > 0:
        # Find a dangling check
        dangling_check_idx = -1
        # Pop checks whose dangling bit has already been unerased
        while stack_size > 0 and dangling_check_idx == -1:
            dangling_check_idx = dangling_check_stack[stack_size-1]
            stack_size -= 1
            if check_erasure_degrees[dangling_check_idx] == 0:
                # Verify for consistency. 
                # Alternatively, defer consistency verifications to the end, by adding these checks to another stack/queue. 
                # Another idea, in the opposite direction, whenever you decrease the degrees, check if the equation became empty, 
                # and verify its consistency right away
                if np.any(check_syndrome[dangling_check_idx] != 0):
                    assert num_guesses > 0
                    # Select inconsistent branches
                    branch_mask = check_syndrome[dangling_check_idx, :1 << max(0, num_guesses - 6)]
                    # Rearrange branches to free up space
                    bit_correction[:, :1 << max(0, num_guesses - 6)] = rearrange(bit_correction[:, :1 << max(0, num_guesses - 6)], branch_mask)
                    check_syndrome[:, :1 << max(0, num_guesses - 6)] = rearrange(check_syndrome[:, :1 << max(0, num_guesses - 6)], branch_mask)
                    num_guesses -= 1
                    
                    assert np.all(check_syndrome[dangling_check_idx] == 0)
                
                dangling_check_idx = -1

        
        # If no dangling check remains, resort to pruning and guessing
        if dangling_check_idx == -1:
            # TODO: implement pruning for any depth
            found_stabilizer = False
            while pruning_depth <= max_pruning_depth and not found_stabilizer:
                while stab_idx[0] < len(Hconjindptr)-1 and not found_stabilizer:
                    # Verify if this stabilizer is supported within the erasure
                    if erasure[Hconjindices[Hconjindptr[stab_idx[0]]:Hconjindptr[stab_idx[0]+1]]].all():
                        # Choose a bit as a gauge and unerase it
                        # Pick first bit of the support
                        gauge_bit = Hconjindices[Hconjindptr[stab_idx[0]]:Hconjindptr[stab_idx[0]+1]][0]
                        erasure[gauge_bit] = False
                        num_erased_bits -= 1
                        # Decrease the degrees of each neighboring check
                        rev_bit_idx = reversed_erased_bits[gauge_bit]
                        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
                            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
                            check_erasure_degrees[check_idx] -= 1
                            # If a check becomes dangling, add it to the stack
                            if check_erasure_degrees[check_idx] == 1:
                                dangling_check_stack[stack_size] = check_idx
                                stack_size += 1
                        
                        found_stabilizer = True
                    stab_idx[0] += 1

                pruning_depth += stab_idx[0] // (len(Hconjindptr)-1)
                stab_idx[0] %= len(Hconjindptr)-1
                
            if found_stabilizer:
                continue


            if num_guesses == 0:
                stopping_size = np.count_nonzero(erasure)
            # If pruning is exhausted, resort to guessing (if available)
            if num_guesses < max_guesses:
                # Guess score to choose the guessed bit;                 
                # guess_candidates = np.nonzero(erasure)[0]
                guess_candidates = smallest_connected_component(erasure, reversed_erased_bits, bit_adj_lists, erased_bit_degrees, Hindices, Hindptr)
                guess_score = np.zeros((len(guess_candidates),), dtype=np.int32)
                for i, b in enumerate(guess_candidates):
                    rev_bit_idx = reversed_erased_bits[b]
                    for check_idx in bit_adj_lists[rev_bit_idx]:
                        if check_erasure_degrees[check_idx] == 2:
                            guess_score[i] += 1
                guess_bit = guess_candidates[np.argmax(guess_score)]
                # guess_bit = npr.choice(guess_candidates)

                # Guess the correction at guess_bit as being 1 and duplicate other values
                # TODO: think of a more efficient way to do this duplication, i.e., duplicate only the nonempty bits/checks (sparse arrays)
                if num_guesses < 6:
                    bit_correction[:, 0] |= (bit_correction[:, 0] << np.uint64(1<<num_guesses))
                    check_syndrome[:, 0] |= (check_syndrome[:, 0] << np.uint64(1<<num_guesses))
                else:
                    bit_correction[:, (1<<(num_guesses-6)):(1<<(num_guesses+1-6))] = bit_correction[:, :(1<<(num_guesses-6))]
                    check_syndrome[:, (1<<(num_guesses-6)):(1<<(num_guesses+1-6))] = check_syndrome[:, :(1<<(num_guesses-6))]
                
                bit_correction[guess_bit, :(1<<(num_guesses+1-6))] = guessed_bit_initialization(num_guesses)

                # Update guessing budget and record stats
                num_guesses += 1
                if num_guesses > max_used_guesses:
                    erasure_sizes[max_used_guesses] = stopping_size
                    stopping_sets[max_used_guesses] = erasure.copy()
                    max_used_guesses = num_guesses
                
                # Unerase the guessed bit
                assert erasure[guess_bit]
                erasure[guess_bit] = False
                num_erased_bits -= 1
                # Decrease the degrees of each neighboring check
                rev_bit_idx = reversed_erased_bits[guess_bit]
                for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
                    check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
                    check_erasure_degrees[check_idx] -= 1
                    # Propagate branches to check syndromes
                    check_syndrome[check_idx, 0:1 << max(0, num_guesses-6)] ^= bit_correction[guess_bit, 0:1 << max(0, num_guesses-6)]
                    # If a check becomes dangling, add it to the stack
                    if check_erasure_degrees[check_idx] == 1:
                        dangling_check_stack[stack_size] = check_idx
                        stack_size += 1
                
                # Restart the search for dangling checks
                continue
                
            else:
                # Guessing budget exhausted: signal failure
                erasure_sizes[max_used_guesses] = stopping_size
                stopping_sets[max_used_guesses] = erasure.copy()
                return np.ones_like(bit_correction)*np.uint64(-1), num_guesses, max_guesses+1, erasure_sizes, stopping_sets

        # Find the corresponding dangling bit
        for bit_idx in Hindices[Hindptr[dangling_check_idx]:Hindptr[dangling_check_idx+1]]:
            if erasure[bit_idx]:
                dangling_bit_idx = bit_idx
                break
        # Unerase it and decrease the erasure count
        erasure[dangling_bit_idx] = False
        # Solve dangling equation for the dangling variable in all branches
        bit_correction[dangling_bit_idx, 0:1 << max(0, num_guesses-6)] = check_syndrome[dangling_check_idx, 0:1 << max(0, num_guesses-6)]
        num_erased_bits -= 1
        # Decrease the degrees of each neighboring check
        rev_bit_idx = reversed_erased_bits[dangling_bit_idx]
        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
            check_erasure_degrees[check_idx] -= 1
            # Propagate branches to syndromes
            check_syndrome[check_idx, 0:1 << max(0, num_guesses-6)] ^= bit_correction[dangling_bit_idx, 0:1 << max(0, num_guesses-6)]
            # If a check becomes dangling, add it to the stack
            if check_erasure_degrees[check_idx] == 1:
                dangling_check_stack[stack_size] = check_idx
                stack_size += 1

    
    # When no erasures are left, check one last time for inconsistencies
    while stack_size > 0:
        dangling_check_idx = dangling_check_stack[stack_size-1]
        stack_size -= 1
        assert check_erasure_degrees[dangling_check_idx] == 0
        
        # Verify for consistency
        if np.any(check_syndrome[dangling_check_idx] != 0):
            assert num_guesses > 0
            # Select inconsistent branches
            branch_mask = check_syndrome[dangling_check_idx, :1 << max(0, num_guesses - 6)]
            # Rearrange branches to free up space
            bit_correction[:, :1 << max(0, num_guesses - 6)] = rearrange(bit_correction[:, :1 << max(0, num_guesses - 6)], branch_mask)
            check_syndrome[:, :1 << max(0, num_guesses - 6)] = rearrange(check_syndrome[:, :1 << max(0, num_guesses - 6)], branch_mask)
            num_guesses -= 1
            
            assert np.all(check_syndrome[dangling_check_idx] == 0)
    
    # If no erasures left, nor inconsistent equations, check if the alternative corrections are nontrivial (up to a stabilizer)
    assert not np.any(erasure)
    return bit_correction, num_guesses, max_used_guesses, erasure_sizes, stopping_sets



def peel(erasure: np.array, H: sp.csr_array) -> bool:
    # return _peel(erasure, H)
    return _fast_peel(erasure, H.indptr, H.indices)


def MC_peeling_classic(num_trials: int, state: nx.MultiGraph, p_vals: list[float]) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(state, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    N = len(v)

    results = {'mean': [], 'std': []}
    if len(p_vals) > 1:
        iterator = tqdm(p_vals)
    else:
        iterator = p_vals

    for erasure_rate in iterator:
        failures = 0
        for _ in range(num_trials):
            erasure = npr.rand(N) < erasure_rate
            if not peel(erasure, H):
                failures += 1
        
        mean, std = failures/num_trials, ((failures*(num_trials - failures)) / (num_trials*(num_trials - 1)))**.5
        results['mean'].append(mean)
        results['std'].append(std)

    results['mean'] = np.array(results['mean'])
    results['std'] = np.array(results['std'])
    
    return results


def HGP(H1: sp.csr_array, H2: sp.csr_array=None):
    # Convention: H1 is the vertical axis, H2 is the horizontal axis
    # BB | BC (Z stab)
    # CB | CC
    # (X stab)
    if H2 is None:
        H2 = H1
    H1 = H1.astype(np.uint)
    H2 = H2.astype(np.uint)
    (m1, n1), (m2, n2) = H1.shape, H2.shape
    I = lambda n: sp.eye_array(n, dtype=np.uint)
    rows, cols, data = sp.find(sp.hstack([sp.kron(I(n1), H2), sp.kron(H1.T, I(m2))]))
    Hz = sp.csr_array((data, (rows, cols))) # make sure that Hz.data contains only 1s, and no nasty 0s
    rows, cols, data = sp.find(sp.hstack([sp.kron(H1, I(n2)), sp.kron(I(m1), H2.T)]))
    Hx = sp.csr_array((data, (rows, cols))) # make sure that Hx.data contains only 1s, and no nasty 0s
    return Hx, Hz

# @numba.njit
def _HGP_peel(erasure: np.array, Hx: sp.csr_array, Hz: sp.csr_array=None, only_X:bool=True) -> bool:
    if Hz is None:
        assert only_X

    peel_X = peel(erasure.copy(), Hz)

    if only_X:
        return peel_X
    else:
        peel_Z = peel(erasure.copy(), Hx)
        return peel_X and peel_Z
    

# @numba.njit
def _MC_peeling_HGP(num_trials: int, Hx: sp.csr_array, Hz: sp.csr_array, p_vals: list[float]) -> dict[str, np.ndarray]:
    N = Hx.shape[1]

    results = {'mean': np.zeros((len(p_vals),)), 'std': np.zeros((len(p_vals),))}
    for t, erasure_rate in enumerate(p_vals):
        failures = 0
        for _ in range(num_trials):
            erasure = npr.rand(N) < erasure_rate
            if not _HGP_peel(erasure, Hx, Hz, only_X=False):
                failures += 1

        mean, std = failures/num_trials, ((failures*(num_trials - failures)) / (num_trials*(num_trials - 1)))**.5
        results['mean'][t] = mean
        results['std'][t] = std

    return results

def MC_peeling_HGP(num_trials: int, state: nx.MultiGraph, p_vals: list[float]) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(state, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    Hx, Hz = HGP(H)
    return _MC_peeling_HGP(num_trials, Hx, Hz, p_vals)

# @numba.jit
def stabilizer_search(H: np.ndarray, erasure: np.ndarray, depth: int, start: int=None) -> tuple[list[bool], np.ndarray, int]:
    m, _ = H.shape

    wt_start = 1 if start is None else sum((start >> i) & 1 for i in range(m))

    found_at_depth = [False]*(wt_start-1) + [True]*(depth - wt_start + 1)
    
    for weight in range(wt_start, depth+1):
        # Smallest subset of size 'weight'
        c = (1 << weight) - 1 if start is None or weight > wt_start else start
        while c < (1 << m):
            # Convert 'c' to a binary representation as a NumPy array
            row_combination = np.array([(c >> i) & 1 for i in range(m)], dtype=bool) # , dtype=np.float32
            stabilizer = np.bitwise_xor.reduce(H[row_combination, :], axis=0, dtype=bool)
            
            # Check if corresponding stabilizer lies within the erasure
            if np.any(stabilizer) and not np.any(stabilizer >> erasure):
                return found_at_depth, stabilizer, long_gosper_next(c) if long_gosper_next(c) < (1<<m) else (1 << (weight+1)) - 1

            c = long_gosper_next(c)  # Get next subset using Gosper's hack

        found_at_depth[weight-1] = False
        # print(f'Failed at depth {weight}! SS size = {np.count_nonzero(erasure)}')
        
    return found_at_depth, np.zeros_like(erasure), None