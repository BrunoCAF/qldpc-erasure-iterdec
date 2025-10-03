import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import pym4ri

import h5py
import argparse
import time
from tqdm import tqdm

from css_code_eval import peel_prune_guess
from experiments_settings import code_distance, code_dimension, smallest_SS_weight

class PeelingGuessingBenchmark:
    def __init__(self, Hx: sp.csr_array, Hz: sp.csr_array):
        self.Hx, self.Hz = Hx, Hz
        assert Hx.shape[1] == Hz.shape[1], "inconsistent number of data qubits"
        self.N = Hx.shape[1]
        self.eta_Hx = pym4ri.gen2chk(self.Hx.astype(bool).todense())
        self.eta_Hz = pym4ri.gen2chk(self.Hz.astype(bool).todense())

    def peel_prune_guess_wrapped(self, erasure: np.ndarray, max_pruning_depth: int, only_X: bool=False) -> list[bool]:
        """
        Peeling+pruning+branching or guessing. 
        Pruning at depth at most 1, but generalizable. 
        Up to 6 guesses available, but will be further generalized. 
        """
        # X-type error:
        erasure_X = erasure.copy()

        # Try to solve until both peeling+branching fails or succeeds
        branching_X_success = self._peel_prune_guess_wrapped(erasure_X, self.Hz, self.Hx, self.eta_Hx, max_pruning_depth, 'X')
            
        if only_X or not branching_X_success[-1]:
            return branching_X_success
        else:
            # Z-type error: 
            erasure_Z = erasure.copy()

            # Try to solve until both peeling+branching fails or succeeds
            branching_Z_success = self._peel_prune_guess_wrapped(erasure_Z, self.Hx, self.Hz, self.eta_Hz, max_pruning_depth, 'Z')
            return [bXs and bZs for bXs, bZs in zip(branching_X_success, branching_Z_success)]

    def _peel_prune_guess_wrapped(self, erasure: np.ndarray, std_H: sp.csr_array, std_Hconj: sp.csr_array, 
                                  eta_H_conj: np.ndarray, max_pruning_depth: int, x_or_z: str) -> list[bool]:
        ML_success = {'X': self.ML_X_success, 'Z': self.ML_Z_success}[x_or_z]
        initial_erasure = erasure.copy()
        correction_u64, num_guesses, max_used_guesses, erasure_sizes = peel_prune_guess(erasure, std_H.indptr, std_H.indices, 
                                                                                        std_Hconj.indptr, std_Hconj.indices, max_pruning_depth)
        num_branches = 1<<num_guesses
        correction = np.unpackbits(correction_u64.view(np.uint8).reshape(-1, 8), axis=1, bitorder='little').astype(bool)[:, :num_branches]

        if ML_success:
            for g in range(6+1):
                if g < max_used_guesses:
                    self.min_SS_size['branch'][g] = min(self.min_SS_size['branch'][g], erasure_sizes[g])
                    self.avg_SS_size['branch'][g].append(erasure_sizes[g])
                    
        if np.any(correction > initial_erasure[:, None]):
            # Branching failure
            return [False]*7
        else:
            assert not np.any(erasure)
            
        succ = not np.any(pym4ri.gf2_mul(eta_H_conj[:, initial_erasure], correction[initial_erasure, :]))

        if not succ:
            assert not ML_success
            
        return ([False]*max_used_guesses + [True]*(7-max_used_guesses)) if succ else [False]*7


    def peel_guess_benchmark(self, p_vals:np.ndarray[float], max_num_trials:int, min_num_fails:int=100, pruning_depth: int=1,
                          only_X:bool=False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
        N = self.N
        
        mean = lambda n_fail, n_total: n_fail/max(1, n_total)
        std = lambda n_fail, n_total: ((n_fail*(n_total - n_fail)) / (n_total*(n_total - 1)))**.5
        eb = lambda n_fail, n_total: 1.96*std(n_fail, n_total)/np.sqrt(n_total)
        
        ML_stats = {"ler": np.zeros_like(p_vals), "ler_eb": np.zeros_like(p_vals)}
        ML_failures = np.zeros_like(p_vals)
        
        branching_stats = {g: {"ler": np.zeros_like(p_vals), "ler_eb": np.zeros_like(p_vals)} for g in range(6+1)}
        branching_failures = {g: np.zeros_like(p_vals) for g in range(6+1)}

        self.min_SS_size = {"branch": [N]*7}
        self.avg_SS_size_by_er = [{"branch": [[] for _ in range(7)]} for _ in range(len(p_vals))]
        
        for i, er in enumerate(p_vals):
            num_trials = max_num_trials
            self.avg_SS_size = self.avg_SS_size_by_er[i]
            for t in (pbar := tqdm(range(max_num_trials), miniters=1e5, maxinterval=np.infty)):
                descr = f'#failures: Branching+Pruning(D={pruning_depth}) '
                for g, branching_failures_g in branching_failures.items():
                    descr += f'G={g}: {branching_failures_g[i]:.0f} | '
                pbar.set_description(descr)

                erasure = npr.rand(N) < er
                
                # Precompute ML_success
                gamma_Hz_E = pym4ri.chk2gen(self.Hz[:, erasure].astype(bool).todense())
                eta_Hx_E = self.eta_Hx[:, erasure]
                self.ML_X_success = not np.any(pym4ri.gf2_mul(eta_Hx_E, gamma_Hz_E))
                gamma_Hx_E = pym4ri.chk2gen(self.Hx[:, erasure].astype(bool).todense())
                eta_Hz_E = self.eta_Hz[:, erasure]
                self.ML_Z_success = not np.any(pym4ri.gf2_mul(eta_Hz_E, gamma_Hx_E))
                self.ML_success = self.ML_X_success and self.ML_Z_success
                
                assert pruning_depth <= 1
                branching_success = self.peel_prune_guess_wrapped(erasure.copy(), max_pruning_depth=pruning_depth, only_X=only_X)

                # Sanity check:
                if self.ML_success < branching_success[-1]:
                    print(f"wtf??? {self.ML_success = }, {branching_success = }")
                    print(np.nonzero(erasure)[0])

                ML_failures[i] += 1 if not self.ML_success else 0
                for branching_failures_g, branching_success_g in zip(branching_failures.values(), branching_success):
                    branching_failures_g[i] += 1 if not branching_success_g else 0              
                
                if branching_failures[6][i] >= min_num_fails:
                    num_trials = t+1
                    break
                        
            ML_stats["ler"][i] = mean(ML_failures[i], num_trials)
            ML_stats["ler_eb"][i] = eb(ML_failures[i], num_trials)
            
            for branching_stats_g, branching_failures_g in zip(branching_stats.values(), branching_failures.values()):
                branching_stats_g["ler"][i] = mean(branching_failures_g[i], num_trials)
                branching_stats_g["ler_eb"][i] = eb(branching_failures_g[i], num_trials)
            
            for g in range(6+1):
                self.avg_SS_size_by_er[i]["branch"][g] = mean(sum(self.avg_SS_size_by_er[i]["branch"][g]), len(self.avg_SS_size_by_er[i]["branch"][g]))

        self.avg_SS_size_by_er = {
            "branch": np.stack([np.array(self.avg_SS_size_by_er[i]["branch"]) for i in range(len(p_vals))]), 
        }
        
        return ML_stats, branching_stats


def load_from_mtx(filename, path='../QT_codes/'):
    m, n, E = np.loadtxt(path+filename, skiprows=2, max_rows=1, dtype=np.uint)
    data = np.loadtxt(path+filename, skiprows=3, max_rows=E, dtype=np.uint)
    data[:, :2] -= 1
    
    return sp.csr_array((data[:, 2], (data[:, 0], data[:, 1])), shape=(m, n))


filenames = lambda XZ, C: f'H{XZ}_k{2*(C+1)}_w8.mtx'

# Tune the physical erasure rates and the MC budget according to the code/decoder
p_vals = np.linspace(0.05, 0.45, 15)
MC_budget = {
    "peel": {
        0: [int(1e7)]*15,
        1: [int(1e7)]*15,
        2: [int(1e7)]*15,
    }, 
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-E', action="store", dest='E', default=0, type=int, required=True)
    parser.add_argument('-D', action="store", dest='D', default=0, type=int)
    parser.add_argument('-M', action="store", dest='MC', default=None, type=int)
    args = parser.parse_args()

    # Choose the codelength,  error rate, decoder and MC budget
    C, E, D = args.C, args.E, args.D
    if args.MC is None:
        MC_peel = MC_budget['peel'][C][E]
    else:
        MC_peel = args.MC
    print(f'Script configuration: {C = }, {E = }, {D = }, {MC_peel = }')

    # Set code
    Hx = load_from_mtx(filenames('X', C))
    Hz = load_from_mtx(filenames('Z', C))
    
    # Compute code params
    assert Hx.shape[1] == Hz.shape[1]
    N = Hx.shape[1]
    rx, rz = Hx.shape[1] - code_dimension(Hx), Hz.shape[1] - code_dimension(Hz)
    K = N - rx - rz
    print(f'Quantum HGP code params: [[N={N}, K={K}]]')
    params = {'N': N, 'K': K}

    # Set simulation parameters
    er = p_vals[E]
    print(f'Error rate: {er:.3f} | MC budget (peel): 10^{np.log10(MC_peel):.0f} trials')

    # Run simulation
    code_bm_obj = PeelingGuessingBenchmark(Hx, Hz)
    print('Running peeling decoding benchmark...')
    t0 = time.time()
    ML_stats, branching_stats = code_bm_obj.peel_guess_benchmark([er], max_num_trials=MC_peel, pruning_depth=D)
    dt = time.time() - t0
    print(f'Peeling done in {dt:.3f} s')
    print(f'{code_bm_obj.min_SS_size = }')


    # Save results
    print('Simulation done, saving results...')    
    time.sleep(E)
    with h5py.File("peeling_guessing_QT_benchmark.hdf5", "a") as f: 
        grp = f.require_group(f'[[{N},{K}]]')
        
        # Store [[N, K, Dbound]] params
        for par, val in params.items():
            grp.attrs[str(par)] = val

        for g, s in enumerate(code_bm_obj.min_SS_size["branch"]):
            grp.attrs[f"minSS_branch_d={D}_g={g}"] = min(grp.attrs[f"minSS_branch_d={D}_g={g}"], s) if f"minSS_branch_d={D}_g={g}" in grp.attrs else s
        
        subgrp = grp.require_group(f'ER={E}')
        
        # Store LER statistics for each decoder
        if D == 1:
            subgrp.create_dataset("ML_ler", data=ML_stats['ler'])
            subgrp.create_dataset("ML_eb", data=ML_stats['ler_eb'])

        for g, branching_stats_g in branching_stats.items():
            subgrp.create_dataset(f"branching_d{D}_g{g}_ler", data=branching_stats_g['ler'])
            subgrp.create_dataset(f"branching_d{D}_g{g}_eb", data=branching_stats_g['ler_eb'])

        # Store SS size statistics for each decoder
        subgrp.create_dataset(f"avgSS_branch_d={D}", data=code_bm_obj.avg_SS_size_by_er["branch"].flatten())
        
    
    print('Results saved. All done.')