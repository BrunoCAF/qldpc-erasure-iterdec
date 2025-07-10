import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import networkx.algorithms.bipartite as bpt

import h5py
import argparse
import time

from css_code_eval import MC_erasure_plog
from experiments_settings import code_dimension, code_distance, smallest_SS_weight
from tanner_codes import TannerCode, TannerCodeHGP

import pickle

names = ["PEG_codes", "SA_codes", "PS_codes", "PE_codes"]
objs = []
for name in names:
    with open(name+'.pkl', 'rb') as f:
        objs.append(pickle.load(f))

codes = ['[625,25]', '[1225,62]', '[1600,64]', '[2025,81]']


p_vals = np.linspace(0.10, 0.3, 10)
MC_budget = {
    "peel": {
        0: [int(1e5)]*2 + [int(1e5)]*3 + [int(1e4)]*5, 
        1: [int(1e5)]*2 + [int(1e5)]*3 + [int(1e4)]*5, 
        2: [int(1e5)]*2 + [int(1e5)]*3 + [int(1e4)]*5, 
        3: [int(1e3)]*2 + [int(1e5)]*3 + [int(1e5)]*5, 
    }, 
    "ML": {
        0: [int(1e6)]*2 + [int(1e5)]*3 + [int(1e4)]*5, 
        1: [int(1e6)]*2 + [int(1e5)]*3 + [int(1e4)]*5, 
        2: [int(1e6)]*2 + [int(1e5)]*3 + [int(1e4)]*5, 
        3: [int(1e3)]*2 + [int(1e6)]*3 + [int(1e5)]*5, 
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-A', action="store", dest='A', default=0, type=int, required=True)
    parser.add_argument('-E', action="store", dest='E', default=0, type=int, required=True)
    parser.add_argument('-D', action="store", dest='D', default=0, type=int)
    parser.add_argument('-M', action="store", dest='MC', default=None, type=int)
    args = parser.parse_args()

    # Choose the codelength, optimization algorithm, error rate, decoder and MC budget
    C, A, E, D = args.C, args.A, args.E, args.D
    if args.MC is None:
        MC_ML = MC_budget['ML'][C][E]
        MC_peel = MC_budget['peel'][C][E]
    else:
        MC_ML = MC_peel = args.MC
    print(f'Script configuration: {C = }, {A = }, {E = }, {D = }, {MC_ML = }, {MC_peel = }')
    print(f'Code length: {codes[C].split('[')[1].split(',')[0]}; Optimization algorithm: {names[A].split('_')[0]}; ')

    # Set code
    code = sp.csr_array(objs[A][codes[C]].astype(np.uint8).todense() & 1)
    m, _ = code.shape
    tanner_code_classic = TannerCode.from_standard_code(code, [np.arange(m)])
    
    # Compute code params
    if E == 9 and D == 0:
        n, k, d, (s, _, _) = code.shape[1], code_dimension(code), code_distance(code), smallest_SS_weight(code)
        nt, kt, dt, (st, _, _) = code.shape[0], code_dimension(code.T), code_distance(code.T), smallest_SS_weight(code.T)
        print(f'Classical (base) code params: [{n=}, {k=}, {d=}, {s=}], [n^t={nt}, k^t={kt}, d^t={dt}, s^t={st}]')    
        print(f'Quantum HGP code params: [[N={n*n+nt*nt}, K={k*k+kt*kt}, D={min(d, dt)}]]')
        params = {
            'n': n, 'k': k, 'd': d, 's': s, 
            'nt': nt, 'kt': kt, 'dt': dt, 'st': st, 
            'N': n*n+nt*nt, 'K': k*k+kt*kt, 'D': min(d, dt), 
        }
    else:
        n, k, d = code.shape[1], code_dimension(code), code_distance(code)
        print('For code params, check output of the script with E=[last], D=0')
        params = {}

    # Set simulation parameters
    er = p_vals[E]
    print(f'Error rate: {er:.3f} | MC budget (ML): 10^{np.log10(MC_ML):.0f} trials; (peel): 10^{np.log10(MC_peel):.0f} trials')

    # Set decoder/cost function -> do both ML and peeling
    if D == 0:
        theta = bpt.from_biadjacency_matrix(code, create_using=nx.MultiGraph)
        print('Running ML decoding benchmark...')    
        t0 = time.time()
        ML_results = MC_erasure_plog(MC_ML, state=theta, p_vals=[er])
        dt = time.time() - t0
        print(f'ML done in {dt:.3f} s')

    
    # Run simulation
    tanner_code_hgp = TannerCodeHGP(tanner_code_classic)
    print('Running peeling decoding benchmark...')
    t0 = time.time()
    _, branching_stats = tanner_code_hgp.peel_guess_benchmark([er], max_num_trials=MC_peel, pruning_depth=D)
    dt = time.time() - t0
    print(f'Peeling done in {dt:.3f} s')
    print(f'{tanner_code_hgp.min_SS_size = }')


    # Save results
    print('Simulation done, saving results...')    
    time.sleep(E)
    with h5py.File("peeling_guessing_benchmark.hdf5", "a") as f: 
        grp = f.require_group(f'[{n},{k},{d}]')
        
        # Store [n, k, d, s], [nt, kt, dt, st], [[N, K, D]] params
        for par, val in params.items():
            grp.attrs[str(par)] = val

        for g, s in enumerate(tanner_code_hgp.min_SS_size["branch"]):
            grp.attrs[f"minSS_branch_d={D}_g={g}"] = min(grp.attrs[f"minSS_branch_d={D}_g={g}"], s) if f"minSS_branch_d={D}_g={g}" in grp.attrs else s
        
        subgrp = grp.require_group(f'ER={E}')
        
        # Store LER statistics for each decoder
        if D == 0:
            subgrp.create_dataset("ML_ler", data=ML_results['mean'])
            subgrp.create_dataset("ML_eb", data=1.96*ML_results['std']/np.sqrt(MC_ML))

        for g, branching_stats_g in branching_stats.items():
            subgrp.create_dataset(f"branching_d{D}_g{g}_ler", data=branching_stats_g['ler'])
            subgrp.create_dataset(f"branching_d{D}_g{g}_eb", data=branching_stats_g['ler_eb'])

        # Store SS size statistics for each decoder
        subgrp.create_dataset(f"avgSS_branch_d={D}", data=tanner_code_hgp.avg_SS_size_by_er["branch"].flatten())
        
    
    print('Results saved. All done.')