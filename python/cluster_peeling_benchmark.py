import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import pym4ri

import h5py
import argparse
import time
import pickle
from tqdm import tqdm

from sympy.abc import x, y
from qldpc.codes import BBCode

from experiments_settings import code_distance, code_dimension, smallest_SS_weight
from css_code_eval import HGP
from cluster_decoder import cluster_decoder

class ClusterPeelingBenchmark:
    def __init__(self, Hx: sp.csr_array, Hz: sp.csr_array):
        self.Hx, self.Hz = Hx, Hz
        assert Hx.shape[1] == Hz.shape[1], "inconsistent number of data qubits"
        self.N = Hx.shape[1]
        self.eta_Hx = pym4ri.gen2chk(self.Hx.astype(bool).todense())
        self.eta_Hz = pym4ri.gen2chk(self.Hz.astype(bool).todense())

    def cluster_decode_wrapped(self, erasure: np.ndarray, C_sizes: list[int], only_X: bool=False) -> list[bool]:
        """
        Wrapper function for Hanwen Yao's cluster peeling decoder. 
        """
        # X-type error:
        erasure_X = erasure.copy()

        # Try to solve until both peeling+clustering fails or succeeds
        cluster_X_success = self._cluster_decode_wrapped(erasure_X, self.Hz, self.eta_Hx, C_sizes)
            
        if only_X or not cluster_X_success[-1]:
            return cluster_X_success
        else:
            # Z-type error: 
            erasure_Z = erasure.copy()

            # Try to solve until both peeling+clustering fails or succeeds
            cluster_Z_success = self._cluster_decode_wrapped(erasure_Z, self.Hx, self.eta_Hz, C_sizes)
            return [cXs and cZs for cXs, cZs in zip(cluster_X_success, cluster_Z_success)]

    def _cluster_decode_wrapped(self, erasure: np.ndarray, std_H: sp.csr_array, eta_H_conj: np.ndarray, C_sizes: list[int]) -> list[bool]:
        
        H_coo = sp.coo_array(std_H)
        ClusterDecoderObj = cluster_decoder(*std_H.shape[::-1], H_coo.row, H_coo.col)
        
        initial_erasure = erasure.copy()
        error = erasure & (npr.rand(self.N) < 0.5)
        syndrome = (std_H @ error) & 1

        decoded_err, max_C_size, *_ = ClusterDecoderObj.decode(erasure, syndrome, C_sizes)
        
        if decoded_err is None:
            return [False for _ in C_sizes]
        else:
            residual_err = (decoded_err ^ error.astype(np.int64)).astype(bool)
            is_stab = not np.any(pym4ri.gf2_mul(eta_H_conj[:, initial_erasure], residual_err.reshape(-1, 1)[initial_erasure, :]))
            return [(max_C_size <= C) and is_stab for C in C_sizes]

    def cluster_peeling_benchmark(self, p_vals: np.ndarray[float], max_num_trials: int, min_num_fails: int=100, C_sizes: list[int]=None,
                                  only_X: bool=False) -> dict[str, np.ndarray]:
        N = self.N
        if C_sizes is None:
            C_sizes = [N]
        
        mean = lambda n_fail, n_total: n_fail/max(1, n_total)
        std = lambda n_fail, n_total: ((n_fail*(n_total - n_fail)) / (n_total*(n_total - 1)))**.5
        eb = lambda n_fail, n_total: 1.96*std(n_fail, n_total)/np.sqrt(n_total)
        
        cluster_stats = {C: {"ler": np.zeros_like(p_vals), "ler_eb": np.zeros_like(p_vals)} for C in C_sizes}
        cluster_failures = {C: np.zeros_like(p_vals) for C in C_sizes}

        # self.min_SS_size = {"cluster": [N for _ in C_sizes]}
        # self.avg_SS_size_by_er = [{"cluster": [[] for _ in C_sizes]} for _ in range(len(p_vals))]
        
        for i, er in enumerate(p_vals):
            num_trials = max_num_trials
            # self.avg_SS_size = self.avg_SS_size_by_er[i]
            for t in (pbar := tqdm(range(max_num_trials), disable=False, miniters=100, mininterval=15, maxinterval=np.infty)):
                if t // 100 == 1:
                    descr = f'#fails '
                    for C, cluster_failures_C in cluster_failures.items():
                        descr += f'C={C}: {cluster_failures_C[i]:.0f} | '
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
                
                cluster_success = self.cluster_decode_wrapped(erasure.copy(), C_sizes=C_sizes, only_X=only_X)

                # Adjust failure criterion:
                cluster_success = [cs and self.ML_success for cs in cluster_success]

                for cluster_failures_C, cluster_success_C in zip(cluster_failures.values(), cluster_success):
                    cluster_failures_C[i] += 1 if not cluster_success_C else 0              
                
                if min(v[0] for k, v in cluster_failures.items()) >= min_num_fails:
                    num_trials = t+1
                    break
                        
            for cluster_stats_C, cluster_failures_C in zip(cluster_stats.values(), cluster_failures.values()):
                cluster_stats_C["ler"][i] = mean(cluster_failures_C[i], num_trials)
                cluster_stats_C["ler_eb"][i] = eb(cluster_failures_C[i], num_trials)
            
            # for C in C_sizes:
            #     self.avg_SS_size_by_er[i]["cluster"][C] = mean(sum(self.avg_SS_size_by_er[i]["cluster"][C]), len(self.avg_SS_size_by_er[i]["cluster"][C]))

        # self.avg_SS_size_by_er = {
        #     "cluster": np.stack([np.array(self.avg_SS_size_by_er[i]["cluster"]) for i in range(len(p_vals))]), 
        # }
        
        return cluster_stats


def load_from_mtx(filename, path='../QT_codes/'):
    m, n, E = np.loadtxt(path+filename, skiprows=2, max_rows=1, dtype=np.uint)
    data = np.loadtxt(path+filename, skiprows=3, max_rows=E, dtype=np.uint)
    data[:, :2] -= 1
    
    return sp.csr_array((data[:, 2], (data[:, 0], data[:, 1])), shape=(m, n))


filenames = lambda XZ, C: f'H{XZ}_k{2*(C+1)}_w8.mtx'
# filenames = lambda XZ, C: f'H{XZ}_k{2*(C+1)}_w8.mtx' if C < 3 else {3: f'H{XZ}_C4rC4_ord16_trial13_B6_perm4.mtx', 
#                                                                     4: f'H{XZ}_C4rC4_ord16_trial13_B6_perm4_redundant.mtx',
#                                                                     5: f'H{XZ}_576_24_24.mtx',
#                                                                     6: f'H{XZ}_576_24_24_redundant.mtx'}[C]

orders = [(12, 12), (30, 6), (21, 18)]
# orders = [(6, 6), (15, 3), (9, 6), (12, 6), (12, 12), (30, 6), (21, 18)]
polys = [
    # {"A": x**3 + y + y**2, "B": y**3 + x + x**2}, 
    # {"A": x**9 + y + y**2, "B": 1 + x**2 + x**7},  
    # {"A": x**3 + y + y**2, "B": y**3 + x + x**2}, 
    # {"A": x**3 + y + y**2, "B": y**3 + x + x**2}, 
    {"A": x**3 + y**2 + y**7, "B": y**3 + x + x**2}, 
    {"A": x**9 + y + y**2, "B": y**3 + x**25 + x**26},  
    {"A": x**3 + y**10 + y**17, "B": y**5 + x**3 + x**19}, 
]

names = ["SA_codes"]
# names = ["PEG_codes", "SA_codes", "PS_codes", "PE_codes"]
objs = []
for name in names:
    with open(name+'.pkl', 'rb') as f:
        objs.append(pickle.load(f))

codes = ['[625,25]', '[1600,64]', '[2025,81]']
# codes = ['[625,25]', '[1225,65]', '[1600,64]', '[2025,81]']

labels = {
    0: [
        '[625,25]', '[1600,64]', '[2025,81]',
    ], 
    1: [
        '[[288,12]]', '[[360,12]]', '[[756,16]]',
        # '[[72,12]]', '[[90,8]]', '[[108,8]]', '[[144,12]]', '[[288,12]]', '[[360,12]]', '[[756,16]]',
    ],
    2: [
        '[[496,2]]', '[[496,4]]', '[[496,6]]',
        # '[[496,2]]', '[[496,4]]', '[[496,6]]', '[[576,24,24]]', '[[576,24,24]]-reg', '[[576,24]]', '[[576,24]]-reg',
    ],
}

# Tune the physical erasure rates and the MC budget according to the code/decoder
p_vals = np.linspace(0.05, 0.45, 15)
MC_budget = [int(1e6)]*15
# MC_budget = {
#     0: [int(1e7)]*15,
#     1: [int(1e7)]*15,
#     2: [int(1e7)]*15,
#     3: [int(1e7)]*15,
#     4: [int(1e7)]*15,
#     5: [int(1e7)]*15,
#     6: [int(1e7)]*15,
# }

C_sizes_param = [10, 20, 50, 100]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', action="store", dest='F', default=0, type=int)
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-E', action="store", dest='E', default=0, type=int, required=True)
    parser.add_argument('-M', action="store", dest='MC', default=None, type=int)
    args = parser.parse_args()

    # Choose the code family, codelength, error rate and MC budget
    F, C, E = args.F, args.C, args.E
    if args.MC is None:
        MC = MC_budget[E]
        # MC = MC_budget[F][C][E]
    else:
        MC = args.MC
    print(f'Script configuration: {['HGP', 'BB', 'QT'][F]}, {C = }, {E = }, {MC = }')

    # Set code
    code_family = ['HGP', 'BB', 'QT'][F]
    if F == 0: # HGP codes
        code = sp.csr_array(objs[0][codes[C]].astype(np.uint8).todense() & 1)
        Hx, Hz = HGP(code)
    elif F == 1: # BB codes
        l, m = orders[C]
        bbcode = BBCode({x: l, y: m}, polys[C]["A"], polys[C]["B"])
        Hx = sp.csr_array(np.array(bbcode._code_x._matrix, dtype=np.uint8))
        Hz = sp.csr_array(np.array(bbcode._code_z._matrix, dtype=np.uint8))
    elif F == 2: # QT codes
        Hx = load_from_mtx(filenames('X', C))
        Hz = load_from_mtx(filenames('Z', C))
    
    # Compute code params
    assert Hx.shape[1] == Hz.shape[1]
    N = Hx.shape[1]
    rx, rz = Hx.shape[1] - code_dimension(Hx), Hz.shape[1] - code_dimension(Hz)
    K = N - rx - rz
    print(f'Quantum code params: [[N={N}, K={K}]]')
    params = {'N': N, 'K': K}

    # Set simulation parameters
    er = p_vals[E]
    print(f'Error rate: {er:.3f} | MC budget: {MC:1.0g} trials')

    # Run simulation
    code_bm_obj = ClusterPeelingBenchmark(Hx, Hz)
    print('Running cluster decoding benchmark...')
    t0 = time.time()
    cluster_stats = code_bm_obj.cluster_peeling_benchmark([er], max_num_trials=MC, C_sizes=C_sizes_param)
    dt = time.time() - t0
    print(f'Cluster done in {dt:.3f} s')
    # print(f'{code_bm_obj.min_SS_size = }')


    # Save results
    print('Simulation done, saving results...')    
    time.sleep(E)
    with h5py.File(f"cluster_peeling_{code_family}_benchmark.hdf5", "a") as f: 
        code_label = labels[F][C]
        grp = f.require_group(code_label)
        
        # Store [[N, K, Dbound]] params
        for par, val in params.items():
            grp.attrs[str(par)] = val

        # for C, s in enumerate(code_bm_obj.min_SS_size["cluster"]):
        #     grp.attrs[f"minSS_cluster_C={C}"] = min(grp.attrs[f"minSS_cluster_C={C}"], s) if f"minSS_cluster_C={C}" in grp.attrs else s
        
        subgrp = grp.require_group(f'ER={E}')
        
        # Store LER statistics for each decoder
        for c, cluster_stats_C in cluster_stats.items():
            if f"cluster_C{c}_ler" not in subgrp:
                subgrp.create_dataset(f"cluster_C{c}_ler", data=cluster_stats_C['ler'])
                subgrp.create_dataset(f"cluster_C{c}_eb", data=cluster_stats_C['ler_eb'])

        # # Store SS size statistics for each decoder
        # if f"avgSS_cluster_C={C}" not in subgrp:
        #     subgrp.create_dataset(f"avgSS_cluster_C={C}", data=code_bm_obj.avg_SS_size_by_er["cluster"].flatten())
        
    
    print('Results saved. All done.')