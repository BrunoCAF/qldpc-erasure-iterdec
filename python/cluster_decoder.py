import numpy as np
import copy # for deep copy of dictionaries
import networkx as nx
import matplotlib.pyplot as plt
import random

# ========================================================================
# erasure decoding of quantum LDPC codes via stopping set decomposition
# Hanwen Yao, 12/1/2024
# TODO: review this program for better documentation
# ========================================================================

class peeling_decoder:
    def __init__(self, n, m, row, col):
        
        # global parameters (const)
        self.n = n
        self.m = m
        self.row = row
        self.col = col
        
        # construct c_nodes and v_nodes (dict)
        self.v_nodes = [set() for _ in range(self.n)]
        self.c_nodes = [set() for _ in range(self.m)]
        for e in range(len(self.row)):
            self.v_nodes[self.col[e]].add(e)
            self.c_nodes[self.row[e]].add(e)
                
        # decoding data structure (dynamic)
        self.original_erasure_vec = np.zeros(self.n, dtype=int)
        self.original_syndrome = np.zeros(self.m, dtype=int)
        
        self.current_v_nodes = {}
        self.current_c_nodes = {}
        self.current_syndrome = np.zeros(self.m, dtype=int)
        
        self.decoded_err = np.zeros(self.n, dtype=int)
        
    def _load(self, erasure_vec, syndrome):
        
        # check the lengths of erasure_vec and syndrome
        if len(erasure_vec) != self.n:
            raise ValueError(f"erasure_vec must be of length {self.n}")
        if len(syndrome) != self.m:
            raise ValueError(f"syndrome must be of length {self.m}")
            
        self.original_erasure_vec[:] = erasure_vec[:]
        self.original_syndrome[:] = syndrome[:]
        self.current_syndrome[:] = syndrome[:]
        self.decoded_err[:] = 0
        
        # load the dynamic dicts
        self.current_v_nodes = {}
        self.current_c_nodes = {}
        for v in range(self.n):
            if self.original_erasure_vec[v]:
                self.current_v_nodes[v] = self.v_nodes[v].copy()
                for e in self.current_v_nodes[v]:
                    c = self.row[e]
                    if c in self.current_c_nodes:
                        self.current_c_nodes[c].add(e)
                    else:
                        self.current_c_nodes[c] = set([e])
        
    def _peel(self, debug = 0):
        '''
        peel self.current_erasure_vec
        update self.current_erasures, self.c_erased, self.current_syndrome and self.decoded_err
        '''
        # initialize the set of dangling checks
        dangling_check = set()
        for c in self.current_c_nodes:
            if len(self.current_c_nodes[c]) == 1:
                dangling_check.add(c)
        
        if debug:
            print(f"dangling_check = {dangling_check}")
            self._plot_tanner_graph(1)
        
        # peeling
        while len(dangling_check) > 0:
            c = dangling_check.pop()
            [e] = self.current_c_nodes.pop(c)
            v = self.col[e]
            self.decoded_err[v] = self.current_syndrome[c]
            for ee in self.current_v_nodes.pop(v):
                if ee != e:
                    cc = self.row[ee]
                    self.current_syndrome[cc] ^= self.decoded_err[v]
                    self.current_c_nodes[cc].remove(ee)
                    if len(self.current_c_nodes[cc]) == 0:
                        if self.current_syndrome[cc] != 0:
                            return 0
                        else:
                            self.current_c_nodes.pop(cc)
                            dangling_check.remove(cc)
                    elif len(self.current_c_nodes[cc]) == 1:
                        dangling_check.add(cc)
                            
            if debug and 0:
                print(f"c = {c}, e = {e}, v = {v}")
                print(f"dangling_check = {dangling_check}")
                self._plot_tanner_graph(1)
        return 1
    
#     def _peel_with_guessing(self, debug = 0):
#         '''
#         '''
#         dec_cnt = 0
#         if self._peel(debug) == 0:
#             return 0,0
        
#         while len(self.current_v_nodes) > 0:
            
#             # randomly fix an erased bit to zero
#             v = random.choice(list(self.current_v_nodes.keys()))
#             self.decoded_err[v] = 0
#             dec_cnt += 1
            
#             if debug:
#                 print(f"fix v_node[{v}] to zero")
            
#             # update the residual Tanner graph
#             for e in self.current_v_nodes.pop(v):
#                 c = self.row[e]
#                 self.current_c_nodes[c].remove(e)
#                 if len(self.current_c_nodes[c]) == 0:
                    
#                     # check contradiction
#                     if self.current_syndrome[c] != 0:
#                         return 0, dec_cnt
#                     else:
#                         self.current_c_nodes.pop(c)
            
#             if self._peel(debug) == 0:
#                 return 0, dec_cnt
            
#         return 1, dec_cnt

    def decode(self, erasure_vec, syndrome, debug=0):
        
        self._load(erasure_vec,syndrome)
        
        self.peel(debug)
        
        if len(self.current_v_nodes) == 0:
            return self.decoded_err
        else:
            return []
        
    def match(self, err):
        
        syn = np.zeros(self.m, dtype=int)
        for e in range(len(self.row)):
            syn[self.row[e]] ^= err[self.col[e]]
        return np.array_equal(syn,self.original_syndrome)
        
    def _plot_tanner_graph(self, bipartite=True):
        
        if self.current_v_nodes:
            
            G = nx.Graph()

            # Adding check nodes and variable nodes
            check_nodes = [f'c{c}' for c in self.current_c_nodes] 
            variable_nodes = [f'v{v}' for v in self.current_v_nodes]
            G.add_nodes_from(check_nodes, bipartite=0)
            G.add_nodes_from(variable_nodes, bipartite=1)

            # Adding edges
            for v in self.current_v_nodes:
                for e in self.current_v_nodes[v]:
                    G.add_edge(f'c{self.row[e]}', f'v{v}', weight = e)

            # Specify node locations
            if bipartite:
                pos = nx.bipartite_layout(G, variable_nodes)
                pos = {node:(x,y) for node,(y,x) in pos.items()}
            else:
                pos = nx.spring_layout(G)
            
            # Specify edge labels
            edge_labels = nx.get_edge_attributes(G, 'weight')

            plt.figure(figsize=(2,2))
            nx.draw(G, pos, with_labels=True, node_color=['lightblue' if n.startswith('c') else 'lightgreen' for n in G.nodes]) #, node_size=100, font_size=16, font_weight='bold')
            
            # Draw edge labels
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.show()
            
class cluster_decoder:
    '''
    (-,0) represent v_nodes and (-,1) represent c_nodes
    '''
    def __init__(self, n, m, row, col):
        self.peeler = peeling_decoder(n, m, row, col)
        self.n = self.peeler.n
        self.m = self.peeler.m
        self.row = self.peeler.row
        self.col = self.peeler.col
                        
        # dict from (row_ind, col_ind) to edge_ind
        self.edge_index = {(self.row[e],self.col[e]):e for e in range(len(self.row))}
        
        # list of root cut_nodes (dynamic)
        # Note: all the root cut_nodes are v_nodes
        self.roots = []
        self.decoded_err = np.zeros(self.n, dtype=int)
        
        # stat
        self.size_isolated_comps_n = []
        self.size_isolated_comps_m = []
        self.size_clusters_n = []
        self.size_clusters_m = []
        
    def decode(self, erasure_vec, syndrome, debug = 0):

        self.peeler._load(erasure_vec, syndrome)
        
#         successful, dec_cnt = self.peeler._peel_with_guessing(debug)
#         if successful:
#             return self.peeler.decoded_err, dec_cnt
#         else:
#             self.peeler._load(erasure_vec, syndrome)
#             self.peeler._peel()
            
#         self.decoded_err = self.peeler.decoded_err
        
#         if debug:
#             print("After peeling:")
#             # print(f"current_syn = {self.peeler.current_syndrome}")
#             print(f"syndrome wt: {sum(self.peeler.original_syndrome)} -> {sum(self.peeler.current_syndrome)}")
#             # print(f"decoded_err = {self.decoded_err}")
#             print(f"erasure cnt: {sum(self.peeler.original_erasure_vec)} -> {len(self.peeler.current_v_nodes)}")
#             print()
#             # print(f"Tanner graph:")
#             # self.peeler._plot_tanner_graph(1)
        
#         self._build(debug)
        
#         for root in self.roots:
#             self._solve(root, None, 0)
#             if root.frozen == False:
#                 root.value = 0
#             self._resolve(root, None, 0)
            
#         return self.decoded_err, -1
    
        self.peeler._peel(debug)
            
        if len(self.peeler.current_v_nodes) == 0:
            # return self.peeler.decoded_err, 0, [], [], [], []
            return 0
        else:
            
            self.decoded_err = self.peeler.decoded_err
            
            self._build(debug)

            # >>> ADD THIS BLOCK <<<
            largest_cluster_size = max(self.size_clusters_n) if self.size_clusters_n else 0
            return largest_cluster_size
            # if largest_cluster_size > max(C_sizes):
            #     if debug:
            #         print(f"[abort] largest cluster size = {largest_cluster_size} > cap = {max(C_sizes)}")
            #     # Return a structured “gave up” signal; err is partial (peeling-only)
            #     return (
            #         None,                  # decoded_err not produced
            #         largest_cluster_size,  # largest cluster size
            #         self.size_isolated_comps_n,
            #         self.size_isolated_comps_m,
            #         self.size_clusters_n,
            #         self.size_clusters_m,
            #     )
            
            # if debug:
            #     self.print_forest()
            #     print(f"stopping set size = {len(self.peeler.current_v_nodes)}")
            #     print(f"self.size_isolated_comps_n = {self.size_isolated_comps_n}")
            #     print(f"self.size_isolated_comps_m = {self.size_isolated_comps_m}")
            #     print(f"self.size_clusters_n = {self.size_clusters_n}")
            #     print(f"self.size_clusters_m = {self.size_clusters_m}")
                
            # for root in self.roots:
            #     self._solve(root, None, 0)
            #     if root.frozen == False:
            #         root.value = 0
            #     self._resolve(root, None, 0)
                
            # # check if self.decoded_err match the syndrome
            # if self.peeler.match(self.decoded_err) == False:
            #     raise Exception(f"solution after cluster decomposition doesn't match the syndrome.")
                
            # return self.decoded_err, largest_cluster_size, self.size_isolated_comps_n, self.size_isolated_comps_m, self.size_clusters_n, self.size_clusters_m                

    def _build(self, debug = 0):
        '''
        build a forest of clusters
        '''
        cutnode_stack = []
        
        self.size_clusters_n = []
        self.size_clusters_m = []
        self.size_isolated_comps_n = []
        self.size_isolated_comps_m = []
        
        for edge_set, node in self._biconnected_dfs():
            
            if debug and 0:
                print(f"edge_set = {edge_set}")
                print(f"node = {node}")
                print(f"cutnode_stack = {[n.node for n in cutnode_stack]}")
                print()
                
            new_cluster = cluster(self, edge_set)
            
            # Record stat
            self.size_clusters_n.append(len(new_cluster.local_v_nodes))
            self.size_clusters_m.append(len(new_cluster.local_c_nodes))
            
            # adopt all its children cut_nodes at the top of the stack
            while cutnode_stack and cutnode_stack[-1].node != node and new_cluster.contain(cutnode_stack[-1].node):
                new_cluster.add_child(cutnode_stack.pop())
                
            # create a new cut_node if needed and join the stack
            if not cutnode_stack or node != cutnode_stack[-1].node:
                cutnode_stack.append(cut_node(node))
            cutnode_stack[-1].add_child(new_cluster)
            
        self.roots = cutnode_stack
        
        for n in self.roots:
            # compute the # of v_nodes and c_nodes on the tree rooted at n
            v_node_cnt, c_node_cnt = n.compute_subtree_stat()
            self.size_isolated_comps_n.append(v_node_cnt)
            self.size_isolated_comps_m.append(c_node_cnt)
            
    def _solve(self, n = None, c = None, debug = 0):
        # for cut_node n
        if n:
            # recursive solving
            for child in n.children:
                self._solve(None, child, debug)
                
            # v_node: frozen if one child is frozen
            if n.node[1] == 0:
                n.frozen = False
                for child in n.children:
                    if child.frozen:
                        n.frozen = True
                        n.value = child.value
                        break
                        
            # c_node: free if one child is free
            elif n.node[1] == 1:
                n.frozen = True
                n.value = 0
                for child in n.children:
                    if child.frozen:
                        n.value ^= child.value
                    else:
                        n.frozen = False
                        break
                        
            if debug:
                print(f"n.node = {n.node}")
                print(f"n.frozen = {n.frozen}, n.value = {n.value}")
                print()
                        
        # for cluster c
        elif c:
            # recursive solving
            for child in c.children:
                self._solve(child, None, debug)
            c.solve(0)
                       
            if debug:
                print(f"c.edge_set = {c.edge_set}")
                print(f"c.frozen = {c.frozen}, c.value = {c.value}, c.solutions = {c.solutions}")
                print()
    
    def _resolve(self, n = None, c = None, debug = 0):
        # for cut_node n
        if n:
            # v_node
            if n.node[1] == 0:
                for child in n.children:
                    child.value = n.value
                     
            # c_node
            elif n.node[1] == 1:
                first_free = True
                for child in n.children:
                    if child.frozen == False:
                        if first_free:
                            child.value = n.value
                            first_free = False
                        else:
                            child.value = 0
            
            if debug:
                print(f"n.node = {n.node}, n.value = {n.value}")
                for child in n.children:
                    print(f"child.edge_set = {child.edge_set}, child.value = {child.value}, child.solutions = {child.solutions}")
                print()
                
            # recursively resolve
            for child in n.children:
                self._resolve(None, child, debug)
                            
        # for cluster c
        elif c:
            # load decoded_err
            for v in c.local_v_nodes:
                self.decoded_err[v] = c.solutions[c.value][v]
            
            # assign values to children cut_nodes
            for child in c.children:
                # child v_node
                if child.node[1] == 0:
                    child.value = c.solutions[c.value][child.node[0]]
                    
                # child c_node
                elif child.node[1] == 1:
                    child.value = self.peeler.current_syndrome[child.node[0]]
                    for e in c.local_c_nodes[child.node[0]]:
                        child.value ^= c.solutions[c.value][self.col[e]]

            if debug:
                print(f"c.edge_set = {c.edge_set}, c.value = {c.edge_set}, c.solutions = {c.solutions}")
                print(f"self.decoded_err = {self.decoded_err}")
                for child in c.children:
                    print(f"child.node = {child.node}, child.value = {child.value}")
                print()
                        
            # recursively resolve
            for child in c.children:
                self._resolve(child, None, debug)
                    
    def _print_tree(self, c = None, n = None, level=0):
        if c:
            print("  " * level + str(c.edge_set), f"{len(c.local_v_nodes)} v_nodes", f"{len(c.local_c_nodes)} c_nodes")
            for child in c.children:
                self._print_tree(None, child, level+1)
        elif n:
            print("  " * level + str(n.node))
            for child in n.children:
                self._print_tree(child, None, level+1)

    def print_forest(self):
        print(f"self.roots = {[n.node for n in self.roots]}")
        for n in self.roots:
            print(f"Tree:")
            self._print_tree(None, n, 0)
            print()

    # depth-first search algorithm to generate articulation points and biconnected components
    def _biconnected_dfs(self):
        '''
        decompose the tanner subgraph induced by the remaining erasure into 
        forests of biconnected components via Tarjan's algorithm.

        reference: "_biconnected_dfs" at https://github.com/valiantljk/graph-partition/blob/master/algorithms/components/biconnected.py#L365
        '''
        # Tarjan's algorithm
        node_set = [(v,0) for v in self.peeler.current_v_nodes] + [(c,1) for c in self.peeler.current_c_nodes]
        visited = set()
        
        for s in self.peeler.current_v_nodes:
            
            if (s,0) in visited:
                continue
                
            discovery = {(s,0):0} # discovery time
            low = {(s,0):0}       # lowest discovery time among reachable nodes
            visited.add((s,0))
            edge_stack = []
            stack = [((s,0),(s,0),iter(self.peeler.current_v_nodes[s]))]
            
            # depth first search
            while stack:
                
                # retrieve a triple from the stack
                grandparent, parent, edge_iter = stack[-1]
                
                try:
                    
                    # if possible, create a child node
                    edge = next(edge_iter)
                    if parent[1] == 0:
                        child = (self.row[edge],1) 
                    else:
                        child = (self.col[edge],0)
                        
                    # pass if grandparent = child
                    if grandparent == child:
                        continue
                        
                    # if visited, update low[parent]
                    if child in visited:
                        # this is always true, right?
                        if discovery[child] <= discovery[parent]:
                            low[parent] = min(low[parent],discovery[child])
                            edge_stack.append(edge)
                            
                    # if new, record it and update stack and erasure stack
                    else:
                        low[child] = discovery[child] = len(discovery)
                        visited.add(child)
                        
                        # update stack
                        if child[1] == 0:
                            stack.append((parent,child,iter(self.peeler.current_v_nodes[child[0]])))
                        else:
                            stack.append((parent,child,iter(self.peeler.current_c_nodes[child[0]])))
                        edge_stack.append(edge)
                        
                # if we have checked every outgoing edge from a node        
                except StopIteration:
                    
                    # pop the last item from stack
                    stack.pop()
                    
                    # if grandparent != root
                    if len(stack) > 1:
                        
                        # if we have a new cluster
                        if low[parent] >= discovery[grandparent]:
                            
                            # obtain ind = edge_stack.index((grandparent,parent))
                            if grandparent[1] == 1:
                                row_col_pair = (grandparent[0],parent[0])
                            else:
                                row_col_pair = (parent[0],grandparent[0])
                            ind = edge_stack.index(self.edge_index[row_col_pair])
                            
                            # construct a new cluster edge_stack[ind:] with grandparent as parent cut_node
                            yield edge_stack[ind:], grandparent
                            
                            # update edge_stack
                            edge_stack = edge_stack[:ind]
                            
                        # otherwise, update low[grandparent]
                        low[grandparent] = min(low[parent], low[grandparent])
                        
                    # if grandparent == root and parent != root
                    elif stack:
                        
                        # construct a new cluster edge_stack[ind:] with root as parent cut_node
                        ind = edge_stack.index(self.edge_index[(parent[0],grandparent[0])])
                        yield edge_stack[ind:], grandparent
            
class cut_node:
    def __init__(self, node):
        
        self.node = node
        self.parent = None
        self.children = []
        
        self.frozen = False
        self.value = 0 # error bit for v_node, syndrome contribution for c_node
        
    def add_child(self, c):
        
        self.children.append(c)
        c.parent = self
        
    # compute the size of the cluster tree rooted at this cut_node
    def compute_subtree_stat(self):
        v_node_cnt = 0
        c_node_cnt = 0
        # add the current cut_node
        if self.node[1] == 0:
            v_node_cnt += 1
        else:
            c_node_cnt += 1
        # go over each child cluster
        for c in self.children:
            v_node_cnt += len(c.local_v_nodes)
            c_node_cnt += len(c.local_c_nodes)
            # double counting
            if self.node[1] == 0:
                v_node_cnt -= 1
            else:
                c_node_cnt -= 1
            # recurisive counting
            for n in c.children:
                v_cnt, c_cnt = n.compute_subtree_stat()
                v_node_cnt += v_cnt
                c_node_cnt += c_cnt
                # double counting
                if n.node[1] == 0:
                    v_node_cnt -= 1
                else:
                    c_node_cnt -= 1

        return v_node_cnt, c_node_cnt
        
class cluster:
    def __init__(self, cluster_decoder, edge_set):
        
        self.cluster_decoder = cluster_decoder
        
        self.edge_set = edge_set
        self.parent = None
        self.children = []
        
        self.frozen = False
        self.value = 0 # error bit for parent v_node, syndrome contribution for parent c_node
        self.solutions = [{},{}]
        
        # static data structure
        self.local_c_nodes = {}
        self.local_v_nodes = {}
        self.local_syndrome = {}
        for e in self.edge_set:
            c = self.cluster_decoder.row[e]
            v = self.cluster_decoder.col[e]
            if c in self.local_c_nodes:
                self.local_c_nodes[c].add(e)
            else:
                self.local_c_nodes[c] = set([e])
            if v in self.local_v_nodes:
                self.local_v_nodes[v].add(e)
            else:
                self.local_v_nodes[v] = set([e])

        # initialize self.local_syndrome
        for c in self.local_c_nodes:
            self.local_syndrome[c] = self.cluster_decoder.peeler.current_syndrome[c]
            
    def print_self(self, bipartite = 0):
        
        print(f"edge_set = {self.edge_set}")
        print(f"solutions = {self.solutions}")
        print(f"parent = {self.parent}")
        print(f"children = {self.children}")
        print(f"local_c_nodes = {self.local_c_nodes}")
        print(f"local_v_nodes = {self.local_v_nodes}")
        print(f"local_syndrome = {self.local_syndrome}")
        
        # Tanner graph
        G = nx.Graph()
        check_nodes = [f'c{c}' for c in self.local_c_nodes] 
        variable_nodes = [f'v{v}' for v in self.local_v_nodes]
        G.add_nodes_from(check_nodes, bipartite=0)
        G.add_nodes_from(variable_nodes, bipartite=1)
        
        # Adding edges
        for e in self.edge_set:
            G.add_edge(f'c{self.cluster_decoder.row[e]}', f'v{self.cluster_decoder.col[e]}', weight=e)
            
        # Specify node locations
        if bipartite:
            pos = nx.bipartite_layout(G, variable_nodes)
            pos = {node:(x,y) for node,(y,x) in pos.items()}
        else:
            pos = nx.spring_layout(G)
            
        # Specify edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        
        # Draw
        plt.figure(figsize=(4,4))
        nx.draw(G, pos, with_labels=True, node_color=['lightblue' if n.startswith('c') else 'lightgreen' for n in G.nodes])
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Show the plot
        plt.show()
        
    def add_child(self, n):
        
        self.children.append(n)
        n.parent = self
        
    def contain(self, node):
        
        if node[1] == 0:
            return (node[0] in self.local_v_nodes)
        else:
            return (node[0] in self.local_c_nodes)

    def solve(self, debug=0):
        
        if self.parent == None:
            self.solutions[0] = self._solve(None, None, debug)
        else:
            self.solutions[0] = self._solve(self.parent.node, 0, debug)
            self.solutions[1] = self._solve(self.parent.node, 1, debug)
            
            if self.solutions[0]:
                if self.solutions[1]:
                    self.frozen = False
                else:
                    self.frozen = True
                    self.value = 0
            elif self.solutions[1]:
                self.frozen = True
                self.value = 1
            else:
                raise Exception(f"no solution for the current cluster.")
        return
        
    def _solve(self, parent_node = None, value = None, debug=0):
        
        current_c_nodes = copy.deepcopy(self.local_c_nodes)
        current_v_nodes = copy.deepcopy(self.local_v_nodes)
        current_syndrome = copy.deepcopy(self.local_syndrome)
        solution = {}
        
        if debug:
            print(f">>> ----------------------------")
            print(f">>> edge_set = {self.edge_set}")
            print(f">>> set parent_node {parent_node} to value = {value}")
        
        # preprocess parent_node
        if parent_node == None:
            pass
        elif parent_node[1] == 0:
            v = parent_node[0]
            solution[v] = value
            for e in current_v_nodes.pop(v):
                c = self.cluster_decoder.row[e]
                current_c_nodes[c].remove(e)
                current_syndrome[c] ^= value
        elif parent_node[1] == 1:
            current_syndrome[parent_node[0]] = value
            
        if debug and 0:
            print("current_c_nodes:", current_c_nodes)
            print("current_v_nodes:", current_v_nodes)
            print("current_syndrome:", current_syndrome)
            print("solution:", solution)
            print()
            
        # preprocess children cut_nodes
        for child in self.children:
            if child.node[1] == 0 and child.frozen:
                v = child.node[0]
                solution[v] = child.value
                for e in current_v_nodes.pop(v):
                    c = self.cluster_decoder.row[e]
                    current_c_nodes[c].remove(e)
                    current_syndrome[c] ^= child.value
            elif child.node[1] == 1:
                c = child.node[0]
                if child.frozen:
                    current_syndrome[c] ^= child.value
                else:
                    current_syndrome.pop(c)
                    for e in current_c_nodes.pop(c):
                        v = self.cluster_decoder.col[e]
                        current_v_nodes[v].remove(e)
            if debug:
                print(f">>> child.node = {child.node}, child.value = {child.value}, child.frozen = {child.frozen}")
                
        if debug:
            print(">>> current_c_nodes:", current_c_nodes)
            print(">>> current_v_nodes:", current_v_nodes)
            print(">>> current_syndrome:", current_syndrome)
            print(">>> solution:", solution)
            print()
                        
        # create mapping to the submatrix indices
        mr = len(current_c_nodes)
        mc = len(current_v_nodes)
        row_map = {}                      # dict:  row ind -> m
        col_map = {}                      # dict:  col ind -> m
        row_ind = np.zeros(mr, dtype=int) # array: m -> row ind
        col_ind = np.zeros(mc, dtype=int) # array: m -> col ind
        
        for c, i in zip(current_c_nodes, range(mr)):
            row_map[c] = i
            row_ind[i] = c
        for v, i in zip(current_v_nodes, range(mc)):
            col_map[v] = i
            col_ind[i] = v
            
        if debug:
            print(">>> mr:", mr)
            print(">>> mc:", mc)
            print(">>> row_map:", row_map)
            print(">>> col_map:", col_map)
            print(">>> row_ind:", row_ind)
            print(">>> col_ind:", col_ind)
            print()
            
        # create submatrix m = [H|s]
        m = np.zeros((mr,mc + 1), dtype=int)
        for v in current_v_nodes:
            for e in current_v_nodes[v]:
                c = self.cluster_decoder.row[e]
                m[row_map[c]][col_map[v]] = 1
                
        # add syndrome into m
        for i in range(mr):
            m[i][-1] = current_syndrome[row_ind[i]]
            
        if debug:
            print(f">>> m:")
            print(m)
            print()
            
        # Gaussian elimination: m -> RREF, record pivots
        pivot_cols = {}
        r = 0
        for i in range(mc):
            p = r
            while p < mr and m[p][i] == 0:
                p += 1
            if p < mr:
                m[[r, p]] = m[[p, r]] # swap row r with row p
                pivot_cols[r] = i
                for j in range(mr):
                    if m[j][i] == 1 and j != r:
                        m[j] ^= m[r]
                r += 1
                
        if debug:
            print(f">>> pivot_cols = {pivot_cols}")
            print(f">>> m (RREF):")
            print(m)
            print()
                
        # check if solvable
        for p in range(r,mr):
            if m[p][-1] == 1:
                if debug:
                    print(f">>> solution = {solution}")
                    print(f">>> ----------------------------")
                return {}
            
        # set solution by assigning zeros to non-pivotal cols
        for i in range(mc):
            solution[col_ind[i]] = 0
        for r in pivot_cols:
            solution[col_ind[pivot_cols[r]]] = m[r][-1]
            
        if debug:
            print(f">>> solution = {solution}")
            print(f">>> ----------------------------")
                    
        return solution
