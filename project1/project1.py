import sys
import networkx as nx
import time
import numpy as np
import numpy.random as rd
import scipy.special as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from joblib import Parallel, delayed, parallel_backend

DIR = "data"

class Variable:
    def __init__(self, name, r):
        self.name = name
        self.r = r


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{},{}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def write_time(time, score, filename):
    with open(filename, 'w') as f:
        f.write(f"Execution time was {time} seconds for a score of {score}")


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

    # Read CSV
    varnames = np.loadtxt(f"{DIR}/{infile}", dtype=str, delimiter=',', max_rows=1)
    D = np.loadtxt(f"{DIR}/{infile}", dtype=int, delimiter=',', skiprows=1)

    # Build vars
    m,n = D.shape
    r = np.max(D, axis=0)
    vars = np.array([Variable(varnames[i],r[i]) for i in range(n)])
    idx2names = dict(zip(range(n),[vars[i].name for i in range(n)]))

    # Build G from example
    # G = nx.DiGraph()
    # G.add_nodes_from(range(n))
    # edges = np.loadtxt(f"{DIR}/{outfile}", dtype=str, delimiter=',')
    # r,c = edges.shape
    # G.add_edges_from([tuple([np.where(edges[i,j]==varnames)[0][0] for j in range(c)]) for i in range(r)])

    # K2 search
    # ordering = range(n)
    # G = K2_search(vars, D, ordering)
    # bs = bayesian_score(vars, G, D)
    # print(f"K2 search: {bs}")

    # Genetic search
    # G = genetic_search(vars, D, r, 5, 5, 30)
    # bs = bayesian_score(vars, G, D)
    # print(f"Memetic search optimized: {bs}")

    # Local search
    # st = time.time()
    # Ginitial = nx.DiGraph()
    # Ginitial.add_nodes_from(range(n))
    # G, bs = local_search(vars, D, r, Ginitial, 30, 0.9)
    # print(f"Local search: {bs}")
    # write_gph(G, idx2names, f"results/{outfile}")
    # et = time.time()
    # elapsed_time = et - st
    # write_time(elapsed_time, bs, f"results/t_{outfile}")
    # return

    # Combo search
    mut_prob = 0.000
    skip_prob = 0.9
    dir_list = os.listdir(f"results")
    version = 0
    while f"{version}_{outfile}" in dir_list:
        version += 1
    G = combo_search(vars, D, r, 10, 25, 30, mut_prob, skip_prob, idx2names, f"{version}_{outfile}")
    bs = bayesian_score(vars, G, D)
    print(f"K2/memetic search optimized: {bs}")

    # K2 search optimized
    # ordering = range(n)
    # G = K2_search_optimized(vars, D, r, ordering)
    # bs = bayesian_score(vars, G, D)
    # print(f"K2 search optimized: {bs}")

    # G = local_search(vars, D, r, G, 30, 0)
    # bs = bayesian_score(vars, G, D)
    # print(f"Local search: {bs}")



    # Oppurtunistic local search
    # Ginitial = nx.DiGraph()
    # Ginitial.add_nodes_from(range(n))
    # G = local_search(vars, D, Ginitial, 500)
    # bs = bayesian_score(vars, G, D)
    # print(f"Opportunistic local search: {bs}")

    # Plot G
    labels = dict(zip(range(n),varnames))
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos=pos, labels=labels, node_color='lightgray', arrowsize=20)
    plt.show()

def generate_alike_graphs(vars, num_genes):
    n = len(vars)
    num_edges = int(n*(n-1)/2)
    varnames = np.loadtxt(f"{DIR}/large.csv", dtype=str, delimiter=',', max_rows=1)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges = np.loadtxt(f"results/32_large.gph", dtype=str, delimiter=',')
    r,c = edges.shape
    G.add_edges_from([tuple([np.where(edges[i,j]==varnames)[0][0] for j in range(c)]) for i in range(r)])

    gene = interpret_graph(G)
    genes = np.zeros((num_genes, num_edges))
    for k in range(num_genes):
        genes[k] = gene
        for e in range(num_edges):
            if rd.rand() < 0.000:
                genes[k,e] = rd.choice(3)
    return genes



def combo_search(vars, D, r, num_genes, generations, local_search_max, mut_prob, skip_prob, idx2names, outfile):
    st = time.time()
    n = len(vars)
    num_edges = int(n*(n-1)/2)
    highest_score = -np.inf

    genes = np.zeros((num_genes, num_edges))
    orderings = np.zeros((num_genes,n), dtype=int)
    for k in range(num_genes):
        ordering = np.arange(n)
        rd.shuffle(ordering)
        orderings[k] = ordering
    with parallel_backend('multiprocessing'):
        results = Parallel(n_jobs=num_genes)(delayed(K2_search_optimized)(vars, D, r, orderings[k]) for k in range(num_genes))
        G, bayes = [i for i,j in results], [j for i,j in results]
        for k in range(num_genes):
            genes[k] = interpret_graph(G[k])
        print(f"Highest score for this generation is {np.max(bayes)}")
        highest_score = np.max(bayes)
        write_gph(G[np.argmax(bayes)], idx2names, f"results/{outfile}")
        et = time.time()
        elapsed_time = et - st
        write_time(elapsed_time, highest_score, f"results/t_{outfile}")

    # genes = np.zeros((num_genes, num_edges))
    # for k in range(num_genes):
    #     ordering = np.arange(n)
    #     rd.shuffle(ordering)
    #     G, bs = K2_search_optimized(vars, D, r, ordering)
    #     print(bs)
    #     genes[k] = interpret_graph(G)
    #     if bs >= highest_score:
    #         write_gph(G, idx2names, f"results/{outfile}")
    #         et = time.time()
    #         elapsed_time = et - st
    #         write_time(elapsed_time, bs, f"results/t_{outfile}")
    #         highest_score = bs

    #genes = rd.choice(3, (num_genes, num_edges), replace=True, p=np.array([0.9, 0.05, 0.05]))

    #genes = generate_alike_graphs(vars, num_genes)

    for _ in range(generations):
        bs = np.zeros(num_genes) - np.inf
        graphs = []

        with parallel_backend('multiprocessing'):
            graphs = Parallel(n_jobs=num_genes)(delayed(build_graph)(n,genes[k]) for k in range(num_genes))
            results = Parallel(n_jobs=num_genes)(delayed(local_search)(vars, D, r, graphs[k], local_search_max, skip_prob) for k in range(num_genes))
            G, bayes = [i for i,j in results], [j for i,j in results]
            print(bayes)
            print([g.edges for g in G])
            print(f"Highest score for this generation is {np.max(bayes)}")
            # if highest_score > np.max(bayes):
            #     return G[np.argmax(bayes)]
            if np.max(bayes) > highest_score:
                print("Updating results")
                highest_score = np.max(bayes)
                write_gph(G[np.argmax(bayes)], idx2names, f"results/{outfile}")
                et = time.time()
                elapsed_time = et - st
                write_time(elapsed_time, highest_score, f"results/t_{outfile}")


        # for k in range(num_genes):
        #     G = build_graph(n,genes[k])
        #     G, bayes = local_search(vars, D, r, G, local_search_max, skip_prob)
        #     graphs.append(G)
        #     bs[k] = bayes
        #     print(bs[k])
        #     if bs[k] >= highest_score:
        #         write_gph(graphs[np.argmax(bs)], idx2names, f"results/{outfile}")
        #         et = time.time()
        #         elapsed_time = et - st
        #         write_time(elapsed_time, bs[k], f"results/t_{outfile}")
        #         highest_score = bs[k]
        # print(f"Highest score for this generation is {np.max(bs)}")
        # if highest_score > np.max(bs):
        #     return graphs[np.argmax(bs)]
        # # highest_score = np.max(bs)
        # # write_gph(graphs[np.argmax(bs)], idx2names, f"results/{outfile}")
        # # et = time.time()
        # # elapsed_time = et - st
        # # write_time(elapsed_time, highest_score, f"results/t_{outfile}")

        
        gene_recombinations = np.zeros((num_genes, num_edges))
        for k in range(num_genes):
            # Tournament selection
            gene_selection = np.zeros(2, dtype=int)

            k_individuals = rd.choice(num_genes, 3, replace=False)
            gene_selection[0] = rd.choice(np.where(bs==np.max(bs[k_individuals]))[0])

            k_individuals = rd.choice(num_genes, 3, replace=False)
            gene_selection[1] = rd.choice(np.where(bs==np.max(bs[k_individuals]))[0])

            print(f"Gene selection: {gene_selection}")
            gene_split = rd.choice(num_edges)
            print(f"Gene split: {gene_split}")

            gene_recombinations[k] = genes[gene_selection[0]]
            gene_recombinations[k,gene_split:] = genes[gene_selection[1],gene_split:]

        for k in range(num_genes):
            for e in range(num_edges):
                if rd.rand() < mut_prob:
                    gene_recombinations[k,e] = rd.choice(3)
        genes = gene_recombinations


def genetic_search(vars, D, r, num_genes, generations, local_search_max, mut_prob=0.01):
    n = len(vars)
    num_edges = int(n*(n-1)/2)

    genes = rd.choice(3, (num_genes, num_edges), replace=True)
    highest_score = -np.inf

    for _ in range(generations):
        bs = np.zeros(num_genes)
        graphs = []
        for k in range(num_genes):
            G = build_graph(n,genes[k])
            G = local_search(vars, D, r, G, local_search_max)
            graphs.append(G)
            bs[k] = bayesian_score(vars, G, D)
            print(bs[k])
        print(bs)
        print(f"Highest score for this generation is {np.max(bs)}")
        if highest_score >= np.max(bs):
            return graphs[np.argmax(bs)]
        highest_score = np.max(bs)
        
        p = bs - np.min(bs)
        #p = p/np.sum(p)
        p = sp.softmax(p)
        #print(p)
        gene_recombinations = np.zeros((num_genes, num_edges))
        for k in range(num_genes):
            # tournament selection
            gene_selection = np.zeros(2, dtype=int)
            k_individuals = rd.choice(num_genes, int(num_genes/2), replace=False)
            gene_selection[0] = np.where(bs==np.max(bs[k_individuals]))[0][0]

            k_individuals = rd.choice(num_genes, int(num_genes/2), replace=False)
            gene_selection[1] = np.where(bs==np.max(bs[k_individuals]))[0][0]

            #gene_selection = rd.choice(num_genes, 2, replace=False, p=p)
            print('gene selection')
            print(gene_selection)
            #gene_split = rd.choice(num_edges, int(num_edges/2), replace=False)
            gene_split = rd.choice(num_edges)
            print('gene split')
            print(gene_split)
            #gene_recombinations[k] = genes[gene_selection[0]]
            #gene_recombinations[k,gene_split] = genes[gene_selection[1],gene_split]
            gene_recombinations[k] = genes[gene_selection[0]]
            gene_recombinations[k,gene_split:] = genes[gene_selection[1],gene_split:]

        for k in range(num_genes):
            for e in range(num_edges):
                if rd.rand() < mut_prob:
                    gene_recombinations[k,e] = rd.choice(3)
        genes = gene_recombinations

    

def build_graph(n, gene):
    edges = []
    index = 0
    for i in range(n):
        for j in range(i+1,n):
            if gene[index] == 1:
                edges.append((i,j))
            elif gene[index] == 2:
                edges.append((j,i))
            index += 1
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G

def interpret_graph(G):
    n = G.number_of_nodes()
    num_edges = int(n*(n-1)/2)
    edges = [e for e in G.edges]
    gene = np.zeros(num_edges)
    index = 0
    for i in range(n):
        for j in range(i+1,n):
            if (i,j) in edges:
                gene[index] = 1
            elif (j,i) in edges:
                gene[index] = 2
            index += 1
    return gene

def local_search(vars, D, r, G, k_max, skip_prob=0):
    n = len(vars)

    # Precomputations
    component_scores = bayesian_score_optimized(vars, G, D)
    parents = [np.array([j for j in G.predecessors(i)], dtype=int) for i in range(n)]

    for k in tqdm(range(k_max)):
        delta_best, i_best, j_best, i_score_best, j_score_best, action = -np.inf, -1, -1, -np.inf, -np.inf, "none"
        for i in range(n):
            for j in range(i+1,n):

                if rd.rand() < skip_prob:
                    continue

                if G.has_edge(i,j):
                    # Try remove
                    G.remove_edge(i,j)
                    if nx.is_directed_acyclic_graph(G):
                    # try:
                    #     nx.find_cycle(G)
                    # except:
                        new_parents = np.delete(parents[j], np.where(parents[j]==i)[0][0])
                        j_score_prime = compute_graph_component(j, new_parents, r, D)
                        delta = j_score_prime - component_scores[j]
                        if delta > delta_best:
                            delta_best, i_best, j_best, j_score_best, action = delta, i, j, j_score_prime, "removeij"
                    G.add_edge(i,j)

                    # Try reverse
                    G.remove_edge(i,j)
                    G.add_edge(j,i)
                    if nx.is_directed_acyclic_graph(G):
                    # try:
                    #     nx.find_cycle(G)
                    # except:
                        new_parents = np.delete(parents[j], np.where(parents[j]==i)[0][0])
                        j_score_prime = compute_graph_component(j, new_parents, r, D)
                        new_parents = np.concatenate(([j], parents[i]))
                        i_score_prime = compute_graph_component(i, new_parents, r, D)
                        delta = (i_score_prime + j_score_prime) - (component_scores[i] + component_scores[j])
                        if delta > delta_best:
                            delta_best, i_best, j_best, i_score_best, j_score_best, action = delta, i, j, i_score_prime, j_score_prime, "reverseij"
                    G.remove_edge(j,i)
                    G.add_edge(i,j)

                elif G.has_edge(j,i):
                    # Try remove
                    G.remove_edge(j,i)
                    if nx.is_directed_acyclic_graph(G):
                    # try:
                    #     nx.find_cycle(G)
                    # except:
                        new_parents = np.delete(parents[i], np.where(parents[i]==j)[0][0])
                        i_score_prime = compute_graph_component(i, new_parents, r, D)
                        delta = i_score_prime - component_scores[i]
                        if delta > delta_best:
                            delta_best, i_best, j_best, i_score_best, action = delta, i, j, i_score_prime, "removeji"
                    G.add_edge(j,i)

                    # Try reverse
                    G.remove_edge(j,i)
                    G.add_edge(i,j)
                    if nx.is_directed_acyclic_graph(G):
                    # try:
                    #     nx.find_cycle(G)
                    # except:
                        new_parents = np.delete(parents[i], np.where(parents[i]==j)[0][0])
                        i_score_prime = compute_graph_component(i, new_parents, r, D)
                        new_parents = np.concatenate(([i], parents[j]))
                        j_score_prime = compute_graph_component(j, new_parents, r, D)
                        delta = (i_score_prime + j_score_prime) - (component_scores[i] + component_scores[j])
                        if delta > delta_best:
                            delta_best, i_best, j_best, i_score_best, j_score_best, action = delta, i, j, i_score_prime, j_score_prime, "reverseji"     
                    G.remove_edge(i,j)
                    G.add_edge(j,i) 

                else:
                    # Try add
                    G.add_edge(j,i)
                    if nx.is_directed_acyclic_graph(G):
                    # try:
                    #     nx.find_cycle(G)
                    # except:
                        new_parents = np.concatenate(([j], parents[i]))
                        i_score_prime = compute_graph_component(i, new_parents, r, D)
                        delta = i_score_prime - component_scores[i]
                        if delta > delta_best:
                            delta_best, i_best, j_best, i_score_best, action = delta, i, j, i_score_prime, "addji"
                    G.remove_edge(j,i)

                    # Try add
                    G.add_edge(i,j)
                    # try:
                    #     nx.find_cycle(G)
                    # except:
                    if nx.is_directed_acyclic_graph(G):
                        new_parents = np.concatenate(([i], parents[j]))
                        j_score_prime = compute_graph_component(j, new_parents, r, D)
                        delta = j_score_prime - component_scores[j]
                        if delta > delta_best:
                            delta_best, i_best, j_best, j_score_best, action = delta, i, j, j_score_prime, "addij"
                    G.remove_edge(i,j)
        
        if delta_best < 0:
            #break
            continue

        if action == "removeij":
            G.remove_edge(i_best,j_best)
            # Update bookkeeping
            component_scores[j_best] = j_score_best
            parents[j_best] = np.delete(parents[j_best], np.where(parents[j_best]==i_best)[0][0])
        elif action == "removeji":
            G.remove_edge(j_best,i_best)
            # Update bookkeeping
            component_scores[i_best] = i_score_best
            parents[i_best] = np.delete(parents[i_best], np.where(parents[i_best]==j_best)[0][0])
        elif action == "addij":
            G.add_edge(i_best,j_best)
            # Update bookkeeping
            component_scores[j_best] = j_score_best
            parents[j_best] = np.concatenate(([i_best], parents[j_best]))
        elif action == "addji":
            G.add_edge(j_best,i_best)
            # Update bookkeeping
            component_scores[i_best] = i_score_best
            parents[i_best] = np.concatenate(([j_best], parents[i_best]))
        elif action == "reverseij":
            G.remove_edge(i_best,j_best)
            G.add_edge(j_best,i_best)
            # Update bookkeeping
            component_scores[i_best] = i_score_best
            parents[i_best] = np.concatenate(([j_best], parents[i_best]))
            component_scores[j_best] = j_score_best
            parents[j_best] = np.delete(parents[j_best], np.where(parents[j_best]==i_best)[0][0])
        elif action == "reverseji":
            G.remove_edge(j_best,i_best)
            G.add_edge(i_best,j_best)
            # Update bookkeeping
            component_scores[j_best] = j_score_best
            parents[j_best] = np.concatenate(([i_best], parents[j_best]))
            component_scores[i_best] = i_score_best
            parents[i_best] = np.delete(parents[i_best], np.where(parents[i_best]==j_best)[0][0])

    #print(f"{k+1} out of {k_max} max iterations")
    return G, sum(component_scores)


def local_search_opportunistic(vars, D, r, G, k_max):
    component_scores = bayesian_score_optimized(vars, G, D)
    #y = bayesian_score(vars, G, D)
    for k in tqdm(range(k_max)):
        Gprime = rand_graph_neighbor(G)
        yprime = -np.inf
        try:
            nx.find_cycle(G)
        except:
            yprime = bayesian_score(vars, Gprime, D)
        if yprime > y:
            y, G = yprime, Gprime
    return G

def rand_graph_neighbor(G):
    n = G.number_of_nodes()
    i = rd.randint(0,n)
    j = (i + rd.randint(1,n)) % n
    Gprime = G.copy()
    if G.has_edge(i,j):
        Gprime.remove_edge(i,j)
    else:
        Gprime.add_edge(i,j)
    return Gprime
    

def K2_search_optimized(vars, D, r, ordering):
    n = len(vars)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Precomputations
    component_scores = bayesian_score_optimized(vars, G, D)
    parents = [np.array([j for j in G.predecessors(i)], dtype=int) for i in range(n)]

    for (k,i) in enumerate(tqdm(ordering[1:])):
        cnt = 0
        while cnt < 4:
            y_best, j_best = -np.inf, -1
            for j in ordering[:k+1]:
                if not G.has_edge(j,i):
                    new_parents = np.concatenate(([j], parents[i]))
                    y_prime = compute_graph_component(i, new_parents, r, D)
                    if y_prime > y_best:
                        y_best, j_best = y_prime, j
            if y_best > component_scores[i]:
                G.add_edge(j_best,i)
                # Update bookkeeping
                component_scores[i] = y_best
                parents[i] = np.concatenate(([j_best], parents[i]))
                cnt += 1
            else:
                break
    return G, sum(component_scores)


def compute_graph_component(i, new_parents, r, D):
    m,n = D.shape
    q = int(np.prod(r[new_parents]))
    M = np.zeros((q, r[i]), dtype=int)
    alpha = np.ones((q, r[i]), dtype=int)
    for o in range(m):
        k = D[o,i] - 1
        j = 0
        if new_parents.size > 0:
            j = np.ravel_multi_index(D[o,new_parents] - 1, r[new_parents])
        M[j,k] += 1
    return bayesian_score_component(M,alpha)


def bayesian_score_component(M, alpha):
    p = np.sum(sp.loggamma(alpha + M))
    p -= np.sum(sp.loggamma(alpha))
    p += np.sum(sp.loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(sp.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p


def bayesian_score_optimized(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return np.array([bayesian_score_component(M[i], alpha[i]) for i in range(n)])
    

def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


def statistics(vars, G, D):
    m,n = D.shape
    r = np.array([vars[i].r for i in range(n)])
    q = np.array([int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)])
    M = [np.zeros((q[i], r[i]), dtype=int) for i in range(n)]
    for o in range(m):
        for i in range(n):
            k = D[o,i] - 1
            parents = np.array([n for n in G.predecessors(i)], dtype=int)
            j = 0
            if parents.size > 0:
                j = np.ravel_multi_index(D[o,parents] - 1, r[parents])
            M[i][j,k] += 1
    return M


def prior(vars, G):
    n = len(vars)
    r = np.array([vars[i].r for i in range(n)])
    q = np.array([int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)])
    return [np.ones((q[i], r[i]), dtype=int) for i in range(n)]


def K2_search(vars, D, ordering):
    n = len(vars)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (k,i) in enumerate(tqdm(ordering[1:])):
        y = bayesian_score(vars, G, D)
        while True:
            y_best, j_best = -np.inf, -1
            for j in ordering[:k+1]:
                if not G.has_edge(j,i):
                    G.add_edge(j,i)
                    yprime = bayesian_score(vars, G, D)
                    if yprime > y_best:
                        y_best, j_best = yprime, j
                    G.remove_edge(j,i)
            if y_best > y:
                y = y_best
                G.add_edge(j_best,i)
            else:
                break
    return G


def local_search_opportunistic(vars, D, G, k_max):
    y = bayesian_score(vars, G, D)
    for k in tqdm(range(k_max)):
        Gprime = rand_graph_neighbor(G)
        yprime = -np.inf
        try:
            nx.find_cycle(G)
        except:
            yprime = bayesian_score(vars, Gprime, D)
        if yprime > y:
            y, G = yprime, Gprime
    return G


def rand_graph_neighbor(G):
    n = G.number_of_nodes()
    i = rd.randint(0,n)
    j = (i + rd.randint(1,n)) % n
    Gprime = G.copy()
    if G.has_edge(i,j):
        Gprime.remove_edge(i,j)
    else:
        Gprime.add_edge(i,j)
    return Gprime


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
