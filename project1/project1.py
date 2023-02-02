import sys
import networkx as nx
import numpy as np
import numpy.random as random
import scipy.special as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

DIR = "data"

class Variable:
    def __init__(self, name, r):
        self.name = name
        self.r = r


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


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

    # Build G from example
    # G = nx.DiGraph()
    # G.add_nodes_from(range(n))
    # edges = np.loadtxt(f"{DIR}/{outfile}", dtype=str, delimiter=',')
    # r,c = edges.shape
    # G.add_edges_from([tuple([np.where(edges[i,j]==varnames)[0][0] for j in range(c)]) for i in range(r)])

    # K2 search
    ordering = range(n)
    G = K2_search(vars, D, ordering)
    bs = bayesian_score(vars, G, D)
    print(f"K2 search: {bs}")


    # Oppurtunistic local search
    Ginitial = nx.DiGraph()
    Ginitial.add_nodes_from(range(n))
    G = local_search(vars, D, Ginitial, 500)
    bs = bayesian_score(vars, G, D)
    print(f"Opportunistic local search: {bs}")

    # Plot G
    # labels = dict(zip(range(n),varnames))
    # pos = nx.planar_layout(G)
    # nx.draw_networkx(G, pos=pos, labels=labels, node_color='lightgray', arrowsize=20)
    # plt.show()


def rand_graph_neighbor(G):
    n = G.number_of_nodes()
    i = random.randint(0,n)
    j = (i + random.randint(1,n)) % n
    Gprime = G.copy()
    if G.has_edge(i,j):
        Gprime.remove_edge(i,j)
    else:
        Gprime.add_edge(i,j)
    return Gprime


def local_search(vars, D, G, k_max):
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
    

def K2_search(vars, D, ordering):
    n = len(vars)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (k,i) in enumerate(tqdm(ordering[1:])):
        y = bayesian_score(vars, G, D)
        while True:
            y_best, j_best = -np.inf, 0
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


def bayesian_score(vars, G, D):
    n = len(vars)
    M = statistics(vars, G, D)
    alpha = prior(vars, G)
    return sum(bayesian_score_component(M[i], alpha[i]) for i in range(n))


def bayesian_score_component(M, alpha):
    p = np.sum(sp.loggamma(alpha + M))
    p -= np.sum(sp.loggamma(alpha))
    p += np.sum(sp.loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(sp.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p


def statistics(vars, G, D):
    m,n = D.shape
    r = np.array([vars[i].r for i in range(n)])
    q = np.array([int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)])
    M = [np.zeros((q[i], r[i]), dtype=int) for i in range(n)]
    for o in range(m):
        for i in range(n):
            k = D[o,i] - 1
            parents = np.array([n for n in G.predecessors(i)])
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


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
