import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# SMALL 100 4 0.95 0.1
# MEDIUM 50000 7 1 0.1
# LARGE 312020 9 0.95 0.1

IN_DIR = "data"
OUT_DIR = "results"


class QLearning:
    def __init__(self, nstates, nactions, gamma, alpha):
        self.nstates = nstates
        self.nactions = nactions
        self.gamma = gamma
        self.Q = np.zeros((nstates, nactions))
        self.alpha = alpha

    def update(self, sample):
        s, a, r, sp = sample
        s -= 1
        a -= 1
        sp -= 1
        self.Q[s,a] += self.alpha * (r + self.gamma * np.amax(self.Q[sp,:]) - self.Q[s,a])

    def extract_policy(self, outfile):
        pi = np.zeros(self.nstates)
        for s in range(self.nstates):
            pi[s] = np.argmax(self.Q[s,:]) + 1
        np.savetxt(f"./{OUT_DIR}/{outfile}", pi, fmt="%d")


def compute(infile, outfile, nstates, nactions, gamma, alpha):
    # Read sampled transitions
    transitions = np.loadtxt(f"{IN_DIR}/{infile}", dtype=int, delimiter=',', skiprows=1)

    st = time.time()

    # Instantiate model
    model = QLearning(nstates, nactions, gamma, alpha)

    # Train model
    for sample in tqdm(transitions):
        model.update(sample)

    # Extract and write policy
    model.extract_policy(outfile)
    
    et = time.time()
    elapsed_time = et - st

    with open(f"{OUT_DIR}/{outfile[:-7]}.t", 'w') as f:
        f.write(f"Execution time was {elapsed_time} seconds for {infile} with nstates={nstates}, nactions={nactions}, gamma={gamma}, alpha={alpha}")


def main():
    if len(sys.argv) != 7:
        raise Exception("usage: python project2.py <size>.csv <size>.policy nstates nactions gamma alpha")

    infile = sys.argv[1]
    outfile = sys.argv[2]
    nstates = int(sys.argv[3])
    nactions = int(sys.argv[4])
    gamma = float(sys.argv[5])
    alpha = float(sys.argv[6])
    compute(infile, outfile, nstates, nactions, gamma, alpha)


if __name__ == '__main__':
    main()