import numpy as np

allsteps = np.loadtxt('allsteps.txt')
prev_ids = np.loadtxt("indices_mnist.txt", dtype=int)

prev_ids = set(prev_ids)
ind = np.argpartition(allsteps, -5000)[-5000:]

np.savetxt("ids.txt", ind.astype(int), fmt='%i', delimiter=",")
