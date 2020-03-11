import numpy as np

allsteps = np.loadtxt('allsteps.txt')

ind = np.argpartition(allsteps, -5000)[-5000:]

np.savetxt("ids.txt", ind.astype(int), fmt='%i', delimiter=",")

dic = {}

for i in range(len(allsteps)):
    dic[i] = allsteps[i]

dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}

tmp = []
ids = []
for i, k in enumerate(dic):
    if i % 6000 < 500:
        ids.append(k)
ids=np.array(ids)
np.savetxt("ids.txt", ids.astype(int), fmt='%i', delimiter=",")