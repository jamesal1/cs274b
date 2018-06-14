import pickle as pkl
from model import Model
import math
import numpy as np

vocab_size = 1000

with open("data/relations_mat.pkl", "rb") as f:
    relmat = pkl.load(f)

with open("data/co.pkl","rb") as f:
    comat = pkl.load(f)

biggest = 0
for a, b, c in comat:
    l = math.log(1 + c)
    biggest = max(l,biggest)

comat = [(a,b,math.log(1 + c)/biggest) for a, b, c in comat]

with open("data/vocab.txt", "r") as f:
    vocab = [(v.split(" ")[0],i) for i, v in enumerate(f.readlines())][:vocab_size]
relmat["co"] = comat

m = Model(vocab, relmat, embedding_dimension=50)

for fname in ['./data/model{}.pkl'.format(i + 1) for i in range(5)]:

    m.load(fname)
    print("Likelihood: {}".format(m.estimateLL()))

# def intlist(arr):
#     return [int(i) for i in arr]
#
#
# def estimateLL(self):
#     U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
#     V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
#
#     uis = intlist(np.random.randint(self.vocab_size, size=10000))
#     vis = intlist(np.random.randint(self.vocab_size, size=10000))
#     log_prob = 0
#     Us = U1[uis]
#     Vs = V1[vis]
#
#     print("Estimating log likelihood")
#     for i, r in enumerate(self.R):
#
#         y_true = torch.FloatTensor([self.Rval[i][(k, j)] for (k, j) in zip(uis, vis)]).cuda()
#
#         act = Us @ self.B[i] @ Vs.t()
#         pred = torch.sigmoid(act)
#         log_prob += torch.sum(torch.log(pred + 1e-5) * y_true) + torch.sum(torch.log(1 - pred + 1e-5) * (1 - y_true))
#
#     return log_prob
#
# estimateLL(m)