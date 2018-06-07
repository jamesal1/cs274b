import numpy as np
import torch
from probit_factor_model import train_logistic_torch

import time

class Model:

    def __init__(self, vocab, relations, embedding_dimension = 50):
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.U = torch.randn((self.vocab_size, embedding_dimension)).cuda()
        self.V = torch.randn((self.vocab_size, embedding_dimension)).cuda()
        self.B = [None] * len(relations)
        self.emb_size = embedding_dimension

        self.R = relations


    def updateB(self):
        uvs = []

        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)

        print(U1.shape)

        for i, r in enumerate(R):

            uis, vis, rs = zip(*r)
            # Check whether there is too much?
            UVs = (U1[list(uis)].view(-1, self.emb_size + 1, 1)) * (V1[list(vis)].view(-1, 1, self.emb_size + 1))
            UVs = UVs.view(-1, (self.emb_size + 1)**2) # includes an intercept
            # includes an intercept

            y = torch.FloatTensor(rs).cuda()

            self.B[i] = train_logistic_torch(UVs, y).view(self.emb_size + 1,self.emb_size + 1)

    def updateU(self):
        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        for w in range(self.vocab_size):
            start = time.time()

            all_X = []
            all_y = []
            for i, r in enumerate(samplesU[w]):
                vis, rs = r
                all_X += [V1[list(vis)] @ self.B[i].t()]
                all_y += [torch.FloatTensor(rs).cuda()]
            Xb = torch.cat(all_X)
            X = Xb[:, :-1]
            b = Xb[:, -1]
            y = torch.cat(all_y)
            print(X.shape,b.shape)
            self.U[w] = train_logistic_torch(X, y, b)
            print(time.time() - start)

    def updateV(self):
        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        for w in range(self.vocab_size):
            start = time.time()
            all_X = []
            all_y = []
            for i, r in enumerate(samplesU[w]):
                uis, rs = r
                all_X += [U1[list(uis)] @ self.B[i]]
                all_y += [torch.FloatTensor(rs).cuda()]
            Xb = torch.cat(all_X)
            X = Xb[:, :-1]
            b = Xb[:, -1]
            y = torch.cat(all_y)
            self.U[w] = train_logistic_torch(X, y, b)
            print(time.time() - start)

    def findBest(self,r,w):
        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        vs = (U1[w].view(1,-1) @ self.B[r]) @ V1.t()
        us = (U1.view)
        print(vs.shape)


if __name__ == '__main__':
    sample_size = 100000
    us = np.random.randint(15000, size=sample_size)
    vs = np.random.randint(15000, size=sample_size)

    R = [[(int(ui), int(vi), int(np.random.randint(2))) for ui, vi in zip(us, vs)] for _ in range(4)]

    def intlist(arr):
        return [int(i) for i in arr]

    sample_size_u = 1000
    #samplesU = [[(intlist(np.random.randint(15000, size=sample_size_u)), intlist(np.random.randint(2, size=sample_size_u))) for _ in range(4)] for _ in range(15000)]

    vocab = list(range(15000))

    tst_model = Model(vocab, R, embedding_dimension = 50)

    tst_model.updateB()
    tst_model.findBest(1,2)
    #print(tst_model.B)
    tst_model.updateU()