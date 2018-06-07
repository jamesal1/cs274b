import numpy as np
import torch
from probit_factor_model import train_logistic_torch
import pickle as pkl
import time
from collections import defaultdict

def intlist(arr):
    return [int(i) for i in arr]


def subsample(li, num):
    indices = np.arange(len(li))
    np.random.shuffle(indices)
    return [li[ind] for ind in indices[:num]]

class Model:

    def __init__(self, vocab, relations, embedding_dimension = 50):
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.U = torch.randn((self.vocab_size, embedding_dimension)).cuda() * 0.001
        self.V = torch.randn((self.vocab_size, embedding_dimension)).cuda() * 0.001
        self.B = [torch.randn((embedding_dimension+1, embedding_dimension+1)).cuda() * 0.001] * len(relations)
        self.emb_size = embedding_dimension
        self.min_sample_size = 20
        self.max_sample_size_B = 10000
        self.max_sample_size_w = 100
        self.relation_names = []
        self.R = []
        for k, v in relations.items():
            self.relation_names+=[k]
            self.R+=[v]
        print(self.relation_names)
        self.Rpos = [set([(int(ui), int(vi)) for ui, vi, _ in r]) for r in self.R]
        self.Rval = [defaultdict(lambda:0,[((int(ui),int(vi)),val) for ui,vi,val in r]) for r in self.R]
        self.RposU = []
        self.RposV = []
        for r in self.R:
            du = defaultdict(lambda:set())
            dv = defaultdict(lambda:set())
            for ui, vi, rel in r:
                du[ui].add((int(vi),rel))
                dv[vi].add((int(ui),rel))
            self.RposU += [du]
            self.RposV += [dv]
        # self.RposU = [{ui: set([(int(vi), rel) for utmp, vi, rel in r if utmp == ui]) for ui, _, _ in r} for r in self.R]
        # self.RposV = [{vi: set([(int(ui), rel) for ui, vtmp, rel in r if vtmp == vi]) for _, vi, _ in r} for r in self.R]
        #

    def sample_neg_R(self, r_ind):

        sample_size = min(len(self.Rpos[r_ind]),self.max_sample_size_B)
        us = np.random.randint(self.vocab_size, size= 2 * sample_size)
        vs = np.random.randint(self.vocab_size, size= 2 * sample_size)

        r_neg = [(int(ui), int(vi), 0) for ui, vi in zip(us, vs) if (int(ui), int(vi)) not in self.Rpos[r_ind]]
        return r_neg

    def get_samples_for_w(self, w_ind, sample_for_U_update=True):
        Rs = []
        for i, rel in enumerate(self.RposU if sample_for_U_update else self.RposV):
            if w_ind in rel:
                rel_w = rel[w_ind]
                sample_size = min(len(rel_w),self.max_sample_size_w)
                vis, rs = zip(*subsample(list(rel_w), sample_size))
            else:
                vis = []
                rs = []
                sample_size = self.min_sample_size
            vis = list(vis) + intlist(np.random.randint(self.vocab_size,size= 2 * sample_size))
            rs = list(rs) + [0] * (2 * sample_size)
            Rs += [(vis, rs)]
        return Rs


    def updateB(self):

        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)

        for i, r in enumerate(self.R):

            ## Subsampling true relations
            indices = np.arange(len(r))
            np.random.shuffle(indices)
            r = [r[ind] for ind in indices[:self.max_sample_size_B]] + self.sample_neg_R(i)

            uis, vis, rs = zip(*r)

            UVs = (U1[intlist(uis)].view(-1, self.emb_size + 1, 1)) * (V1[intlist(vis)].view(-1, 1, self.emb_size + 1))
            UVs = UVs.view(-1, (self.emb_size + 1)**2) # includes an intercept
            # includes an intercept

            y = torch.FloatTensor(rs).cuda()

            self.B[i] = train_logistic_torch(UVs, y, thetas=self.B[i].view(-1)).view(self.emb_size + 1,self.emb_size + 1)

    def updateU(self):

        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        for w in range(self.vocab_size):
            start = time.time()

            all_X = []
            all_y = []
            for i, r in enumerate(self.get_samples_for_w(w,sample_for_U_update=True)):
                vis, rs = r
                all_X += [V1[list(vis)] @ self.B[i].t()]
                all_y += [torch.FloatTensor(rs).cuda()]
            Xb = torch.cat(all_X)
            X = Xb[:, :-1]
            b = Xb[:, -1]
            y = torch.cat(all_y)
            print(X.shape,y.shape,b.shape)
            self.U[w] = train_logistic_torch(X, y, thetas=self.U[w], b= b)
            print(time.time() - start)

    def updateV(self):

        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        for w in range(self.vocab_size):

            start = time.time()
            all_X = []
            all_y = []
            for i, r in enumerate(self.get_samples_for_w(w, sample_for_U_update=False)):
                uis, rs = r
                all_X += [U1[list(uis)] @ self.B[i]]
                all_y += [torch.FloatTensor(rs).cuda()]
            Xb = torch.cat(all_X)
            X = Xb[:, :-1]
            b = Xb[:, -1]
            y = torch.cat(all_y)
            self.V[w] = train_logistic_torch(X, y, thetas=self.V[w], b= b)
            print(time.time() - start)

    def findBest(self, r, w):
        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        vs = (U1[w].view(1,-1) @ self.B[r]) @ V1.t()
        us = (U1 @ (self.B[r] @ V1[w]))
        return np.argsort(us.cpu().numpy()), np.argsort(vs.cpu().numpy())

    def estimateLL(self):
        U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)

        uis = intlist(np.random.randint(self.vocab_size, size=10000))
        vis = intlist(np.random.randint(self.vocab_size, size=10000))
        log_prob = 0
        Us = U1[uis]
        Vs = V1[vis]

        print("Estimating log likelihood")
        for i, r in enumerate(self.R):

            y_true = torch.FloatTensor([self.Rval[i][(k, j)] for (k, j) in zip(uis, vis)]).cuda()

            act = Us @ self.B[i] @ Vs.t()
            pred = torch.sigmoid(act)
            log_prob += torch.sum(torch.log(pred + 1e-7) * (y_true)) + torch.sum(torch.log(1 - pred + 1e-7) * (1 - y_true))

        return log_prob

    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump({"U": self.U, "B": self.B, "V": self.V, "vocab": self.vocab, "rel": self.relation_names}, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            state = pkl.load(f)
            self.U = state["U"]
            self.B = state["B"]
            self.V = state["V"]
            self.vocab = state["vocab"]
            self.relation_names = state["rel"]

if __name__ == '__main__':
    vocab_size = 10000

    sample_size = 1000
    us = np.random.randint(vocab_size, size=sample_size)
    vs = np.random.randint(vocab_size, size=sample_size)
    #
    R = [[(int(ui), int(vi), int(np.random.random() > 0.9)) for ui, vi in zip(us, vs)] for _ in range(4)]


    sample_size_u = 1000
    #samplesU = [[(intlist(np.random.randint(vocab_size, size=sample_size_u)), intlist(np.random.randint(2, size=sample_size_u))) for _ in range(4)] for _ in range(vocab_size)]

    vocab = list(range(vocab_size))

    tst_model = Model(vocab, R, embedding_dimension = 5)

    tst_model.updateB()
    #tst_model.findBest(1, 2)

    #print(tst_model.B)
    tst_model.updateV()
    tst_model.updateU()