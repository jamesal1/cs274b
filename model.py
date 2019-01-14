import numpy as np
import torch
from probit_factor_model import train_logistic_torch, train_linear_torch
import pickle as pkl
from collections import defaultdict
import settings

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time

import pdb

def intlist(arr):
    return [int(i) for i in arr]


def subsample(li, num):
    indices = np.arange(len(li))
    np.random.shuffle(indices)
    return [li[ind] for ind in indices[:num]]


class ModelTorch(nn.Module):

    def __init__(self, vocab, relations, embedding_dimension = 50, logistic = False, lambdaB = 1e-3, lambdaUV = 1e-3,
        co_is_identity = False, sampling_scheme = "uniform", proportion_positive = 0.3, sample_size_B=100000):

        """
        Initialize the model and pre-process the relation matrices (for the sake of efficiency)
        Might take up to 10 minutes for a large vocabulary (30 000 words)

        Sampling schemes:
        Complete sampling - all possible uv pairs are considered for B update step,
        all words are sampled for the U and V updates.

        Uniform sampling uniformly subsamples UV pairs, as the name suggests.

        The problem with uniform and complete sampling is that some relations are very rare and thus the positive examples
        do not get sample enough. We assume another baseline distribution in order to deal with such cases.

        """

        super(ModelTorch, self).__init__()

        self.logistic = logistic

        self.lin = True  # Usually True  # linear embedding terms # Need to try both
        self.sampling_scheme = sampling_scheme  # either "uniform" or "proportional"
        if sampling_scheme not in ("uniform", "proportional", "complete"):
            raise ValueError("Unknown sampling scheme {}".format(sampling_scheme))

        if sampling_scheme == "proportional":
            self.proportion_positive = proportion_positive  # ignored for uniform sampling

        self.co_is_identity = co_is_identity
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.w_to_i = {}
        self.i_to_w = {}

        for w, i in self.vocab:
            self.w_to_i[w] = i
            self.i_to_w[i] = w
        self.emb_size = embedding_dimension
        self.b_size = embedding_dimension + (1 if self.lin else 0)

        self.relation_names = []
        self.R = []
        for k, v in relations.items():
            self.relation_names += [k]
            self.R += [v]
        print(self.relation_names)

        try:
            self.co_ind = self.relation_names.index('co')
        except:
            self.co_ind = -1

        self.B = [nn.Parameter(torch.randn((self.b_size, self.b_size)).cuda() * 0.001) for _ in range(len(relations))]
        if self.co_is_identity:
            self.B[self.co_ind] = nn.Parameter(torch.eye(self.b_size).cuda(), requires_grad=False)

        self.B = nn.ParameterList(self.B)
        self.U = nn.Parameter(torch.randn((self.vocab_size, embedding_dimension)).cuda() * 0.01)
        self.V = nn.Parameter(torch.randn((self.vocab_size, embedding_dimension)).cuda() * 0.01)

        self.sample_size_B = sample_size_B
        self.sample_size_w = 100  # for every relation

        self.lambdaB = lambdaB
        self.lambdaUV = lambdaUV
        start = time.time()
        self.Rpos = [set([(int(ui), int(vi)) for ui, vi, _ in r]) for r in self.R]
        print("Rpos initialization time: {}".format(time.time() - start))
        start = time.time()
        self.Rval = [dict([((int(ui), int(vi)), val) for ui, vi, val in r]) for i, r in
                     enumerate(self.R)]  # change 0
        print("Rval initialization time: {}".format(time.time() - start))


    def forward(self, us_ind, vs_ind, r_ind):

        if self.lin:
            U1 = torch.cat([self.U, nn.Parameter(torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda(),
                                                 requires_grad=False)], 1)
            V1 = torch.cat([self.V, nn.Parameter(torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda(),
                                                 requires_grad=False)], 1)
        else:
            U1 = self.U
            V1 = self.V

        #return U1, V1

        Us = U1[us_ind]
        Vs = V1[vs_ind]

        act = ((Us @ self.B[r_ind]) * Vs).sum(dim=1)

        return act

    def new_sample_neg_R(self, r_ind):
        """
        Sample negative examples for a given relation
        """

        # To do: deal with co-oc case (the potential no negative case).

        zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0

        ## NOT YET READY AT ALL
        desired_neg_sample_size = int(np.floor((1 - self.proportion_positive) * self.sample_size_B))

        us = np.random.randint(self.vocab_size, size=int(
            desired_neg_sample_size * 2))  # To do * 2 part is just a multiplier in order to give a chance to sample enough. The more - the better the precision. Modification needed. max sample size could be appropriate here
        vs = np.random.randint(self.vocab_size, size=int(desired_neg_sample_size * 2))

        r_neg = [(int(ui), int(vi), zero_value) for ui, vi in zip(us, vs) if
                 (int(ui), int(vi)) not in self.Rpos[r_ind]]

        # # Way to enforce a constant sample size, after all
        if len(r_neg) > desired_neg_sample_size:
            r_neg = subsample(r_neg, desired_neg_sample_size)
        elif len(r_neg) and len(r_neg) < desired_neg_sample_size:
            indices = np.random.choice(np.arange(len(r_neg)), size=desired_neg_sample_size, replace=True)
            r_neg = [r_neg[i] for i in indices]

        neg_weight = (desired_neg_sample_size / (len(r_neg)) if len(r_neg) else None)

        # neg_weight = 1
        r_neg = [x + (neg_weight,) for x in r_neg]

        return r_neg

    def get_samples_for_B(self, r_ind):
        """
        Obtain a sample for a given relation for a subsequent B update or likelihood estimation.
        """

        if self.sampling_scheme == "uniform":
            zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0
            us = np.random.choice(np.arange(self.vocab_size), size=self.sample_size_B, replace=True)
            vs = np.random.choice(np.arange(self.vocab_size), size=self.sample_size_B, replace=True)

            samples = [(u, v, self.Rval[r_ind][u, v], 1) if (u, v) in self.Rpos[r_ind] else (u, v, zero_value, 1)
                       for u, v in zip(us, vs)]
            return samples
        elif self.sampling_scheme == "complete":  # Falls on memory access on B update step
            zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0
            #            r_all = [(u, v, self.Rval[r_ind][u, v], 1) for u in range(self.vocab_size) for v in range(self.vocab_size)] # change to not grow the Rval dict
            samples = [(u, v, self.Rval[r_ind][u, v], 1) if (u, v) in self.Rpos[r_ind] else (u, v, zero_value, 1)
                       for u in range(self.vocab_size) for v in range(self.vocab_size)]
            return samples

        elif self.sampling_scheme == "proportional":

            desired_pos_sample_size = int(np.ceil(self.proportion_positive * self.sample_size_B))
            indices = np.random.choice(np.arange(len(self.R[r_ind])), size=desired_pos_sample_size, replace=True)
            pos_weight = 1

            res = [self.R[r_ind][ind] + (pos_weight,) for ind in indices] + self.new_sample_neg_R(r_ind)
            if len(res) != self.sample_size_B:
                print("Sample incomplete or too big: {} instead of {}".format(len(res), self.sample_size_B))
            return res

        else:
            raise ValueError("Unknown sampling scheme {}".format(self.sampling_scheme))

    def estimateLL(self, verbose=False):
        log_prob = torch.autograd.Variable(torch.zeros((1,)).cuda())

        correct = 0
        if verbose:
            print("Estimating log likelihood")
        for i, r in enumerate(self.R):
            ## Subsampling true relations
            uis, vis, rs, ws = zip(*self.get_samples_for_B(i))

            y_true = torch.FloatTensor(rs).cuda()

            act = self.forward(intlist(uis), intlist(vis), i)

            if self.logistic:
                pred = torch.sigmoid(act)

                log_prob += (torch.sum(torch.log(pred + 1e-7) * y_true) + torch.sum(
                    torch.log(1 - pred + 1e-7) * (1 - y_true)))
            else:
                pred = act
                y_true_var = Variable(y_true, requires_grad=False)
                ws_var = Variable(torch.FloatTensor(ws).cuda(), requires_grad=False)
                log_prob += -0.5 * torch.sum(ws_var * (pred - y_true_var) ** 2)

            log_prob -= self.lambdaB * 0.5 * (self.B[i] ** 2).sum()

            cur_correct = np.corrcoef(pred.data.cpu().numpy(), y_true.cpu().numpy())[0, 1]  # correlation
            if verbose:
                print("Correlation for factor {}: {}".format(self.relation_names[i], cur_correct))
            if cur_correct != cur_correct:
                print("nan correlation")
                cur_correct = 1

            correct += cur_correct
        log_prob -= self.lambdaUV * 0.5 * ((self.U ** 2).sum() + (self.V ** 2).sum())
        if verbose:
            print("Log prob: {}, accuracy: {}".format(log_prob, correct / len(self.R)))
        return log_prob, correct / len(self.R)

    def findBest(self, r, w, top=20, restrict=True):
        """
        Find closest pairs for a word in a given relation
        """
        raise NotImplemented #RposU not used anymore
        possible_us = self.RposU[r] # change? why restrict?
        possible_vs = self.RposV[r]

        w = self.w_to_i[w]
        if self.lin:
            U1 = torch.cat([self.U, nn.Parameter(torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda(),
                                                 requires_grad=False)], 1)
            V1 = torch.cat([self.V, nn.Parameter(torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda(),
                                                 requires_grad=False)], 1)
        else:
            U1 = self.U
            V1 = self.V

        vs = (U1[w].view(1, -1) @ self.B[r]) @ V1.t()
        us = (U1 @ (self.B[r] @ V1[w]))
        ubest = [(self.i_to_w[x], score) for x, score in zip(np.argsort(np.abs(1 - us.data.cpu().numpy())), np.sort(np.abs(1 - us.data.cpu().numpy()))) if x in possible_us][:top]
        vbest = [(self.i_to_w[x], score) for x, score in zip(np.argsort(np.abs(1 - vs.data.cpu().numpy())).flatten(), np.sort(np.abs(1 - vs.data.cpu().numpy())).flatten()) if x in possible_vs][:top]
        return ubest, vbest

    def getEmbeddingModel(self, relation=None):
        """
        Helper function to return embeddings in a proper format
        """
        if self.lin:
            U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
            V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        else:
            U1 = self.U
            V1 = self.V
        if relation is not None:
            U1 = U1 @ self.B[relation]
        embs = torch.cat([U1, V1], 1).cpu().numpy()
        return embs, dict([(self.i_to_w[i], embs[i]) for i in range(self.vocab_size)])

    def save(self, fname):
        with open(fname, "wb") as f:
           pkl.dump({"U": self.U, "V": self.V, "B": self.B, "lambdaB": self.lambdaB, "lambdaUV": self.lambdaUV, "co_is_identity": self.co_is_identity, "relation_names": self.relation_names, "lin": self.lin, "logistic": self.logistic, "vocab": self.vocab}, f)

    def load(self, filename):
        """
        Load model data from a file
        """
        with open(filename, "rb") as f:
            state = pkl.load(f)
            self.U = state["U"]
            self.B = state["B"]
            self.V = state["V"]
            self.vocab = state["vocab"]
            self.relation_names = state["relation_names"]
            self.lambdaB = state["lambdaB"]
            self.lambdaUV = state["lambdaUV"]
            self.co_is_identity = state["co_is_identity"]
            self.lin = state["lin"]
            self.logistic = state["logistic"]

    #convenience method that takes in triples or u,v lists and outputs the relationship activations.
    def getActivations(self, rel, *args):
        if len(args)==1:
            if len(args[0]) == 0:
                return []
            us, vs, _ = zip(*(args[0]))
        else:
            us = args[0]
            vs = args[1]
            if len(us) == 0:
                return []
        us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()
        return list(self.forward(us, vs, rel).data.cpu().numpy())

    def getScores(self, rel, *args):
        return self.getActivations(rel, *args)

    def __str__(self):

        path = "ModelBimodal_vocab_size{}_dim{}_lambdaB{}UV{}_logit{}_coId{}_sampling{}_pos{}_B{}"
        return path.format(self.vocab_size, self.emb_size, self.lambdaB,
                           self.lambdaUV,
                           self.logistic, self.co_is_identity, self.sampling_scheme,
                           self.proportion_positive,
                           self.sample_size_B)


class ModelUnimodal(ModelTorch):

    def estimateLL(self, verbose=False):
        log_prob = torch.autograd.Variable(torch.zeros((1,)).cuda())

        correct = 0

        if verbose:
            print("Estimating log likelihood")
        for i, r in enumerate(self.R):
            ## Subsampling true relations
            uis, vis, rs, ws = zip(*self.get_samples_for_B(i))

            y_true = torch.FloatTensor(rs).cuda()

            act = self.forward(intlist(uis), intlist(vis), i)

            y_true_var = Variable(y_true, requires_grad=False)
            ws_var = Variable(torch.FloatTensor(ws).cuda(), requires_grad=False)

            pred = act**2

            log_prob += torch.sum(-pred * (1-y_true_var) * ws_var) + torch.sum(
                torch.log(1 - torch.exp(-pred) + 1e-7) * y_true_var * ws_var)

            log_prob -= self.lambdaB * 0.5 * (self.B[i] ** 2).sum()

            cur_correct = np.corrcoef(pred.data.cpu().numpy(), y_true.cpu().numpy())[0, 1]  # correlation
            if verbose:
                print("Correlation for factor {}: {}".format(self.relation_names[i], cur_correct))
            if cur_correct != cur_correct:
                print("nan correlation")
                cur_correct = 1

            correct += cur_correct
        log_prob -= self.lambdaUV * 0.5 * ((self.U ** 2).sum() + (self.V ** 2).sum())
        if verbose:
            print("Log prob: {}, accuracy: {}".format(log_prob, correct / len(self.R)))
        return log_prob, correct / len(self.R)

    def __str__(self):
        path = "ModelUnimodalrev_vocab_size{}_dim{}_lambdaB{}UV{}_logit{}_coId{}_sampling{}_pos{}_B{}"
        return path.format(self.vocab_size,  self.emb_size, self.lambdaB,
                           self.lambdaUV,
                           self.logistic, self.co_is_identity, self.sampling_scheme,
                           self.proportion_positive,
                           self.sample_size_B)

    def getScores(self, rel, *args):
        return [np.abs(x) for x in self.getActivations(rel, *args)]

class ModelDistMatch1dUniform(ModelTorch):


    def energyDistance(self, activations, dist = None):
        size = activations.shape[0]//2
        x = activations[0:size]
        x_prime = activations[size:2*size]
        y = torch.rand(*x.shape) * 5 - 2.5
        y = Variable(y.cuda(), requires_grad=False)
        x_sort, x_arg = torch.sort(x, 0)
        x_prime_sort, x_prime_arg = torch.sort(x_prime, 0)
        y_sort, y_arg = torch.sort(y, 0)
        return 2 * torch.sum((x_sort - y_sort) ** 2) - torch.sum((x_sort - x_prime_sort) ** 2)

    def estimateLL(self, verbose=False):
        log_prob = torch.autograd.Variable(torch.zeros((1,)).cuda())

        correct = 0

        if verbose:
            print("Estimating log likelihood")
        for i, r in enumerate(self.R):
            ## Subsampling true relations
            samples = self.get_samples_for_B(i)
            #pos_uis, pos_vis, _, pos_ws = zip(*filter(lambda x:x[2],samples))
            neg_uis, neg_vis, _, neg_ws = zip(*filter(lambda x: not x[2], samples))
            uis, vis, rs, ws = zip(*samples)

            y_true = torch.FloatTensor(rs).cuda()

            act = self.forward(intlist(uis), intlist(vis), i)
            act_neg = self.forward(intlist(neg_uis), intlist(neg_vis), i)

            y_true_var = Variable(y_true, requires_grad=False)
            ws_var = Variable(torch.FloatTensor(ws).cuda(), requires_grad=False)

            pred = act**2

            hinge_threshold = 1

            # add something separate for co oc

            log_prob += torch.sum(-pred * y_true_var * ws_var) \
                        + 10 * torch.sum(torch.min(torch.zeros_like(act), torch.abs(act) - hinge_threshold) * (1 - y_true_var) * ws_var)
                        #+ torch.sum(torch.log(1 - torch.exp(-pred) + 1e-7) * (1 - y_true_var) * ws_var)



            log_prob -= self.energyDistance(act_neg) * 10

            log_prob -= self.lambdaB * 0.5 * (self.B[i] ** 2).sum()

            cur_correct = np.corrcoef(-pred.data.cpu().numpy(), y_true.cpu().numpy())[0, 1]  # correlation
            if verbose:
                print("Correlation for factor {}: {}".format(self.relation_names[i], cur_correct))
            if cur_correct != cur_correct:
                print("nan correlation")
                cur_correct = 1

            correct += cur_correct
        log_prob -= self.lambdaUV * 0.5 * ((self.U ** 2).sum() + (self.V ** 2).sum())
        if verbose:
            print("Log prob: {}, accuracy: {}".format(log_prob, correct / len(self.R)))
        return log_prob, correct / len(self.R)

    def __str__(self):
        path = "ModelDistMatch1dUniform_vocab_size{}_dim{}_lambdaB{}UV{}_logit{}_coId{}_sampling{}_pos{}_B{}"
        return path.format(self.vocab_size,  self.emb_size, self.lambdaB,
                           self.lambdaUV,
                           self.logistic, self.co_is_identity, self.sampling_scheme,
                           self.proportion_positive,
                           self.sample_size_B)

    def getScores(self, rel, *args):
        return [-np.abs(x) for x in self.getActivations(rel, *args)]


class ModelDistMatch2dUniform(ModelDistMatch1dUniform):

    def __init__(self, *args, **kwargs):
        self.uniform_range = kwargs.pop("uniform_range", 5)
        #self.negative_weight = kwargs.pop("negative_weight", 10)
        self.energy_weight = kwargs.pop("energy_weight", 1)
        self.energy_slice_count = kwargs.pop("energy_slice_count", 1) #number of slices to take in Sliced Wasserstein Distance
        #self.hinge_threshold = 1
        super().__init__(*args, **kwargs)
        self.modelName = "ModelDistMatch2dUniform"
        self.Btens = nn.ParameterList([nn.Parameter(torch.randn((self.b_size, self.b_size, 2)).cuda() * 0.1) for _ in range(len(self.relation_names))])

        if self.co_is_identity: ## TO DO
            self.Btens[self.co_ind] = nn.Parameter(torch.eye(self.b_size + (2,)).cuda(), requires_grad=False)


    #sliced wasserstein distance
    def energyDistance(self, activations, dist=None):
        size, dim = activations.shape
        size = size // 2

        x = activations[0:size, :]
        x_prime = activations[size:2*size, :]
        y = torch.rand(x.shape) * self.uniform_range - self.uniform_range / 2

        y = Variable(y.cuda(), requires_grad=False)
        dist = 0

        for _ in range(self.energy_slice_count):
            tmp = torch.randn((dim, 1)) # * 0 + 1

            tmp = tmp / torch.norm(tmp)
            projection = Variable(tmp.cuda(), requires_grad=False)

            x_proj, x_arg = torch.sort(x @ projection, 0)
            x_prime_proj, x_prime_arg = torch.sort(x_prime @ projection, 0)
            y_proj, y_arg = torch.sort(y @ projection, 0)

            dist += 2 * torch.mean((x_proj - y_proj) ** 2) - torch.mean((x_proj - x_prime_proj) ** 2)

            # Quick hack to add the orthogonality part (for 2d only)

            tmp2 = tmp.clone()
            tmp[0] = tmp2[1]
            tmp[1] = -tmp2[0]

            projection = Variable(tmp.cuda(), requires_grad=False)
            x_proj, x_arg = torch.sort(x @ projection, 0)
            x_prime_proj, x_prime_arg = torch.sort(x_prime @ projection, 0)
            y_proj, y_arg = torch.sort(y @ projection, 0)

            dist += torch.mean((x_proj - y_proj) ** 2) - torch.mean((x_proj - x_prime_proj) ** 2)

        return dist / self.energy_slice_count



    def estimateLL(self, new_samples=None, verbose=False):
        log_prob = torch.autograd.Variable(torch.zeros((1,)).cuda())

        correct = 0

        if verbose:
            print("Estimating log likelihood")
        for i, r in enumerate(self.R):


            ## Subsampling true relations
            if not new_samples:
                samples = self.get_samples_for_B(i)
            else:
                samples = new_samples[self.relation_names[i]]
                if not samples:
                    continue



            neg_uis, neg_vis, _, neg_ws = zip(*filter(lambda x: not x[2], samples))
            uis, vis, rs, ws = zip(*samples)

            y_true = torch.FloatTensor(rs).cuda()

            act = self.forward(intlist(uis), intlist(vis), i)
            act_neg = self.forward(intlist(neg_uis), intlist(neg_vis), i)

            y_true_var = Variable(y_true, requires_grad=False)
            ws_var = Variable(torch.FloatTensor(ws).cuda(), requires_grad=False)

            act_l2 = (act**2).sum(dim=1)
            # act_l1 = torch.abs(act).sum(dim=1)

            # add something separate for co oc

            pure_ll = torch.sum(-act_l2 * y_true_var * ws_var) + torch.sum(
                torch.log(1 - torch.exp(-act_l2) + 1e-7) * (1 - y_true_var) * ws_var)

            pure_ll = pure_ll / y_true.shape[0]

            #pure_ll = torch.mean(-act_l2 * y_true_var * ws_var) + torch.mean(act_l2 * (1 - y_true_var) * ws_var)


          #  loss_fn = torch.nn.SoftMarginLoss()
          #  pure_ll = loss_fn(act)

            log_prob += pure_ll
            #         + self.negative_weight * torch.sum(torch.min(torch.zeros_like(act_l1), act_l1 - self.hinge_threshold) * (1 - y_true_var) * ws_var)

           # pdb.set_trace()
            ED = self.energyDistance(act_neg)
            #print(ED)
            #pdb.set_trace()
            log_prob -= ED * self.energy_weight


            ll_reg_B = self.lambdaB * 0.5 * (self.Btens[i] ** 2).sum()
            log_prob -= ll_reg_B

            cur_correct = np.corrcoef(-act_l2.data.cpu().numpy(), y_true.cpu().numpy())[0, 1]  # correlation
            if verbose:
                print("Correlation for factor {}: {}".format(self.relation_names[i], cur_correct))
            if cur_correct != cur_correct:
                print("nan correlation")
                cur_correct = 1

            correct += cur_correct
        ll_reg_UV = self.lambdaUV * 0.5 * ((self.U ** 2).sum() + (self.V ** 2).sum())
        log_prob -= ll_reg_UV
        if verbose:
            print("Log prob: {}, accuracy: {}".format(log_prob, correct / len(self.R)))
        return log_prob, correct / len(self.R), (pure_ll, ED, ll_reg_B, ll_reg_UV)

    def __str__(self):
        path = "{}_vocab_size{}_dim{}_lambdaB{}UV{}_logit{}_coId{}_sampling{}_pos{}_B{}_UniRange{}_EnergyWeight{}_Slices{}"
        return path.format(self.modelName, self.vocab_size,  self.emb_size, self.lambdaB,
                           self.lambdaUV,
                           self.logistic, self.co_is_identity, self.sampling_scheme,
                           self.proportion_positive,
                           self.sample_size_B,
                           self.uniform_range,
                           self.energy_weight,
                           self.energy_slice_count
        )

    def getScores(self, rel, *args):
        return [-(x**2).sum() for x in self.getActivations(rel, *args)]


    def forward(self, us_ind, vs_ind, r_ind):

        if self.lin:
            U1 = torch.cat([self.U, nn.Parameter(torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda(),
                                                 requires_grad=False)], 1)
            V1 = torch.cat([self.V, nn.Parameter(torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda(),
                                                 requires_grad=False)], 1)
        else:
            U1 = self.U
            V1 = self.V

        Us = U1[us_ind]
        Vs = V1[vs_ind]

        act = ((Us @ self.Btens[r_ind]) * Vs.t()[:, :, None]).sum(dim=0) # change to Btens size (2, 51, 51)

        return act


    def findBest(self, r, w, top=20, restrict=True):

        if restrict:
            possible_us = self.RposU[r]  # change? why restrict?
            possible_vs = self.RposV[r]
        else:
            possible_us = list(range(self.vocab_size))
            possible_vs = list(range(self.vocab_size))

        w = self.w_to_i[w]

        acts_u = self.forward(possible_us, (w, ), r) # Check if r is ind or name. Change to appropriate index if needed
        acts_v = self.forward((w,), possible_vs, r)  # Check if r is ind or name. Change to appropriate index if needed

        acts_u = (acts_u ** 2).sum(dim=1).data.cpu().numpu()
        acts_v = (acts_v ** 2).sum(dim=1).data.cpu().numpu()

        ubest = [(self.i_to_w[x], score) for x, score in zip(np.argsort(acts_u),
                                                             np.sort(acts_u)) if x in possible_us][:top]

        vbest = [(self.i_to_w[x], score) for x, score in zip(np.argsort(acts_v),
                                                             np.sort(acts_v)) if x in possible_vs][:top]

        return ubest, vbest
