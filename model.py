import numpy as np
import torch
from probit_factor_model import train_logistic_torch, train_linear_torch
import pickle as pkl
from collections import defaultdict
import settings

import time


def intlist(arr):
    return [int(i) for i in arr]


def subsample(li, num):
    indices = np.arange(len(li))
    np.random.shuffle(indices)
    return [li[ind] for ind in indices[:num]]


class Model:

    def __init__(self, vocab, relations, embedding_dimension=50, logistic=False, lambdaB=1e-3, lambdaUV=1e-3,
                 co_is_identity=False, sampling_scheme="uniform", proportion_positive=0.3):
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

        self.lin = True # Usually True  # linear embedding terms # Need to try both
        self.sampling_scheme = sampling_scheme  # either "uniform" or "proportional"
        if sampling_scheme not in ("uniform", "proportional", "complete"):
            raise ValueError("Unknown sampling scheme {}".format(sampling_scheme))

        if sampling_scheme == "proportional":
            self.proportion_positive = proportion_positive  # ignored for uniform sampling

        self.logistic = logistic
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

        self.B = [torch.randn((self.b_size, self.b_size)).cuda() * 0.001] * len(relations)
        self.U = torch.randn((self.vocab_size, embedding_dimension)).cuda() * 0.01
        self.V = torch.randn((self.vocab_size, embedding_dimension)).cuda() * 0.01

        if self.co_is_identity:
            self.B[self.co_ind] = torch.eye(self.b_size).cuda()

        self.min_sample_size = 100  # was 1 k
        self.max_sample_size_B = 100000  # was 10k # was 1 M
        self.max_sample_size_w = 2000
        self.neg_sample_rate = 2

        self.sample_size_B = 100000
        self.sample_size_w = 100  # for every relation

        self.lambdaB = lambdaB
        self.lambdaUV = lambdaUV
        start = time.time()
        self.Rpos = [set([(int(ui), int(vi)) for ui, vi, _ in r]) for r in self.R]
        print("time", time.time() - start)
        start = time.time()
        self.Rval = [defaultdict(lambda: settings.zero_value if i != self.co_ind else 0,
                                 [((int(ui), int(vi)), val) for ui, vi, val in r]) for i, r in
                     enumerate(self.R)]  # change 0
        print("time", time.time() - start)
        start = time.time()
        self.RposU = []
        self.RposV = []
        self.RposU2 = [] # only store indexes
        self.RposV2 = [] # only store indexes
        for r in self.R:
            du = defaultdict(lambda: set())
            dv = defaultdict(lambda: set())

            du2 = defaultdict(lambda: set())
            dv2 = defaultdict(lambda: set())
            for ui, vi, rel in r:
                du[ui].add((int(vi), rel))
                dv[vi].add((int(ui), rel))
                du2[ui].add(int(vi))
                dv2[vi].add(int(ui))

            self.RposU += [du]
            self.RposV += [dv]

            self.RposU2 += [du]
            self.RposV2 += [dv]

        self.Rval = [dict(dd) for dd in self.Rval]
        self.RposU = [dict(dd) for dd in self.RposU]
        self.RposV = [dict(dd) for dd in self.RposV]
        self.RposU2 = [dict(dd) for dd in self.RposU2]
        self.RposV2 = [dict(dd) for dd in self.RposV2]

        print("time", time.time() - start)
        #
        # self.RposU = [{ui: set([(int(vi), rel) for utmp, vi, rel in r if utmp == ui]) for ui, _, _ in r} for r in self.R]
        # self.RposV = [{vi: set([(int(ui), rel) for ui, vtmp, rel in r if vtmp == vi]) for _, vi, _ in r} for r in self.R]
        #
        # self.global_neg_sample_weight = self.neg_sample_rate * sum(len(r) for r in self.R)  / len(self.R) / self.max_sample_size_B
        # self.global_sam

        # min(sum(len(r) for r in self.R) * self.neg_sample_rate / len(self.R), self.max_global_neg_samples) ? Should be a limit

    def sample_neg_R(self, r_ind):
        """
        Sample negative examples for a given relation
        """
        zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0

        #### old sampling ####
        ####
        # sample_size = min(len(self.Rpos[r_ind]), self.max_sample_size_B)
        # us = np.random.randint(self.vocab_size, size= self.neg_sample_rate * sample_size)
        # vs = np.random.randint(self.vocab_size, size= self.neg_sample_rate * sample_size)
        ######################

        possible_us = np.array(list(self.RposU[r_ind]))
        possible_vs = np.array(list(self.RposV[r_ind]))
        us = possible_us[np.random.randint(len(possible_us), size=int(
            self.max_sample_size_B))]  # just add completely random negative pairs here?
        vs = possible_vs[np.random.randint(len(possible_vs), size=int(self.max_sample_size_B))]

        r_neg = [(int(ui), int(vi), zero_value) for ui, vi in zip(us, vs) if
                 (int(ui), int(vi)) not in self.Rpos[r_ind]]  # why not filter before?
        neg_weight = (self.max_sample_size_B / (len(r_neg)) if len(r_neg) else 1)
        r_neg = [x + (neg_weight,) for x in r_neg]

        return r_neg

    def new_sample_neg_R(self, r_ind):
        """
        Sample negative examples for a given relation
        """

        # To do: deal with co-oc case (the potential no negative case).

        zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0

        ## NOT YET READY AT ALL
        desired_neg_sample_size = int(np.floor((1 - self.proportion_positive) * self.sample_size_B))

        us = np.random.randint(self.vocab_size, size=int(desired_neg_sample_size * 2)) # To do * 2 part is just a multiplier in order to give a chance to sample enough. The more - the better the precision. Modification needed. max sample size could be appropriate here
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

        #neg_weight = 1
        r_neg = [x + (neg_weight,) for x in r_neg]

        return r_neg

    def get_samples_for_B(self, r_ind):
        """
        Obtain a sample for a given relation for a subsequent B update or likelihood estimation.
        """
        ### Old sampling
        # zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0
        # r_all = [(u, v, self.Rval[r_ind][u,v], 1) for u in range(self.vocab_size) for v in range(self.vocab_size)]
        # return r_all

        ## Subsampling true relations

        # Sampling that worked
        # indices = np.arange(len(self.R[r_ind]))
        # np.random.shuffle(indices)
        # pos_weight = max(self.max_sample_size_B/len(self.R[r_ind]), 1)
        # return [self.R[r_ind][ind] + (pos_weight,) for ind in indices[:self.max_sample_size_B]] + self.sample_neg_R(r_ind)

        if self.sampling_scheme == "uniform":
            zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0
            us = np.random.choice(np.arange(self.vocab_size), size=self.sample_size_B, replace=True)
            vs = np.random.choice(np.arange(self.vocab_size), size=self.sample_size_B, replace=True)

            samples = [(u, v, self.Rval[r_ind][u, v], 1) if (u, v) in self.Rpos[r_ind] else (u, v, zero_value, 1)
                       for u, v in zip(us, vs)]
            return samples
        elif self.sampling_scheme == "complete": # Falls on memory access on B update step
            zero_value = settings.zero_value if self.relation_names[r_ind] != 'co' else 0
#            r_all = [(u, v, self.Rval[r_ind][u, v], 1) for u in range(self.vocab_size) for v in range(self.vocab_size)] # change to not grow the Rval dict
            samples = [(u, v, self.Rval[r_ind][u, v], 1) if (u, v) in self.Rpos[r_ind] else (u, v, zero_value, 1)
                       for u in range(self.vocab_size) for v in range(self.vocab_size)]
            return samples

        elif self.sampling_scheme == "proportional":

            # no replacement option: might create trouble
            # indices = np.arange(len(self.R[r_ind])) - no repetition might create trouble
            # np.random.shuffle(indices)
            # desired_pos_sample_size = np.ceil(self.proportion_positive * self.sample_size_B)
            # pos_weight = max(desired_pos_sample_size / len(self.R[r_ind]), 1)
            # return [self.R[r_ind][ind] + (pos_weight,) for ind in indices[:self.max_sample_size_B]] + self.new_sample_neg_R(r_ind)

            desired_pos_sample_size = int(np.ceil(self.proportion_positive * self.sample_size_B))
            indices = np.random.choice(np.arange(len(self.R[r_ind])), size=desired_pos_sample_size, replace=True)
            pos_weight = 1


            res = [self.R[r_ind][ind] + (pos_weight,) for ind in indices] + self.new_sample_neg_R(r_ind)
            if len(res) != self.sample_size_B:
                print("Sample incomplete or too big: {} instead of {}".format(len(res), self.sample_size_B))
            return res

        else:
            raise ValueError("Unknown sampling scheme {}".format(self.sampling_scheme))

    def get_word_proportion_positive(self, w_ind, rel_ind, sample_for_U_update):
        total_positive = len(self.R[rel_ind])
        total_possible = self.vocab_size**2
        if sample_for_U_update:
            positive_for_a_word = len(self.RposU[rel_ind][w_ind])  if w_ind in self.RposU[rel_ind] else 0
        else:
            positive_for_a_word = len(self.RposV[rel_ind][w_ind]) if w_ind in self.RposV[rel_ind] else 0

        word_load = positive_for_a_word / total_positive

        return (word_load * self.proportion_positive) / (word_load * self.proportion_positive + (self.vocab_size - positive_for_a_word)/(total_possible - total_positive) * (1-self.proportion_positive))

    def get_word_prob_selected(self, w_ind, rel_ind, sample_for_U_update):
        total_positive = len(self.R[rel_ind])
        total_possible = self.vocab_size**2
        if sample_for_U_update:
            positive_for_a_word = len(self.RposU[rel_ind][w_ind])  if w_ind in self.RposU[rel_ind] else 0
        else:
            positive_for_a_word = len(self.RposV[rel_ind][w_ind]) if w_ind in self.RposV[rel_ind] else 0

        word_load = positive_for_a_word / total_positive

        return (word_load * self.proportion_positive + (self.vocab_size - positive_for_a_word)/(total_possible - total_positive) * (1-self.proportion_positive))


    def get_samples_for_w(self, w_ind, sample_for_U_update=True):
        """
        Obtain a sample for the V or U embeddings update step
        Returns a list of triples. First element of a triple is a list of related words,
        second - list of relation values, third - list of weights
        """

        # Sampling that works
        # Rs = []
        # for i, rel in enumerate(self.RposU if sample_for_U_update else self.RposV):
        #     possible_vis = np.array(list(self.RposV[i] if sample_for_U_update else self.RposU[i]))
        #     #fixed_neg = min(len(self.Rpos[i]), self.max_sample_size_B) * self.neg_sample_rate / self.vocab_size ## old sampling
        #
        #     if w_ind in rel:
        #         natural_pos_weight = self.max_sample_size_B / len(self.R[i])
        #         rel_w = rel[w_ind]
        #         pos_sample_size = min(len(rel_w), self.max_sample_size_w)
        #         vis, rs = zip(*subsample(list(rel_w), pos_sample_size))
        #         pos_len = len(vis)
        #         pos_weights = natural_pos_weight * max(len(rel_w) / self.max_sample_size_w, 1)
        #         natural_neg_weight = self.max_sample_size_B / (len(rel) * len(possible_vis) - len(self.R[i]))
        #         non_rel_w = [x for x in possible_vis if x not in rel_w]
        #
        #         neg_samples = [int(non_rel_w[x]) for x in np.random.randint(len(non_rel_w), size=self.max_sample_size_w)] if len(non_rel_w) else []
        #         neg_weights = (natural_neg_weight * len(non_rel_w) / self.max_sample_size_w)
        #     else:
        #         vis = []
        #         rs = []
        #         pos_len = 0
        #         pos_weights = 1
        #         neg_samples = [] # or the alternative below # faills if neg samples are empty and only co is used
        #         #neg_samples = intlist(np.random.randint(self.vocab_size, size=self.max_sample_size_w))
        #         neg_weights = 1
        #
        #     vis = list(vis) + neg_samples
        #     if self.relation_names[i] != 'co':
        #         rs = list(rs) + [settings.zero_value] * len(neg_samples) # change 0
        #     else:
        #         rs = list(rs) + [0] * len(neg_samples)  # change 0
        #     ws = [pos_weights] * pos_len + [neg_weights] * len(neg_samples)

        #######################
        ##### full sample #####
        #######################

        # if w_ind in rel:
        #     rel_w = rel[w_ind]
        #     vis, rs = zip(*list(rel_w))
        #
        # else:
        #     vis = ()
        #     rs = ()
        #
        # vset = set(vis)
        # new_vis, new_rs = zip(*[(i, 0) for i in range(self.vocab_size) if i not in vset])
        # vis = vis + new_vis
        # rs = rs + new_rs
        # ws = [1] * len(vis)
        ##########################
        ##### full sample end #####
        ##########################

        # Rs += [(vis, rs, ws)]

        # New sampling

        if self.sampling_scheme == "uniform":
            Rs = []
            for i, rel in enumerate(self.RposU if sample_for_U_update else self.RposV):

                zero_value = settings.zero_value if self.relation_names[i] != 'co' else 0

                sampled_inds = np.random.choice(np.arange(self.vocab_size), size=self.sample_size_w, replace=False)
                #sampled_inds = np.random.choice(np.arange(self.vocab_size), size=self.sample_size_w, replace=False)
                #sampled_inds = sorted(sampled_inds)
                #sampled_inds = subsample(np.arange(self.vocab_size), self.sample_size_w)
                #sampled_inds  = intlist(np.arange(self.vocab_size))
                ws = [1] * self.sample_size_w

                if sample_for_U_update:
                    rs = [self.Rval[i][w_ind, v_ind] if (w_ind, v_ind) in self.Rpos[i] else zero_value
                          for v_ind in sampled_inds]
                else:
                    rs = [self.Rval[i][u_ind, w_ind] if (u_ind, w_ind) in self.Rpos[i] else zero_value
                          for u_ind in sampled_inds]

                Rs += [(intlist(sampled_inds), rs, ws)]
        elif self.sampling_scheme == "complete":
            Rs = []
            for i, rel in enumerate(self.RposU if sample_for_U_update else self.RposV):
                zero_value = settings.zero_value if self.relation_names[i] != 'co' else 0
                if w_ind in rel:
                    rel_w = rel[w_ind]
                    vis, rs = zip(*list(rel_w))

                else:
                    vis = ()
                    rs = ()

                vset = set(vis)
                if len(vset) < self.vocab_size: # For the co-oc case, when there is nothing to unzip otherwise
                    new_vis, new_rs = zip(*[(i, zero_value) for i in range(self.vocab_size) if i not in vset])
                    vis = vis + new_vis
                    rs = rs + new_rs
                ws = [1] * len(vis)

                Rs += [(vis, rs, ws)]

        elif self.sampling_scheme == "proportional":

            Rs = []
            for i, rel in enumerate(self.RposU if sample_for_U_update else self.RposV):
                zero_value = settings.zero_value if self.relation_names[i] != 'co' else 0

                if w_ind in rel:
                    rel_w = rel[w_ind]
                    vis, rs = zip(*list(rel_w))
                else:
                    vis = ()
                    rs = ()

                prop_positive_for_a_word = self.get_word_proportion_positive(w_ind, i, sample_for_U_update)

                expected_number_of_positive = int(np.ceil(self.sample_size_w * prop_positive_for_a_word))
                expected_number_of_negative = int(np.floor(self.sample_size_w * (1-prop_positive_for_a_word)))

                # print("Expected number of positive examples is {}".format(expected_number_of_positive))
                # print("Expected number of negative examples is {}".format(expected_number_of_negative))

                # determine the number of samples here
                if len(vis) or len(vis) > expected_number_of_positive:
                    sub_ind = np.random.choice(np.arange(len(vis)), size=expected_number_of_positive, replace=True) # replace or not?
                    vissub = [vis[i] for i in sub_ind]
                    rssub = [rs[i] for i in sub_ind]
                else:
                    vissub = list(vis)
                    rssub = list(rs)

                vset = set(vis)
                tmp = [(i, zero_value) for i in range(self.vocab_size) if i not in vset]
                if expected_number_of_negative > 0: # add try except for no negative?
                    neg_vis, neg_rs = zip(*tmp)
                    neg_vis, neg_rs = list(neg_vis), list(neg_rs)
                else:
                    neg_vis = []
                    neg_rs = []

                if len(neg_vis) or len(neg_vis) > expected_number_of_negative:
                    sub_ind = np.random.choice(np.arange(len(neg_vis)), size=expected_number_of_negative, replace=True) # replace or not?
                    neg_vissub = [neg_vis[i] for i in sub_ind]
                    neg_rssub = [neg_rs[i] for i in sub_ind]
                else:
                    neg_vissub = neg_vis
                    neg_rssub = neg_rs


                pos_w = expected_number_of_positive / len(vissub) if len(vissub) else None

                # if pos_w is not None:
                #     B_step_desired_pos_sample_size = np.ceil(self.proportion_positive * self.sample_size_B)
                #     pos_weight_B_step = max(B_step_desired_pos_sample_size / len(self.R[i]), 1)
                #     pos_w *= pos_weight_B_step

                neg_w = expected_number_of_negative / len(neg_vissub) if len(neg_vissub) else None

                baseline_rate_w = self.get_word_prob_selected(w_ind, i, sample_for_U_update)
                baseline_weight = baseline_rate_w * self.sample_size_B / self.sample_size_w

                if pos_w is not None:
                    pos_w *= baseline_weight
                if neg_w is not None:
                    neg_w *= baseline_weight




                ws = [pos_w] * len(vissub) + [neg_w] * len(neg_vissub)

                vissub = vissub + neg_vissub
                rssub = rssub + neg_rssub

                Rs += [(vissub, rssub, ws)]

        else:
            raise ValueError("Unknown sampling scheme {}".format(self.sampling_scheme))

        return Rs

    def updateB(self):
        """
        Perform the B update step
        """

        if self.lin:
            U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
            V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        else:
            U1 = self.U
            V1 = self.V

        for i, r in enumerate(self.R):
            if self.co_is_identity and i == self.co_ind:
                continue
            uis, vis, rs, ws = zip(*self.get_samples_for_B(i))
            UVs = (U1[intlist(uis)].view(-1, self.b_size, 1)) * (V1[intlist(vis)].view(-1, 1, self.b_size))
            UVs = UVs.view(-1, self.b_size ** 2)
            y = torch.FloatTensor(rs).cuda()
            ws = torch.FloatTensor(ws).cuda()
            if self.logistic:
                self.B[i] = train_logistic_torch(UVs, y, thetas=self.B[i].view(-1), reg=self.lambdaB).view(self.b_size,
                                                                                                           self.b_size)
            else:
                self.B[i] = train_linear_torch(UVs, y, thetas=self.B[i].view(-1), weights=ws, reg=self.lambdaB).view(
                    self.b_size, self.b_size)

    def updateU(self):
        """
        Perform the U update step
        """

        if self.lin:
            V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        else:
            V1 = self.V
        for w in range(self.vocab_size):
            # start = time.time()
            all_X = []
            all_y = []
            all_weights = []
            for i, r in enumerate(self.get_samples_for_w(w, sample_for_U_update=True)):
                vis, rs, ws = r
                if vis:
                    all_X += [V1[list(vis)] @ self.B[i].t()]
                    all_y += [torch.FloatTensor(rs).cuda()]
                    all_weights += [torch.FloatTensor(ws).cuda()]
            Xb = torch.cat(all_X)  # gives error when there is no CO

            if self.lin:
                X = Xb[:, :-1]
                b = Xb[:, -1]
            else:
                X = Xb
                b = 0
            y = torch.cat(all_y)
            weight = torch.cat(all_weights)
            # print(X.shape, y.shape, b.shape)
            if self.logistic:
                self.U[w] = train_logistic_torch(X, y, thetas=self.U[w], b=b, reg=self.lambdaUV)
            else:
                self.U[w] = train_linear_torch(X, y, thetas=self.U[w], weights=weight, b=b, reg=self.lambdaUV)
                # self.U[w] = train_linear_torch(X, y, thetas=self.U[w], b=b, reg=self.lambdaUV)
            # print(time.time() - start)

    def updateV(self):
        """
        Perform the V update step
        """

        if self.lin:
            U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        else:
            U1 = self.U
        for w in range(self.vocab_size):
            # start = time.time()
            all_X = []
            all_y = []
            all_weights = []
            for i, r in enumerate(self.get_samples_for_w(w, sample_for_U_update=False)):
                uis, rs, ws = r
                if uis:
                    all_X += [U1[list(uis)] @ self.B[i]]
                    all_y += [torch.FloatTensor(rs).cuda()]
                    all_weights += [torch.FloatTensor(ws).cuda()]
            Xb = torch.cat(all_X)
            if self.lin:
                X = Xb[:, :-1]
                b = Xb[:, -1]
            else:
                X = Xb
                b = 0

            y = torch.cat(all_y)
            weight = torch.cat(all_weights)
            if self.logistic:
                self.V[w] = train_logistic_torch(X, y, thetas=self.V[w], b=b, reg=self.lambdaUV)
            else:
                self.V[w] = train_linear_torch(X, y, thetas=self.V[w], weights=weight, b=b, reg=self.lambdaUV)
                # self.V[w] = train_linear_torch(X, y, thetas=self.V[w], b=b, reg=self.lambdaUV)
            # print(time.time() - start)

    def findBest(self, r, w, top=20, restrict=True):
        """
        Find closest pairs for a word in a given relation
        """
        possible_us = self.RposU[r]
        possible_vs = self.RposV[r]
        w = self.w_to_i[w]
        if self.lin:
            U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
            V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        else:
            U1 = self.U
            V1 = self.V
        print(U1.size())
        print(V1.size())
        print(self.B[r].size())

        vs = (U1[w].view(1, -1) @ self.B[r]) @ V1.t()
        us = (U1 @ (self.B[r] @ V1[w]))
        ubest = [self.i_to_w[x] for x in np.argsort(np.abs(1 - us.cpu().numpy())) if x in possible_us][:top]
        vbest = [self.i_to_w[x] for x in np.argsort(np.abs(1 - vs.cpu().numpy())).flatten() if x in possible_vs][:top]
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

    def estimateLL(self):
        """
        Estimate log likelihood of the data and the correlation with the correct answer (based on 10 000 samples)
        """
        if self.lin:
            U1 = torch.cat([self.U, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
            V1 = torch.cat([self.V, torch.FloatTensor(torch.ones((self.vocab_size, 1))).cuda()], 1)
        else:
            U1 = self.U
            V1 = self.V

        log_prob = 0

        correct = 0
        print("Estimating log likelihood")
        for i, r in enumerate(self.R):
            ## Subsampling true relations
            uis, vis, rs, ws = zip(*self.get_samples_for_B(i))
            Us = U1[intlist(uis)]
            Vs = V1[intlist(vis)]
            # UVs = (Us.view(-1, self.b_size, 1)) * (Vs.view(-1, 1, self.b_size))
            # UVs = UVs.view(-1, self.b_size**2)
            y_true = torch.FloatTensor(rs).cuda()

            # y_true = torch.FloatTensor([self.Rval[i][(k, j)] for (k, j) in zip(uis, vis)]).cuda()

            act = ((Us @ self.B[i]) * Vs).sum(dim=1)

            if self.logistic:
                pred = torch.sigmoid(act)

                log_prob += (torch.sum(torch.log(pred + 1e-7) * y_true) + torch.sum(
                    torch.log(1 - pred + 1e-7) * (1 - y_true)))
            else:
                pred = act
                log_prob += -0.5 * torch.sum(torch.FloatTensor(ws).cuda() * (pred - y_true) ** 2)
            log_prob -= self.lambdaB * 0.5 * (self.B[i] ** 2).sum()
            # cur_correct = (torch.round(pred) == y_true).sum()
            cur_correct = np.corrcoef(pred.cpu().numpy(), y_true.cpu().numpy())[0, 1]  # correlation
            print("Correlation for factor {}: {}".format(self.relation_names[i], cur_correct))
            if cur_correct != cur_correct:
                print("nan correlation")
                cur_correct = 1
            # print(y_true.shape, pred.shape)
            # print("Accuracy for the factor {} is {}".format(self.relation_names[i], cur_correct/samples))

            correct += cur_correct
        log_prob -= self.lambdaUV * 0.5 * ((self.U ** 2).sum() + (self.V ** 2).sum())
        # return log_prob, correct/(samples*len(self.R))
        print("Log prob: {}, accuracy: {}".format(log_prob, correct / len(self.R)))
        return log_prob, correct / len(self.R)

    def save(self, filename):
        """
        Save the crucial model components to a file
        """
        with open(filename, "wb") as f:
            pkl.dump({"U": self.U, "B": self.B, "V": self.V, "vocab": self.vocab, "rel": self.relation_names}, f)

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
            self.relation_names = state["rel"]
