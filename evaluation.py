import pathlib
from model import ModelTorch, ModelUnimodal, ModelDistMatch1dUniform, ModelDistMatch2dUniform
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import settings
import random
import pdb

plt.ioff() # avoid showing figures by default

from utils import getCooccurrenceMatrix, getVocab, getRelationNumericTriples

from sklearn.metrics import roc_auc_score


def getNegativeUVs(triples):
    if len(triples) == 0:
        return [], []
    us, vs, _ = zip(*triples)
    us_neg = np.random.choice(settings.vocab_size, len(us))
    vs_neg = np.random.choice(settings.vocab_size, len(us))

    return list(us) + [int(x) for x in us_neg], [int(x) for x in vs_neg] + list(vs)

def compareHistograms(activations, title, path):
    bins = np.linspace(-2, 2, 300)

    fig = plt.figure()
    plt.title(title)
    plt.hist(activations[0][0], bins=bins, alpha=0.5, label=activations[0][1], color="green")
    plt.hist(activations[1][0], bins=bins, alpha=0.5, label=activations[1][1], color="red")
    # plt.hist(np.arange(100), alpha=0.5, label="test", color="red")
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(path+title+".png") #BAD
    plt.close(fig)

def getAUC(model, relation_triples_train, relation_triples_test, path, iteration=0):

    if iteration == 0:
        with open(path + "AUC.csv".format(iteration), "w") as f:
            f.write("iteration")
            for r in model.relation_names:
                f.write(",{}_train_AUC,{}_test_AUC".format(r, r))

            f.write("\n")

    with open(path + "AUC.csv".format(iteration), "a") as f:
        f.write(str(iteration))
        #f.write("relation,train_AUC,test_AUC\n")

        for i, r in enumerate(model.relation_names):

           # print("Calculating AUC for relation " + r)
            if r == "co":
                continue

           # if relation_triples_train[r]:
            scores_pos_train = model.getScores(i, relation_triples_train[r])
            scores_neg_train = model.getScores(i, *getNegativeUVs(relation_triples_train[r]))

            scores_train = np.hstack([scores_pos_train, scores_neg_train])

            true_ans_train = np.hstack([np.ones_like(scores_pos_train),
                                        np.zeros_like(scores_neg_train)])

            try:
                train_auc = roc_auc_score(true_ans_train, scores_train)
            except ValueError:
                train_auc = None

            # else:
            #     print("Not enough train data")
            #     train_auc = None

         #   if relation_triples_test[r]:
            scores_pos_test = model.getScores(i, relation_triples_test[r])
            scores_neg_test = model.getScores(i, *getNegativeUVs(relation_triples_test[r]))

            scores_test = np.hstack([scores_pos_test, scores_neg_test])


            true_ans_test = np.hstack([np.ones_like(scores_pos_test),
                                        np.zeros_like(scores_neg_test)])


            try:
                test_auc = roc_auc_score(true_ans_test, scores_test)
            except ValueError:
                test_auc = None
            # else:
            #     test_auc = None

            f.write(",{},{}".format(train_auc, test_auc))
            print(r, train_auc, test_auc)
            #return train_auc, test_auc
        f.write("\n")


### Load cached matrices from file, or create them

relation_triples = getRelationNumericTriples(settings.vocab_size)


# Subsetting relations so that there are enough positive examples
# relation_triples_old = relation_triples
# relation_triples = {}
#
# relation_triples['/r/IsA'] = relation_triples_old['/r/IsA']
# relation_triples['/r/Antonym'] = relation_triples_old['/r/Antonym']
# relation_triples['/r/Synonym'] = relation_triples_old['/r/Synonym']
####



cooccurrence_triples = getCooccurrenceMatrix(settings.vocab_size)

### scale the co-occurrence (specify a way in settings!)
_, _, c = zip(*cooccurrence_triples)
co_mean = np.mean(np.log1p(c))
# make mean covariance to be 0.5. Think about a more principled choice
cooccurrence_triples = [(a, b, np.log1p(c)/(2 * co_mean)) for a, b, c in cooccurrence_triples]
### subsample the relation matrix, split into train-test

relation_triples_train = {}
relation_triples_test = {}
for relation, relation_triples in relation_triples.items():
    l = len(relation_triples)
    sample_indices = np.arange(l)
    np.random.shuffle(sample_indices)
    test_count = int(settings.test_frac * l)
    if test_count < 10:
        print("Not enough test samples for {}: {}".format(relation, test_count))
    relation_triples_test[relation] = [relation_triples[i] for i in sample_indices[:test_count]]
    relation_triples_train[relation] = [relation_triples[i] for i in sample_indices[test_count:]]

#relation_triples_train["co"] = cooccurrence_triples
#relation_triples_test["co"] = cooccurrence_triples

#
# ### load and create the model
# mt = ModelDistMatch1dUniform(getVocab(settings.vocab_size), relation_triples_train, embedding_dimension=settings.embedding_dimension, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
#                 logistic=settings.logistic, co_is_identity=settings.co_is_identity,
#                 sampling_scheme=settings.sampling_scheme,
#                 proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)
#
# path = "evaluation/{}_test{}_seed{}/"
# path = path.format(str(mt),settings.test_frac, settings.seed)
# pathlib.Path(path).mkdir(parents=True, exist_ok=True)
#
#
# ### train the model
# optimizer = torch.optim.Adam(mt.parameters())
# lls = []
# accs = []
#
# print_every = 50
#
#
# for i in range(0):#range(settings.epochs):
#
#     if i % print_every == 0:
#         print("#######################")
#         print("Update {}".format(i))
#         print("#######################")
#         ### evaluate performance, save the report in a readable form
#         getAUC(mt, relation_triples_train, relation_triples_test, path, i)
#
#     ll, acc = mt.estimateLL(verbose= (i % print_every == 0))
#     lls.append(ll.data)
#     accs.append(acc)
#     nll = -ll
#     nll.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     if i % print_every == 0:
#         print("#######################")
#         print("Update {}".format(i))
#         print("#######################")
# mt.save(path+"model.pkl")
#
# ### Print likelihoods
# fig = plt.figure()
# plt.plot(lls)
# plt.xlabel("iteration")
# plt.ylabel("log likelihood")
#
# plt.savefig(path+"ll.png")
# plt.close(fig)
#
# fig = plt.figure()
# plt.plot(accs)
# plt.xlabel("iteration")
# plt.ylabel("correlation with correct answers")
# plt.savefig(path+"corr.png")
# plt.close(fig)

###
#
# train_acts = []
# test_acts = []
# train_acts_neg = []
# test_acts_neg = []
#
# for i, r in enumerate(mt.relation_names):
#     print(r)
#     if r == "co":
#         continue
#
#     train_acts.extend(mt.getActivations(i, relation_triples_train[r]))
#     test_acts.extend(mt.getActivations(i, relation_triples_test[r]))
#     train_acts_neg.extend(mt.getActivations(i, *getNegativeUVs(relation_triples_train[r])))
#     test_acts_neg.extend(mt.getActivations(i, *getNegativeUVs(relation_triples_test[r])))
#
# ## !save settings alongside the model
# bins = np.linspace(-2, 2, 300)
#
# compareHistograms([(train_acts,"train"),(test_acts,"test")],"traintest", path)
#
# ## Compare with negatives
# compareHistograms([(train_acts,"train positive"),(train_acts_neg,"train negative")], "train", path)
# compareHistograms([(test_acts,"test positive"),(test_acts_neg,"test negative")], "test", path)
#
# ## Compare with negatives for specific relations
#
# for i, r in enumerate(mt.relation_names):
#
#     print(r)
#     if r == "co":
#         continue
#
#
#     train_acts = mt.getActivations(i, relation_triples_train[r])
#     test_acts = mt.getActivations(i, relation_triples_test[r])
#
#     train_acts_neg = mt.getActivations(i, *getNegativeUVs(relation_triples_train[r]))
#     test_acts_neg = mt.getActivations(i, *getNegativeUVs(relation_triples_test[r]))
#     ## Try full negative?
#
#     compareHistograms([(train_acts, "train positive"), (train_acts_neg, "train negative")], 'Train_answers_{}'.format(r.split("/")[-1]), path)
#     compareHistograms([(test_acts, "test positive"), (test_acts_neg, "test negative")], 'Test_answers_{}'.format(r.split("/")[-1]), path)
#
#
# ###

mt2d = ModelDistMatch2dUniform(getVocab(settings.vocab_size), relation_triples_train, embedding_dimension=settings.embedding_dimension, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
                logistic=settings.logistic, co_is_identity=settings.co_is_identity,
                sampling_scheme=settings.sampling_scheme,
                proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)

def plot_2d_acts(ax, pos_acts, neg_acts, rel_name):

   if len(pos_acts) == 0 or len(neg_acts) == 0:
       return

   all_acts = np.vstack([pos_acts, neg_acts])
   colors = np.zeros((all_acts.shape[0], 3))
   colors[0:pos_acts.shape[0], 1] = 1
   colors[pos_acts.shape[0]:, 0] = 1
   scatterplot = ax.scatter(all_acts[:, 0], all_acts[:, 1], c=colors, alpha=0.15)
   ax.set_title(rel_name)


   return scatterplot

    # xmin, xmax, ymin, ymax = -3, 3, -3, 3
    #
    # fig, ax1 = plt.subplots(1, 1, sharey=True)
    #
    # ax1.set_xlabel("x")
    # ax1.set_ylabel("y")
    #
    # generated = all_acts
    #
    # xgen, ygen = generated[:, 0], generated[:, 1]
    #
    # #pdb.set_trace()
    #
    # counts, ybins, xbins, image = ax1.hist2d(xgen, ygen, bins=(50, 50), cmap=plt.cm.Reds, normed=True, range=[[xmin, xmax], [ymin, ymax]])
    #
    # #CS = ax1.contourf(X, Y, preds.reshape(X.shape), extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
    # #                  cmap=plt.cm.BuPu, alpha=0.5)  # linewidths=3,
    #
    # #cbar = fig.colorbar(CS)
    # #cbar.ax.set_ylabel('Discriminator output')
    #
    # fig.show()

def plot_2d_act_dists(ax, pos_acts, neg_acts, rel_name):

   if len(pos_acts) == 0 or len(neg_acts) == 0:
       return

   all_acts = np.vstack([pos_acts, neg_acts])

   pos_dists = np.sqrt(np.sum(pos_acts**2, 1))
   neg_dists = np.sqrt(np.sum(neg_acts ** 2, 1))

   hist = ax.hist(pos_dists, density=True, alpha=0.5, facecolor="g", label="Positive")
   hist2 = ax.hist(neg_dists, density=True, alpha=0.5, facecolor="r", label="Negative")
   ax.set_title(rel_name)


   return ax


def create_acts(m, relation_triples_train, relation_triples_test):

    '''m - model instance, implementing get activations method'''

    for i, r in enumerate(m.relation_names):

        train_acts = []
        test_acts = []
        train_acts_neg = []
        test_acts_neg = []

        #print(r)
        if r == "co":
           continue

        train_acts.extend(m.getActivations(i, relation_triples_train[r]))
        test_acts.extend(m.getActivations(i, relation_triples_test[r]))
        train_acts_neg.extend(m.getActivations(i, *getNegativeUVs(relation_triples_train[r])))
        test_acts_neg.extend(m.getActivations(i, *getNegativeUVs(relation_triples_test[r])))

        train_acts, test_acts, train_acts_neg, test_acts_neg = map(np.array, [train_acts, test_acts, train_acts_neg, test_acts_neg])

        fig, ax = plt.subplots()
        scatter = plot_2d_acts(ax, test_acts, test_acts_neg, r)
        fig.show()

        fig, ax = plt.subplots()
        scatter = plot_2d_act_dists(ax, test_acts, test_acts_neg, r)
        fig.show()

       # return

    return train_acts, test_acts, train_acts_neg, test_acts_neg

def append_weight_to_triples(relation_triples):
    # simply appends 1 to every triple (since log likelihood requires weights

    return {k: list(map(lambda triple: triple + (1,), v)) for k, v in relation_triples.items()}



##

path = "./TemporaryReport12_Jan_2019/"

## Initial evaluations for the quantities of interest
lp, corr, detail = mt2d.estimateLL()
pure_lls, EDs, reg_Bs, reg_UVs = map(lambda x: [x.data.cpu().numpy()], detail)
lls = [lp.data.cpu().numpy()]
accs = [corr]

## Test set
relation_quadruples_test = append_weight_to_triples(relation_triples_test)

for k, v in relation_quadruples_test.items():
    num_neg_elts = len(v) * 2
    us = np.random.randint(mt2d.vocab_size, size=num_neg_elts)
    vs = np.random.randint(mt2d.vocab_size, size=num_neg_elts)
    ws = (1, ) * len(us)
    rs = (0, ) * len(us)

    quadruples = list(zip(us, vs, rs, ws))
    v.extend(quadruples)

lp_test, corr_test, detail_test = mt2d.estimateLL(relation_quadruples_test)
pure_lls_test, EDs_test, reg_Bs_test, reg_UVs_test = map(lambda x: [x.data.cpu().numpy()], detail_test)
lls_test = [lp_test.data.cpu().numpy()]
accs_test = [corr_test]

del lp_test
del corr_test
del detail_test

optimizer = torch.optim.Adam(mt2d.parameters())

### train the model

print_every = 50

# For best results - change energy weight back to 1, remove 2 * factor from the energy distance


for i in range(30000, 40000):#range(settings.epochs):

    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")
        ### evaluate performance, save the report in a readable form
        getAUC(mt2d, relation_triples_train, relation_triples_test, path, i)
##        getAUC(mt, relation_triples_train, relation_triples_test, path, i)
        mt2d.save(path + str(mt2d) + "_update_{}.model.pkl".format(i))

    ll, acc, detail = mt2d.estimateLL(verbose=(i % print_every == 0))
    lls.append(ll.data.cpu().numpy())
    accs.append(acc)
    nll = -ll
    nll.backward()

    for history, new_el in zip((pure_lls, EDs, reg_Bs, reg_UVs), map(lambda x: x.data.cpu().numpy(), detail)):
        history.append(new_el)

    optimizer.step()
    optimizer.zero_grad()

    del ll
    del acc
    del detail

    lp_test, corr_test, detail_test = mt2d.estimateLL(new_samples=relation_quadruples_test, verbose=(i % print_every == 0))
    lls_test.append(lp_test.data.cpu().numpy())
    accs_test.append(corr_test)

    for history, new_el in zip((pure_lls_test, EDs_test, reg_Bs_test, reg_UVs_test), map(lambda x: x.data.cpu().numpy(), detail_test)):
        history.append(new_el)

    del lp_test
    del corr_test
    del detail_test

    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")

create_acts(mt2d, relation_triples_train, relation_triples_test)

def print_likelihoods(pure_lls, EDs, reg_Bs, reg_UVs, title=None):

    fig, ax = plt.subplots()

    step_num = list(range(len(pure_lls)))
    ax.plot(step_num, pure_lls, label="Model likelihood")
    ax.plot(step_num, EDs, label="Energy distance")
    ax.plot(step_num, reg_Bs, label="Regularization cost: B matrices")
    ax.plot(step_num, reg_UVs, label="Regularization cost: UV")

    if title:
        ax.set_title(title)

    plt.xlabel("Update number")
    plt.ylabel("log likelihood")
    plt.legend()


    return fig, ax

# with open(path + "ModelDistMatch2dUniform_vocab_size7000_dim50_lambdaB0.001UV0.001_logitFalse_coIdFalse_samplingproportional_pos0.5_B500_UniRange5_EnergyWeight1_Slices1_update_5350.model.pkl", "r") as f:
#     mt2d = pkl.load(f)

fig, ax = print_likelihoods(pure_lls, EDs, reg_Bs, reg_UVs, 'Train data')
fig.show()

fig, ax = print_likelihoods(pure_lls_test, EDs_test, reg_Bs_test, reg_UVs_test, 'Test data')
fig.show()

