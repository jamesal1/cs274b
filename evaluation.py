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
    with open(path + "AUC_update_{}.csv".format(iteration),"w") as f:
        f.write("relation,train_AUC,test_AUC\n")

        for i, r in enumerate(model.relation_names):
            if r == "co":
                continue
            scores_pos_train = model.getScores(i, relation_triples_train[r])
            scores_pos_test = model.getScores(i, relation_triples_test[r])

            scores_neg_train = model.getScores(i, *getNegativeUVs(relation_triples_train[r]))
            scores_neg_test = model.getScores(i, *getNegativeUVs(relation_triples_test[r]))

            scores_train = np.hstack([scores_pos_train, scores_neg_train])
            scores_test = np.hstack([scores_pos_test, scores_neg_test])

            true_ans_train = np.hstack([np.ones_like(scores_pos_train),
                                        np.zeros_like(scores_neg_train)])

            true_ans_test = np.hstack([np.ones_like(scores_pos_test),
                                        np.zeros_like(scores_neg_test)])
            try:
                train_auc = roc_auc_score(true_ans_train, scores_train)
            except ValueError:
                print("Not enough train data")
                train_auc = None

            try:
                test_auc = roc_auc_score(true_ans_test, scores_test)
            except ValueError:
                print("Not enough test data")
                test_auc = None

            f.write("{},{},{}\n".format(r, train_auc, test_auc))
            print(r, train_auc, test_auc)
            #return train_auc, test_auc


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

### load and create the model
mt = ModelDistMatch1dUniform(getVocab(settings.vocab_size), relation_triples_train, embedding_dimension=settings.embedding_dimension, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
                logistic=settings.logistic, co_is_identity=settings.co_is_identity,
                sampling_scheme=settings.sampling_scheme,
                proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)

path = "evaluation/{}_test{}_seed{}/"
path = path.format(str(mt),settings.test_frac, settings.seed)
pathlib.Path(path).mkdir(parents=True, exist_ok=True)


### train the model
optimizer = torch.optim.Adam(mt.parameters())
lls = []
accs = []

print_every = 50


for i in range(0):#range(settings.epochs):

    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")
        ### evaluate performance, save the report in a readable form
        getAUC(mt, relation_triples_train, relation_triples_test, path, i)

    ll, acc = mt.estimateLL(verbose= (i % print_every == 0))
    lls.append(ll.data)
    accs.append(acc)
    nll = -ll
    nll.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")
mt.save(path+"model.pkl")

### Print likelihoods
fig = plt.figure()
plt.plot(lls)
plt.xlabel("iteration")
plt.ylabel("log likelihood")

plt.savefig(path+"ll.png")
plt.close(fig)

fig = plt.figure()
plt.plot(accs)
plt.xlabel("iteration")
plt.ylabel("correlation with correct answers")
plt.savefig(path+"corr.png")
plt.close(fig)

###

train_acts = []
test_acts = []
train_acts_neg = []
test_acts_neg = []

for i, r in enumerate(mt.relation_names):
    print(r)
    if r == "co":
        continue

    train_acts.extend(mt.getActivations(i, relation_triples_train[r]))
    test_acts.extend(mt.getActivations(i, relation_triples_test[r]))
    train_acts_neg.extend(mt.getActivations(i, *getNegativeUVs(relation_triples_train[r])))
    test_acts_neg.extend(mt.getActivations(i, *getNegativeUVs(relation_triples_test[r])))

## !save settings alongside the model
bins = np.linspace(-2, 2, 300)

compareHistograms([(train_acts,"train"),(test_acts,"test")],"traintest", path)

## Compare with negatives
compareHistograms([(train_acts,"train positive"),(train_acts_neg,"train negative")], "train", path)
compareHistograms([(test_acts,"test positive"),(test_acts_neg,"test negative")], "test", path)

## Compare with negatives for specific relations

for i, r in enumerate(mt.relation_names):

    print(r)
    if r == "co":
        continue


    train_acts = mt.getActivations(i, relation_triples_train[r])
    test_acts = mt.getActivations(i, relation_triples_test[r])

    train_acts_neg = mt.getActivations(i, *getNegativeUVs(relation_triples_train[r]))
    test_acts_neg = mt.getActivations(i, *getNegativeUVs(relation_triples_test[r]))
    ## Try full negative?

    compareHistograms([(train_acts, "train positive"), (train_acts_neg, "train negative")], 'Train_answers_{}'.format(r.split("/")[-1]), path)
    compareHistograms([(test_acts, "test positive"), (test_acts_neg, "test negative")], 'Test_answers_{}'.format(r.split("/")[-1]), path)


###

mt2d = ModelDistMatch2dUniform(getVocab(settings.vocab_size), relation_triples_train, embedding_dimension=settings.embedding_dimension, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
                logistic=settings.logistic, co_is_identity=settings.co_is_identity,
                sampling_scheme=settings.sampling_scheme,
                proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)

mt2d.forward([1, 2, 3], [4, 5, 6], 1)

def plot_2d_acts(ax, pos_acts, neg_acts):
    all_acts = np.vstack([pos_acts, neg_acts])
    colors = np.zeros(all_acts.shape[0])
    colors[0:pos_acts.shape[0]] = 1
    scatterplot = ax.scatter(all_acts[:, 0], all_acts[:, 1], c=colors)
    #pdb.set_trace()
    #scatterplot = ax.hist(all_acts[:, 0])
    return scatterplot

def create_acts(m, relation_triples_train, relation_triples_test):

    '''m - model instance, implementing get activations method'''

    for i, r in enumerate(m.relation_names):

        train_acts = []
        test_acts = []
        train_acts_neg = []
        test_acts_neg = []

        print(r)
        if r == "co":
           continue

        train_acts.extend(m.getActivations(i, relation_triples_train[r]))
        test_acts.extend(m.getActivations(i, relation_triples_test[r]))
        train_acts_neg.extend(m.getActivations(i, *getNegativeUVs(relation_triples_train[r])))
        test_acts_neg.extend(m.getActivations(i, *getNegativeUVs(relation_triples_test[r])))

        train_acts, test_acts, train_acts_neg, test_acts_neg = map(np.array, [train_acts, test_acts, train_acts_neg, test_acts_neg])


        fig, ax = plt.subplots()
        plot_2d_acts(ax, train_acts, train_acts_neg)
        fig.show()

       # return

    return train_acts, test_acts, train_acts_neg, test_acts_neg

#train_acts, test_acts, train_acts_neg, test_acts_neg = map(np.array, create_acts(mt2d, relation_triples_train, relation_triples_test))
lp, corr = mt2d.estimateLL()
#train_acts, test_acts, train_acts_neg, test_acts_neg = create_acts(mt2d, relation_triples_train, relation_triples_test)

### train the model
optimizer = torch.optim.Adam(mt2d.parameters())
lls = [lp]
accs = [corr]

for i in range(300):#range(settings.epochs):

    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")
        ### evaluate performance, save the report in a readable form
##        getAUC(mt, relation_triples_train, relation_triples_test, path, i)

    ll, acc = mt2d.estimateLL(verbose= (i % print_every == 0))
    lls.append(ll.data.cpu().numpy())
    accs.append(acc)
    nll = -ll
    nll.backward()

    optimizer.step()
    optimizer.zero_grad()
    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")


create_acts(mt2d, relation_triples_train, relation_triples_test)