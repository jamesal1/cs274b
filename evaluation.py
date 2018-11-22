import pathlib
from model import Model, ModelTorch
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

def getAUC(model, relation_triples_train, relation_triples_test):
    for i, r in enumerate(model.relation_names):
        if r == "co":
            continue
        scores_pos_train = model.getActivations(i, relation_triples_train[r])
        scores_pos_test = model.getActivations(i, relation_triples_test[r])

        scores_neg_train = model.getActivations(i, *getNegativeUVs(relation_triples_train[r]))
        scores_neg_test = model.getActivations(i, *getNegativeUVs(relation_triples_test[r]))

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



cooccurrence_triples     = getCooccurrenceMatrix(settings.vocab_size)

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

relation_triples_train["co"] = cooccurrence_triples
relation_triples_test["co"] = cooccurrence_triples

### load and create the model
mt = ModelTorch(getVocab(settings.vocab_size), relation_triples_train, embedding_dimension=settings.embedding_dimension, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
                logistic=settings.logistic, co_is_identity=settings.co_is_identity,
                sampling_scheme=settings.sampling_scheme,
                proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)


getAUC(mt,relation_triples_train,relation_triples_test)

### train the model
optimizer = torch.optim.Adam(mt.parameters())
lls = []
accs = []

print_every = 50

for i in range(5000):

    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")
        ### evaluate performance, save the report in a readable form
        getAUC(mt, relation_triples_train, relation_triples_test)

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

path = "evaluation/vocab_size{}_test{}_dim{}_lambdaB{}UV{}_logit{}_coId{}_sampling{}_pos{}_B{}_seed{}/"
path = path.format(settings.vocab_size, settings.test_frac, settings.embedding_dimension, settings.reg_B, settings.reg_UV,
                   settings.logistic, settings.co_is_identity, settings.sampling_scheme, settings.proportion_positive,
                   settings.sample_size_B, settings.seed)
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

mt.save(path+"model.pkl")

### Print likelihoods

plt.figure()
plt.plot(lls)
plt.xlabel("iteration")
plt.ylabel("log likelihood")

#plt.savefig(path+".png") #BAD
#plt.close(fig)

plt.figure()
plt.plot(accs)
plt.xlabel("iteration")
plt.ylabel("correlation with correct answers")


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
