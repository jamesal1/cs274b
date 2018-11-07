
from model import Model, ModelTorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import settings
import random
import pdb

from utils import getCooccurrenceMatrix, getVocab, getRelationNumericTriples


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
mt = ModelTorch(getVocab(settings.vocab_size), relation_triples_train, embedding_dimension=50, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
                logistic=settings.logistic, co_is_identity=settings.co_is_identity,
                sampling_scheme=settings.sampling_scheme,
                proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)


### train the model

optimizer = torch.optim.Adam(mt.parameters())

lls = []
accs = []

print_every = 50

for i in range(200):

    if i % print_every == 0:
        print("#######################")
        print("Update {}".format(i))
        print("#######################")

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

### evaluate performance, save the report in a readable form

### Print likelihoods

plt.figure()
plt.plot(lls)
plt.xlabel("iteration")
plt.ylabel("log likelihood")

plt.figure()
plt.plot(accs)
plt.xlabel("iteration")
plt.ylabel("correlation with correct answers")


###
us = []
vs = []
train_acts = []

for i, r in enumerate(mt.relation_names):
    print(r)
    if r == "co":
        continue
    us, vs, _ = zip(*relation_triples_train[r])
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()

    train_acts.extend(list(mt.forward(us, vs, i).data.cpu().numpy()))

us = []
vs = []
test_acts = []

for i, r in enumerate(mt.relation_names):
    if r == "co":
        continue
    us, vs, _ = zip(*relation_triples_test[r])
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()

    test_acts.extend(list(mt.forward(us, vs, i).data.cpu().numpy()))
## !save settings alongside the model
bins = np.linspace(-2, 2, 300)

plt.figure()
plt.hist(train_acts, bins=bins, alpha=0.5, label="train", color="green")
plt.hist(test_acts, bins=bins, alpha=0.5, label="test", color="red")
# plt.hist(np.arange(100), alpha=0.5, label="test", color="red")
plt.legend(loc='upper right')
plt.show()

## Compare with negatives


us = []
vs = []
train_acts_neg = []

for i, r in enumerate(mt.relation_names):
    print(r)
    if r == "co":
        continue
    us, vs, _ = zip(*relation_triples_train[r])

    us_neg = np.random.choice(settings.vocab_size, len(us))
    vs_neg = np.random.choice(settings.vocab_size, len(us))
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()
    us_neg, vs_neg = torch.LongTensor(us_neg).cuda(), torch.LongTensor(vs_neg).cuda()

    train_acts_neg.extend(list(mt.forward(us_neg, vs, i).data.cpu().numpy()))
    train_acts_neg.extend(list(mt.forward(us, vs_neg, i).data.cpu().numpy()))



us = []
vs = []
test_acts_neg = []

for i, r in enumerate(mt.relation_names):
    print(r)
    if r == "co":
        continue
    us, vs, _ = zip(*relation_triples_test[r])

    us_neg = np.random.choice(settings.vocab_size, len(us))
    vs_neg = np.random.choice(settings.vocab_size, len(us))
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()
    us_neg, vs_neg = torch.LongTensor(us_neg).cuda(), torch.LongTensor(vs_neg).cuda()

    test_acts_neg.extend(list(mt.forward(us_neg, vs, i).data.cpu().numpy()))
    test_acts_neg.extend(list(mt.forward(us, vs_neg, i).data.cpu().numpy()))


plt.figure()
plt.hist(test_acts, bins=bins, alpha=0.5, label="test positive", color="green")
plt.hist(test_acts_neg, bins=bins, alpha=0.5, label="test negative", color="red")
# plt.hist(np.arange(100), alpha=0.5, label="test", color="red")
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.hist(train_acts, bins=bins, alpha=0.5, label="train positive", color="green")
plt.hist(train_acts_neg, bins=bins, alpha=0.5, label="train negative", color="red")
# plt.hist(np.arange(100), alpha=0.5, label="test", color="red")
plt.legend(loc='upper right')
plt.show()


## Compare with negatives for specific relations




for i, r in enumerate(mt.relation_names):

    if i > 3:
        break
    print(r)
    if r == "co":
        continue

    ###
    us = []
    vs = []
    train_acts = []

    us, vs, _ = zip(*relation_triples_train[r])
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()
    train_acts.extend(list(mt.forward(us, vs, i).data.cpu().numpy()))

    us = []
    vs = []
    test_acts = []
    us, vs, _ = zip(*relation_triples_test[r])
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()

    test_acts.extend(list(mt.forward(us, vs, i).data.cpu().numpy()))



    us = []
    vs = []
    train_acts_neg = []


    us, vs, _ = zip(*relation_triples_train[r])

    us_neg = np.random.choice(settings.vocab_size, len(us))
    vs_neg = np.random.choice(settings.vocab_size, len(us))
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()
    us_neg, vs_neg = torch.LongTensor(us_neg).cuda(), torch.LongTensor(vs_neg).cuda()

    train_acts_neg.extend(list(mt.forward(us_neg, vs, i).data.cpu().numpy()))
    train_acts_neg.extend(list(mt.forward(us, vs_neg, i).data.cpu().numpy()))



    us = []
    vs = []
    test_acts_neg = []

    us, vs, _ = zip(*relation_triples_test[r])

    us_neg = np.random.choice(settings.vocab_size, len(us))
    vs_neg = np.random.choice(settings.vocab_size, len(us))
    us, vs = torch.LongTensor(us).cuda(), torch.LongTensor(vs).cuda()
    us_neg, vs_neg = torch.LongTensor(us_neg).cuda(), torch.LongTensor(vs_neg).cuda()

    test_acts_neg.extend(list(mt.forward(us_neg, vs, i).data.cpu().numpy()))
    test_acts_neg.extend(list(mt.forward(us, vs_neg, i).data.cpu().numpy()))
    ## Try full negative?



    plt.figure()
    plt.title('Test anwers {}'.format(r))
    plt.hist(test_acts, bins=bins, alpha=0.5, label="test positive ", color="green")
    plt.hist(test_acts_neg, bins=bins, alpha=0.5, label="test negative", color="red")
    # plt.hist(np.arange(100), alpha=0.5, label="test", color="red")
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    plt.title('Train anwers {}'.format(r))
    plt.hist(train_acts, bins=bins, alpha=0.5, label="train positive", color="green")
    plt.hist(train_acts_neg, bins=bins, alpha=0.5, label="train negative", color="red")
    # plt.hist(np.arange(100), alpha=0.5, label="test", color="red")
    plt.legend(loc='upper right')
    plt.show()