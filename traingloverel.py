import pickle as pkl
from model import Model
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import settings
vocab_size = settings.vocab_size

with open("data/relations_mat.pkl", "rb") as f:
    relmat = pkl.load(f)

with open("data/co.pkl", "rb") as f:
    comat = pkl.load(f)

biggest = 0
for a, b, c in comat:
    l = math.log(1 + c)
    biggest = max(l, biggest)

comat = [(a, b, math.log(1 + c)/biggest) for a, b, c in comat]

with open("data/vocab.txt", "r") as f:
    vocab = [(v.split(" ")[0], i) for i, v in enumerate(f.readlines())][:vocab_size]

gloveList=[]
with open("data/gloveEmbs200.txt","r") as f:
    for row in f.readlines():
        arr=row.strip().split()
        gloveList +=[[float(x) for x in arr[1:]]]
        if len(gloveList)==settings.vocab_size:
            break
gloveArr = np.array(gloveList)
gloveU = gloveArr[:, :50]
gloveV = gloveArr[:, 51:-1]

relmat["co"] = comat
m = Model(vocab, relmat, embedding_dimension=50, lambdaB=1e-3, lambdaUV=1e-3, logistic=False)
if __name__=="__main__":
    m.U = torch.FloatTensor(gloveU).cuda()
    m.V = torch.FloatTensor(gloveV).cuda()
    m.updateB()
    print(m.estimateLL())
    m.save("data/gloveRel200.pkl")
else:
    m.load("data/gloveRel200.pkl")