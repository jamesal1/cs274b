import pickle as pkl
from model import Model

with open("data/relations_mat.pkl", "rb") as f:
    relmat = pkl.load(f)

with open("data/co.pkl","rb") as f:
    comat = pkl.load(f)

biggest = 0
for a, b, c in comat:
    l = math.log(1 + c)
    biggest = max(l,biggest)

comat = [(a,b,math.log(1 + c)/biggest) for a,b,c in comat]

with open(indexfile, "r") as f:
    vocab_dict = dict([(v.split(" ")[0],i) for i, v in enumerate(f.readlines())])

Model()
