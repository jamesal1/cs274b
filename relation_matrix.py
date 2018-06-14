import pickle as pkl
import settings

filename = "data/relations_dict.pkl"
indexfile = "data/vocab.txt"
output = "data/relations_mat.pkl"

with open(filename, "rb") as f:
    d = pkl.load(f)


with open(indexfile, "r") as f:
    vocab_dict = dict([(v.split(" ")[0],i) for i, v in enumerate(f.readlines()) if i < settings.vocab_size])


rel_mats = []
output_dict = {}
for relation, inner_list in d.items():
    output_dict[relation] = [(vocab_dict[a], vocab_dict[b], c) for a, b, c in inner_list if a in vocab_dict and b in vocab_dict]

with open(output, "wb") as f:
    pkl.dump(output_dict, f)