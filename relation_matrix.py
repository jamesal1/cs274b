import pickle as pkl


filename = ""
indexfile = ""
output = ""

with open(filename, "rb") as f:
    d = pkl.load(f)


with open(indexfile, "rb") as f:
    vocab_dict = [(v.split(" ")[0],i) for i, v in enumerate(f.readlines())]


rel_mats = []
output_dict = {}
for relation, inner_list in d.items():
    output_dict[relation] = [(vocab_dict[a], vocab_dict[b], c) for a, b, c in inner_list]

with open(output, "wb") as f:
    pkl.dump(output_dict,f)