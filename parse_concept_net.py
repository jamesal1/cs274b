import json
import os
from collections import defaultdict
import pickle as pkl
import settings

def getRelationMatrices(vocab_size,vocab_file = "./data/vocab.txt"):
    file_loc = "./data/relations_dict_vocab_size_{}.pkl".format(vocab_size)
    if not os.path.exists(file_loc):
        
        R = defaultdict(lambda: set())
        vocab = set()
        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                if i == vocab_size:
                    break
                else:
                    vocab.add(line.split()[0])
        
        symmetric_rels = {'/r/RelatedTo', '/r/Synonym', '/r/Antonym', '/r/DistinctFrom',
                          '/r/SimilarTo', '/r/LocatedNear', '/r/EtymologicallyRelatedTo'}
        
        with open('./data/conceptnet-assertions-5.6.0.csv', 'r') as f:
        
            flag = False
            for i, line in enumerate(f):
        
                if not i % 1000000:
                    print("Processed {} lines".format(i))
                
                elements = line.split(sep='\t')
                if len(elements) != 5:
                    continue
        
                
                rel, source, target, js  = elements[1:5]
        
                try:
                    weight = json.loads(js)['weight']
                    if weight != 1 and not flag:
                        flag = True
                        print("Non unit weight, {}!".format(weight))
                except:
                    continue
                
        
                source_w = source.split('/')[3]
                target_w = target.split('/')[3]
        
                if rel in ['/r/RelatedTo', '/r/FormOf', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/Causes', '/r/HasProperty', '/r/Desires', '/r/Synonym', '/r/Antonym', '/r/DistinctFrom', '/r/Entails', '/r/SimilarTo', '/r/InstanceOf']:
                    for start in ['the ', 'an ', 'a ']:
                        if source_w.startswith(start):
                            source_w = source_w[len(start):]
                        if target_w.startswith(start):
                            target_w = target_w[len(start):]
                            
                    
                    if source_w in vocab and target_w in vocab:
                        R[rel].add((source_w, target_w, 1))
        
                        if rel in symmetric_rels:
                            R[rel].add((target_w, source_w, 1))

        
        for k in R:
            R[k] = list(R[k])
        
        with open(file_loc, 'wb') as f:
            pkl.dump(dict(R), f)

    with open(file_loc,"rb") as f:
        return pkl.load(f)


def getRelationNumericTriples(vocab_size, vocab_file = './data/vocab.txt'):
    relation_dict = getRelationMatrices(vocab_size)

    with open(vocab_file, "r") as f:
        vocab_dict = dict([(v.split(" ")[0],i) for i, v in enumerate(f.readlines()) if i < settings.vocab_size])
    output_dict = {}
    for relation, inner_list in relation_dict.items():
        output_dict[relation] = [(vocab_dict[a], vocab_dict[b], c) for a, b, c in inner_list if a in vocab_dict and b in vocab_dict]
    return output_dict