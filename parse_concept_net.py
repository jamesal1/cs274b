import json

from collections import defaultdict

R = defaultdict(lambda: [])

num_words = 10000

vocab = set()

with open('./vocab.txt', 'r') as f:
    for i, line in enumerate(f):
        if i == num_words:
            break
        else:
            vocab.add(line.split()[1])



with open('./conceptnet-assertions-5.6.0.csv', 'r') as f:

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
        

        source_w = source.split('/')[2]
        target_w = target.split('/')[2]

        if rel in ['/r/RelatedTo', '/r/FormOf', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor', '/r/CapableOf', '/r/Causes', '/r/HasProperty', '/r/Desires', '/r/Synonym', '/r/Antonym', '/r/DistinctFrom', '/r/Entails', '/r/SimilarTo', '/r/InstanceOf']:
            for start in ['the ', 'an ', 'a ']:
                if source_w.startswith(start):
                    source_w = source_w[len(start):]
                if target_w.startswith(start):
                    target_w = target_w[len(start):]
                    
            
            if source_w in vocab and target_w in vocab:
                R[rel].append((source, target, 1))
        
        
