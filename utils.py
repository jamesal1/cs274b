from unbin import getCooccurrenceMatrix
from parse_concept_net import getRelationMatrices, getRelationNumericTriples

def getVocab(vocab_size):
    with open("data/vocab.txt", "r") as f:
        vocab = [(v.split(" ")[0], i) for i, v in enumerate(f.readlines())][:vocab_size]
    return vocab