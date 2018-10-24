import struct
import os
record = struct.Struct('<iid')
record2 = struct.Struct('=iid')
record3 = struct.Struct('@iid')
import pickle as pkl

def getCooccurrenceMatrix(vocab_size):
    file_loc="data/co_vocab_size_{}.pkl".format(vocab_size)
    if not os.path.exists(file_loc):
        pairs = []

        with open('data/cooccurrence.bin', 'rb') as f:
            while True:
                buf = f.read(record.size)
                if not buf:
                    break
                w1, w2, c = record.unpack(buf)
                if w1 <= vocab_size and w2 <= vocab_size:
                    pairs += [(w1-1, w2-1, c)] # was w1, w2 (without -1)
        with open(file_loc, "wb") as f:
            pkl.dump(pairs, f)
    with open(file_loc,"rb") as f:
        return pkl.load(f)
