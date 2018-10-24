import struct
record = struct.Struct('<iid')
record2 = struct.Struct('=iid')
record3 = struct.Struct('@iid')
import pickle as pkl
import settings
pairs = []

with open('data/cooccurrence.bin', 'rb') as f:
    while True:
        buf = f.read(record.size)
        if not buf:
            break
        w1, w2, c = record.unpack(buf)
        if w1 <= settings.vocab_size and w2 <= settings.vocab_size:
            pairs += [(w1-1, w2-1, c)] # was w1, w2 (without -1)

with open("data/co.pkl", "wb") as f:
    pkl.dump(pairs, f)

print("Unbinning is done")