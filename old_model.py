import numpy as np
from numpy import random
from scipy import stats
u = np.arange(3)
v = np.arange(3)
B = np.arange(49).reshape((7,7))
B[6,6] = 1
B2 = np.ones((7,7))
r = 1
r2 = 0

class Model:

    def __init__(self, vocabulary, relations, embedding_size=150):
        self.vocabulary = vocabulary
        self.nwords = len(vocabulary)
        self.relations = relations
        self.embedding_size = embedding_size
        self.u = random.normal(size=(len(vocabulary), embedding_size))
        self.v = random.normal(size=(len(vocabulary), embedding_size))
        self.u_var = random.normal(size=(len(vocabulary), embedding_size))
        self.v_var = random.normal(size=(len(vocabulary), embedding_size))
        self.B = random.normal(size=(len(relations), 2 * embedding_size + 1, 2 * embedding_size + 1))

    def updateB(self, learning_rate=1e-4, batch_size=100):

        uind, vind = random.randint(self.embedding_size, size=(2, batch_size))


        uv = np.hstack([self.u[uind], self.v[vind]])
        uv_var = np.hstack([self.u_var[uind], self.v_var[vind]])
        uv_outer = uv[:, :, np.newaxis] * uv[:, np.newaxis, :]
        uv_outer[:, range(2 * self.embedding_size), range(2 * self.embedding_size)] += uv_var
        uvr_outer = np.zeros((batch_size,) + self.B.shape[1:])
        uvr_outer[:, :-1, :-1] = uv_outer

        for i in range(self.B.shape[0]):
            r = self.relations[i, uind, vind]
            uvr_outer[:, -1, :-1] = uv * r[:, np.newaxis]
            uvr_outer[:, :-1, -1] = uv * r[:, np.newaxis]
            uvr_outer[:, -1, -1] = r ** 2

            dLdB = np.linalg.inv(self.B[i]) - uvr_outer.mean(axis=0)

            ## Updating B

            self.B[i] += learning_rate * dLdB # should B[-1,-1] be 1 or free?



model = Model([0]*1000, np.ones((20,1000,1000)))
for i in range(10):
    a = model.B.copy()
    model.updateB()
    print(np.sum(np.abs(model.B-a)))