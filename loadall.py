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

# biggest = 0
# for a, b, c in comat:
#     l = math.log(1 + c)
#     biggest = max(l, biggest)
#
# comat = [(a, b, math.log(1 + c)/biggest) for a, b, c in comat]

_, _, c = zip(*comat)
co_mean = np.mean(np.log1p(c))
comat = [(a, b, np.log1p(c)/(2 * co_mean)) for a, b, c in comat] # make mean covariance to be 0.5. Think about a more principled choice
# Gives some weird error with only co. I suspect that it checks co equality with one. Will double check now. Works better with many factors.

with open("data/vocab.txt", "r") as f:
    vocab = [(v.split(" ")[0], i) for i, v in enumerate(f.readlines())][:vocab_size]

## Test only a subset of rels

relmat_old = relmat
relmat = {}
relmat['/r/IsA'] = relmat_old['/r/IsA']
relmat['/r/Antonym'] = relmat_old['/r/Antonym']
relmat['/r/CapableOf'] = relmat_old['/r/CapableOf']
relmat['/r/RelatedTo'] = relmat_old['/r/RelatedTo']
relmat['/r/Synonym'] = relmat_old['/r/Synonym']


relmat["co"] = comat
m = Model(vocab, relmat, embedding_dimension=5, lambdaB=settings.reg_B, lambdaUV=settings.reg_UV,
          logistic=settings.logistic, co_is_identity=settings.co_is_identity,
          sampling_scheme=settings.sampling_scheme, proportion_positive=settings.proportion_positive)

print("Checking sampling scheme quality")

# TO DO
# estimating sampling quality
# cntB = np.zeros((m.vocab_size, m.vocab_size))
# num_sim = 100
# for i in range(num_sim):
#     if i % 10 == 0:
#         print("Iteration {}".format(i) )
#     uis, vis, rs, ws = zip(*m.get_samples_for_B(0))
#     for u, v in zip(uis, vis):
#         cntB[u, v] += 1
#
# cntB = cntB / num_sim
#
# cntW = np.zeros((m.vocab_size, m.vocab_size))
# wind = 0
# for i in range(num_sim):
#     inds, _, _ = zip(*m.get_samples_for_w(wind, True))
#     for j in inds:
#         cntW[wind, j] += 1



if __name__ == "__main__":
    start = time.time()
    estimates = [m.estimateLL()]
    print("Likelihood estimation time:", time.time() - start)
    m.save("data/model{}.pkl".format(0))
    for i in range(10):

        start = time.time()
        print("Updating B:")
        m.updateB()
        #print(time.time()-start)
        print("Max B values: ", [torch.max(torch.abs(b)) for b in m.B])
        estimates += [m.estimateLL()]
        start = time.time()

        if i % 2 == 1:
            print("Updating V:")
            m.updateV()
            print("Max V value: {}".format(torch.max(torch.abs(m.V))))
        else:
            print("Updating U:")
            m.updateU()
            print("Max U value: {}".format(torch.max(torch.abs(m.U))))

        #print(time.time()-start)
        estimates += [m.estimateLL()]
        # if i % 2 == 0:
        #     m.updateV()
        #     print(torch.max(torch.abs(m.V)))
        # else:
        #     m.updateU()
        #     print(torch.max(torch.abs(m.U)))
        # estimates += [m.estimateLL()]

        m.save("data/model{}.pkl".format(i + 1))


    lls, accs = zip(*estimates)

    plt.figure()
    plt.scatter(np.hstack((m.U[:, 1].cpu().numpy(), m.V[:, 1].cpu().numpy())),
                np.hstack((m.U[:, 0].cpu().numpy(), m.V[:, 0].cpu().numpy())), c=[1] * vocab_size + [2] * vocab_size)
    plt.xlabel("dim1")
    plt.ylabel("dim2")


    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(list(np.hstack((m.U[:, 1].cpu().numpy(), m.V[:, 1].cpu().numpy()))),
                   list(np.hstack((m.U[:, 0].cpu().numpy(), m.V[:, 0].cpu().numpy()))),
                   list(np.hstack((m.U[:, 2].cpu().numpy(), m.V[:, 2].cpu().numpy()))),
                   c=['b'] * vocab_size + ['purple'] * vocab_size, )
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    #plt.zlabel("dim3")
    plt.savefig("data/singularity/{}.png".format(i + 1))

    # for angle in range(0, 360, 1):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     #plt.pause(.001)
    #     fig.savefig("data/3dplot5dim/angle{}.png".format(angle))


    # print(lls)
        # plt.figure()
        # plt.plot(lls)
        # plt.show()
        # plt.savefig("data/ll{}.png".format(i+1))
        # plt.close()
        # m.save("data/model{}.pkl".format(i+1))
        #
    plt.figure()
    plt.plot(lls)
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.show()

    plt.figure()
    plt.plot(accs)
    plt.xlabel("iteration")
    plt.ylabel("correlation with correct answers")
    plt.show()

    # print(m.findBest(0,"good"))

    # nice one - model 5

    # for i in range(11):
    #     m.load("./data/model{}.pkl".format(i))
    #
    #     print("################### Iteration {} ##################".format(i))
    #     rnum = len(m.relation_names)
    #     for i, r in enumerate(m.relation_names):
    #         print("Rel: {}, B: {}, U: {}, V: {}".format(r, torch.norm(m.B[i]), torch.norm(m.U), torch.norm(m.V)))


m.load("./data/model{}.pkl".format(10))

#word = "garfield" # actually garfield is an interesing case. Acquired some information by just being an antonym to a cat
word = "good" # actually garfield is an interesing case. Acquired some information by just being an antonym to a cat
for i, r in enumerate(m.relation_names):
    print("Best words for word {} and relation {} are {}".format(word, m.relation_names[i], m.findBest(i, word, 5)))

    # The first part is somewhat weird. Maybe a bug?