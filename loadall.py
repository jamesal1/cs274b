import pickle as pkl
from model import Model, ModelTorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import settings
vocab_size = settings.vocab_size

with open("data/relations_mat.pkl", "rb") as f:
    relmat = pkl.load(f)

print(relmat["/r/Antonym"][:100])
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

# Test only a subset of rels

relmat_old = relmat
relmat = {}
# relmat['/r/IsA'] = relmat_old['/r/IsA']
relmat['/r/Antonym'] = relmat_old['/r/Antonym']
# relmat['/r/CapableOf'] = relmat_old['/r/CapableOf']
# relmat['/r/RelatedTo'] = relmat_old['/r/RelatedTo']
relmat['/r/Synonym'] = relmat_old['/r/Synonym']

# mt = ModelTorch(vocab, relmat, embedding_dimension=50, lambdaB=settings.reg_B, lambdaUV=settings.reg_B,
#                 logistic=settings.logistic, co_is_identity=settings.co_is_identity,
#                 sampling_scheme=settings.sampling_scheme,
#                 proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)

relmat["co"] = comat
m = Model(vocab, relmat, embedding_dimension=5, lambdaB=settings.reg_B, lambdaUV=settings.reg_UV,
          logistic=settings.logistic, co_is_identity=settings.co_is_identity,
          sampling_scheme=settings.sampling_scheme, proportion_positive=settings.proportion_positive)

# embs usually 5, for simplicity

# optimizer = torch.optim.Adam(mt.parameters())
#
# lls = []
# accs = []
#
# print_every = 50
#
# for i in range(10000):
#
#     if i % print_every == 0:
#         print("#######################")
#         print("Update {}".format(i))
#         print("#######################")
#
#     ll, acc = mt.estimateLL(verbose= (i % print_every == 0))
#     lls.append(ll.data)
#     accs.append(acc)
#
#     nll = -ll
#     nll.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#
#     if i % print_every == 0:
#         print("#######################")
#         print("Update {}".format(i))
#         print("#######################")
#

#for proportion_positive in [0.01, 0.1, 0.2, 0.3, 0.5]:
# for proportion_positive in [0.3, 0.5]:
#     for reg in [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
#
#         print("***************************************")
#         print("#######################################")
#         print("Results for proportion positive {} and reg {}".format(proportion_positive, reg))
#         print("#######################################")
#         print("***************************************")
#         mt = ModelTorch(vocab, relmat, embedding_dimension=50, lambdaB=reg, lambdaUV=reg,
#                   logistic=settings.logistic, co_is_identity=settings.co_is_identity,
#                   sampling_scheme=settings.sampling_scheme, proportion_positive=proportion_positive, sample_size_B=settings.sample_size_B)
#
#         optimizer = torch.optim.Adam(mt.parameters())
#
#         lls = []
#         accs = []
#
#         print_every = 1000
#
#         for i in range(5000):
#
#             if i % print_every == 0:
#                 print("#######################")
#                 print("Update {}".format(i))
#                 print("#######################")
#
#             ll, acc = mt.estimateLL(verbose= (i % print_every == 0))
#             lls.append(ll.data)
#             accs.append(acc)
#
#             nll = -ll
#             nll.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             if i % print_every == 0:
#                 print("#######################")
#                 print("Update {}".format(i))
#                 print("#######################")
#
#         mt.save("./ReportPlots/7000wordsGradient5000sampleSize5000updates1e-3reg/proppositive{}_reg{}.pkl".format(proportion_positive, reg))
#         with open("./ReportPlots/7000wordsGradient5000sampleSize5000updates1e-3reg/llsaccs_proppositive{}_reg{}.pkl".format(proportion_positive, reg), "wb") as f:
#             pkl.dump([lls, accs], f)
#
#         plt.figure()
#         plt.plot(lls)
#         plt.xlabel("iteration")
#         plt.ylabel("log likelihood")
#         plt.savefig("./ReportPlots/7000wordsGradient5000sampleSize5000updates1e-3reg/lls_proppositive{}_reg{}.jpg".format(proportion_positive, reg))
#         plt.close()
#
#         plt.figure()
#         plt.plot(accs)
#         plt.xlabel("iteration")
#         plt.ylabel("correlation with correct answers")
#         plt.savefig("./ReportPlots/7000wordsGradient5000sampleSize5000updates1e-3reg/corrs_proppositive{}_reg{}.jpg".format(
#             proportion_positive, reg))
#         plt.close()
#
#         word = "god" # actually garfield is an interesing case. Acquired some information by just being an antonym to a cat
#         for i, r in enumerate(mt.relation_names):
#             print("Best words for word {} and relation {} are {}".format(word, mt.relation_names[i], mt.findBest(i, word, 5)))
#
#

# print("Checking sampling scheme quality")
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


# Uncomment to test the model from a file
# mt = ModelTorch(vocab, relmat, embedding_dimension=50, lambdaB=settings.reg_B, lambdaUV=settings.reg_UV,
#                    logistic=settings.logistic, co_is_identity=settings.co_is_identity,
#                    sampling_scheme=settings.sampling_scheme, proportion_positive=settings.proportion_positive, sample_size_B=settings.sample_size_B)
#mt.load("./ReportPlots/7000wordsGradient5000sampleSize5000updates1e-3reg/proppositive0.1_reg1e-05.pkl")

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