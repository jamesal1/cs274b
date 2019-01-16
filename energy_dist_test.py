import torch
from torch.autograd import Variable
import numpy as np

def energyDistance(activations, dist=None):
    size, dim = activations.shape
    size = size // 2

    # pdb.set_trace()
    x = activations[0:size]
    x_prime = activations[size:2 * size]
    y = torch.rand(*x.shape) * 5 - 5/2
    y = Variable(y, requires_grad=False)
    # print(y.sum())
    # exit()
    dist = 0
    for _ in range(5):
        tmp = torch.randn((dim, 1))
        tmp = tmp / torch.norm(tmp)
        projection = Variable(tmp, requires_grad=False)
        x_proj, x_arg = torch.sort(x @ projection, 0)
        x_prime_proj, x_prime_arg = torch.sort(x_prime @ projection, 0)
        y_proj, y_arg = torch.sort(y @ projection, 0)
        # print(x_proj)
        # print(y_proj)
        # dist += torch.sum((y_proj - x_prime_proj) ** 2) + torch.sum((y_proj - x_proj) ** 2) - torch.sum((x_proj - x_prime_proj) ** 2)
        dist += torch.sum((y_proj - x_proj) ** 2) + torch.sum((y_proj - x_prime_proj) ** 2) - torch.sum((x_proj - x_prime_proj) ** 2)

    return dist / 5



n_ex = 400
a = Variable(torch.rand(n_ex, 2)-.5,requires_grad=True)
optimizer = torch.optim.SGD([a], lr=0.001) # , lr = .0001
history = []

print(a.t())
for i in range(12000):
    order = np.random.choice(list(range(n_ex)), size=n_ex, replace=False)
    ed = energyDistance(a[order, :])
    ed.backward()
    history.append(ed.data.cpu().numpy())
    optimizer.step()
    optimizer.zero_grad()
print(a.t())
from matplotlib import pyplot as plt

at=a.t().detach().data.numpy()
plt.scatter(at[0], at[1])


plt.figure()
plt.plot(history)
plt.show()
#plt.savefig("energy_test.png")
#plt.close()
