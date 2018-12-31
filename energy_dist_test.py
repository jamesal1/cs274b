import torch
from torch.autograd import Variable

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
        tmp = torch.rand((dim, 1))-.5
        tmp = tmp / torch.norm(tmp)
        projection = Variable(tmp, requires_grad=False)
        x_proj, x_arg = torch.sort((x @ projection).flatten())
        x_prime_proj, x_prime_arg = torch.sort((x_prime @ projection).flatten())
        y_proj, y_arg = torch.sort((y @ projection).flatten())
        # print(x_proj)
        # print(y_proj)
        # dist += torch.sum((y_proj - x_prime_proj) ** 2) + torch.sum((y_proj - x_proj) ** 2) - torch.sum((x_proj - x_prime_proj) ** 2)
        dist += 2 * torch.sum((y_proj - x_proj) ** 2) - torch.sum((x_proj - x_prime_proj) ** 2)

    return dist / 5





a = Variable(torch.rand(40,2)-.5,requires_grad=True)
optimizer = torch.optim.Adam([a], lr = .0001)
print(a.t())
for i in range(1000):
    ed = energyDistance(a)
    ed.backward()
    optimizer.step()
    optimizer.zero_grad()
print(a.t())
from matplotlib import pyplot as plt

at=a.t().detach().numpy()
plt.scatter(at[0],at[1])
plt.savefig("energy_test.png")
plt.close()
