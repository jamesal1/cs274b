import torch
from torch.autograd import Variable

def energyDistance(activations, dist=None):
    size, dim = activations.shape
    size = size // 2

    # pdb.set_trace()
    x = activations[0:size]
    x_prime = activations[size:2 * size]
    y = torch.rand(*x.shape) * 5 - 5 / 2
    y = Variable(y, requires_grad=False)
    dist = 0
    for _ in range(1):
        tmp = torch.rand((dim, 1))
        tmp = tmp / torch.norm(tmp)
        projection = Variable(tmp, requires_grad=False)

        x_proj, x_arg = torch.sort(x @ projection)
        x_prime_proj, x_prime_arg = torch.sort(x_prime @ projection)
        y_proj, y_arg = torch.sort(y @ projection)
        dist += 2 * torch.sum((x_proj - y_proj) ** 2) - torch.sum((x_proj - x_prime_proj) ** 2)

    return dist / 1

a = Variable(torch.rand(5,2),requires_grad=True)
ed = energyDistance(a)
print(ed)
ed.backward()
print(a.grad)