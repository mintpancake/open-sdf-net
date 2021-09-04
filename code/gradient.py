import torch


class Gradient(object):

    @staticmethod
    def evaluate(net, xy):
        net.eval()
        net.zero_grad()
        input = xy.clone().detach().requires_grad_(True)
        output = net(input)
        output.backward()
        return torch.norm(input.grad)
