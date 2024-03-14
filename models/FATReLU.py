import math
import torch

class Binarize(torch.autograd.Function): # same as a spike function
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x, width)
        out = x.gt(0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, width = ctx.saved_tensors
        grad_input = grad_output.clone()

        sg = 1 / (1 + width * x * x)
            
        # print(sg.size(), grad_input.size())
        return grad_input * sg, None

def binarize(x, width=torch.tensor(10.0)):
    return Binarize.apply(x, width)