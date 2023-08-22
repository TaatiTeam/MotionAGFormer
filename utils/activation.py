import torch

class SymSum(torch.nn.Module):
    def __init__(self, dims=1):
        super(SymSum, self).__init__()
        self.dims = dims

    def forward(self, x):
        zero = torch.tensor(0.).to(x.device)
        shift = int(x.shape[self.dims] // 2)  # This is so that R+(a) + R-(b) and R-(a) + R+(b)
        relu = torch.maximum(zero, x)
        inv_relu = torch.minimum(zero, x)
        out = torch.roll(inv_relu, shift, dims=self.dims) + relu
        return out

    def __str__(self):
        return f"SymSum(dims={self.dims})"

    def __repr__(self):
        return f"SymSum(dims={self.dims})"