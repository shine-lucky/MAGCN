import torch as t
from torch import nn

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, prediction):
        loss_fn = t.nn.BCELoss()
        return loss_fn(prediction, target)
