import torch
import torch.nn.functional as F


def MSE(o, t):
    loss = F.mse_loss(o, t)
    correct = torch.round(t.data) == torch.round(o.data)
    return loss, correct

def MSE_SIG(o, t):
    o = torch.sigmoid(o)
    loss = F.mse_loss(o, t)
    correct = torch.round(t.data) == torch.round(o.data)
    return loss, correct


def CE(o, t):
    loss = F.cross_entropy(o.view(-1, len(o)), t.view(-1))
    correct = t.data == torch.argmax(o.data)
    return loss, correct


def NLL(o, t):
    loss = F.nll_loss(o, t)
    return loss


def BCE_L(o, t):

    t = t.reshape(o.shape)
    loss = F.binary_cross_entropy_with_logits(o, t)
    correct = torch.round(t.data) == torch.round(torch.sigmoid(o.data))

    return loss, correct


def BCE(o, t):
    t = t.reshape(o.shape)

    loss = F.binary_cross_entropy(o, t)
    correct = torch.round(t.data) == torch.round(o.data)
    return loss, correct


def MEAN_ABS(o, t):
    loss = torch.abs(t - o)
    correct = torch.round(t.data) == torch.round(o.data)
    return loss, correct
