import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('inf')
    return out


def plot_losses(trainer): 
    plt.figure(figsize=(20,5))
    
    # plotting train losses
    plt.subplot(1,2,1)
    plt.title('%s training losses' % str(trainer)[1:8])
    for i, losses in enumerate(trainer.train_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")
    
    # plotting testing losses
    plt.subplot(1,2,2)
    plt.title('%s testing losses' % str(trainer)[1:8])
    for i, losses in enumerate(trainer.test_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")

    plt.show()


def set_plot_white():
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams["figure.facecolor"] = 'white'
    plt.rcParams['axes.facecolor']= 'white'
    plt.rcParams['savefig.facecolor']= 'white'


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluch the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)

        return x


@torch.no_grad()
def predict_trace(model, loader, steps):
    true_trace = []
    predicted_trace = []
    model.eval()
    context = 0
    for it, (x, y) in enumerate(loader):
        # set context vector if beginning
        if it == 0:
            context = x[:, :, 0].flatten()
            true_trace = context
            predicted_trace = context
        logits, _ = model(x, y)
        # take logits of final step
        logits = logits[:, -1, :]
        # apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_trace = torch.cat((true_trace, y[:, -1]))
        predicted_trace = torch.cat((predicted_trace, ix.flatten()))
        if it > steps:
            return true_trace, predicted_trace


@torch.no_grad()
def predict_intervals(model, loader, steps):
    true_trace = []
    predicted_trace = []
    model.eval()
    context = 0
    for it, (x, y) in enumerate(loader):
        # set context vector if beginning
        if it == 0:
            context = x.flatten()
            true_trace = context
            predicted_trace = context
        logits, _ = model(x, y)
        # take logits of final step
        logits = logits[:, -1, :]
        # apply softmax
        probs = F.softmax(logits, dim=-1)
        # choose highest topk (1) sample
        _, ix = torch.topk(probs, k=1, dim=-1)
        # append true and predicted in lists
        true_trace = torch.cat((true_trace, y[:, -1]))
        predicted_trace = torch.cat((predicted_trace, ix.flatten()))
        if it > steps:
            return true_trace, predicted_trace



