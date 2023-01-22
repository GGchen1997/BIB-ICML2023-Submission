import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from transformers import BertModel, DNATokenizer
import higher

import os
import re
import requests
import numpy as np
import random
#from my_model import *
import math
m = nn.Softmax(dim=1)

#char2idx = np.load("data/char2idx.npy", allow_pickle=True).item()

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def prepare_data(seq_batch, label_batch, tokenizer, device):
    #sequences = ["A E T C Z A O","S K T Z P"]
    #label = [0, 1]
    #device = torch.device('cuda:7')
    #seq_batch = [re.sub(r"[UZOB]", "X", seq) for seq in seq_batch]
    ids = tokenizer.batch_encode_plus(seq_batch, add_special_tokens=True, padding=True)
    input_ids = F.one_hot(torch.tensor(ids['input_ids']), 69).float().to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    label_batch = torch.LongTensor(label_batch).to(device)
    return input_ids, attention_mask, label_batch

def adjust_learning_rate(optimizer, lr0, epoch, T):
    lr = lr0 * (1+np.cos((np.pi*epoch*1.0)/(T*1.0)))/2.0
    print("epoch {} lr {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def gfp_char2idx(candidate):
    tmp = "".join(candidate.split())
    #char2idx = np.load("data/char2idx.npy", allow_pickle=True).item()
    my_seq = np.ones((1, 237)).astype(int)
    for j in range(237):
        my_seq[0, j] = char2idx[tmp[j]]
    return my_seq

def normalize(array):
    m = torch.mean(array)
    std = torch.std(array)
    norm_array = (array - m.data)/(std.data+1e-9)
    return norm_array, m, std

def denormalize(norm_array, m, std):
    array = norm_array*std + m
    return array

def m_func(distilled):
    #if True:
    #    return F.softmax(logits, dim=1)
    #return F.gumbel_softmax(logits, tau=1, hard=True)
    distilled_prob = F.softmax(distilled, dim=1)
    L = distilled_prob.shape[0]
    soft_label = torch.zeros(L-2, 64).to(distilled.get_device())
    for l in range(0, L-2):
        kmer_prob = distilled_prob[l:l+3]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    index = i * 16 + j * 4 + k 
                    prob = kmer_prob[0][i] * kmer_prob[1][j] * kmer_prob[2][k]
                    soft_label[l][index] = prob
    max_index = torch.argmax(soft_label, dim=1)
    hard_label = torch.eye(soft_label.shape[1])[max_index].to(distilled.get_device())
    label = hard_label - soft_label.data + soft_label
    return label

def m_func_backward(distilled):
    distilled_prob = F.softmax(distilled, dim=1)
    L = distilled_prob.shape[0]
    soft_label = torch.zeros(L-2, 64).to(distilled.get_device())
    for l in range(0, L-2):
        kmer_prob = distilled_prob[l:l+3]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    index = i * 16 + j * 4 + k
                    prob = kmer_prob[0][i] * kmer_prob[1][j] * kmer_prob[2][k]
                    soft_label[l][index] = prob
    return soft_label

def distance_m_func(logits):
    #if True:
    #    return F.softmax(logits, dim=1)
    #return F.gumbel_softmax(logits, tau=1, hard=True)
    soft_label = F.softmax(logits, dim=1)
    max_index = torch.argmax(logits, dim=1)
    hard_label = torch.eye(logits.shape[1])[max_index].to(logits.get_device())
    label = hard_label - soft_label.data + soft_label
    return label

def compute_distance(distilled, distilled0, tau=1e-1):
    distance = torch.sum(torch.pow(distance_m_func(distilled) - distilled0, 2))/2.0
    return distance

def compute_distance_grad(distilled, distilled0, t, T, tau=1.0):
    #temperature = (1 + np.cos(t/T*np.pi))/2*tau
    #temperature = max(temperature, 0.01)
    temperature = tau
    distilled = F.softmax(distilled/temperature, dim=1)
    distance = torch.sum(torch.pow(distilled - distilled0, 2))/2.0
    #print("inner", t, temperature, distance)
    return distance

def compute_pcc(valid_preds, valid_labels):
    #vx = valid_preds.shape[0]  - torch.argsort(valid_preds)
    #vy = valid_labels.shape[0]  - torch.argsort(valid_labels)
    #vx = vx.float()
    #vy = vy.float()
    vx = valid_preds - torch.mean(valid_preds)
    vy = valid_labels - torch.mean(valid_labels)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-12) * torch.sqrt(torch.sum(vy ** 2) + 1e-12))
    return pcc

def set_model(model, flag=False):
    for param in model.parameters():
        param.requires_grad = flag

def seq2kmer(seq, k=3):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def smooth_model(f, x):
    x.requires_grad = True
    f_x = f(x)
    dx = torch.autograd.grad(f_x, [x], retain_graph=True, create_graph=True)[0]
    loss = torch.sum(torch.pow(dx, 2)) + torch.pow(f_x - f_x.data, 2)
    loss.backward()
    dtheta = f.main.weight.grad.data
    theta = f.main.weight.data
    lr_max = 5e-4*torch.sqrt(torch.norm(theta))/torch.sqrt(torch.norm(dtheta))
    if lr_max > 0.1:
        lr = 0.1
    else:
        lr = lr_max
    f.main.weight.data = f.main.weight.data - lr * dtheta
    return f

def compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim):
    z_emb = (z @ rf_w) / sigma + rf_b
    z_emb = torch.cos(z_emb) * (2. / rf_dim) ** 0.5
    return z_emb

def compute_mmd_mean_rf(z, sigma=14, rf_dim=500):
    rf_w = torch.randn((z.shape[1], rf_dim), device=z.device)
    rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z.device)
    z_rf = compute_gaussian_rf(z, rf_w, rf_b, sigma, rf_dim)
    mu_rf = z_rf.mean(0, keepdim=False)
    return mu_rf

def wae_mmd_gaussianprior(z):
    z_prior = torch.randn_like(z)
    #mmd_rf(z, z_prior)
    mu1 = compute_mmd_mean_rf(z)
    mu2 = compute_mmd_mean_rf(z_prior)
    loss = ((mu1 - mu2) ** 2).sum()
    return loss

if __name__ == "__main__":
    sequences = ["AAA ACT GCA GGG TTT"]#,"S K T Z P"]
    label = [0, 1]
    device = torch.device('cuda:7')
    print("before", sequences)
    tokenizer = DNATokenizer.from_pretrained("DNABertModel", do_lower_case=False)
    input_ids, attention_mask, label_batch = prepare_data(sequences, label, tokenizer, device)
    tmp1 = torch.argmax(input_ids, dim=2)
    print(tmp1)
    #exit(0)
    decoded = tokenizer.decode(tmp1[0])
    #decoded = "".join(decoded.split(" ")[1:-1])
    decoded = "".join(decoded.split(" "))
    print("after", decoded)

