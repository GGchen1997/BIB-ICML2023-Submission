import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from transformers import BertModel, DNATokenizer
from my_model import *
from transformers import logging
from torch.autograd import grad

import os
import re
import requests

import higher
import random
from utils import *
import argparse
import time

parser = argparse.ArgumentParser(description="sequence distillation")
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--Tmax', default=100, type=int)
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--init', default=3.7698, type=float)
parser.add_argument('--device', default=7, type=int)
parser.add_argument('--mode', default="BDI", type=str, choices=["BDI", "forward", "backward"])
parser.add_argument('--gamma_learn', default=1, type=int)
parser.add_argument('--eta_learn', default=0, type=int)
parser.add_argument('--task', default="TFBind8-Exact-v0", type=str, choices=["TFBind8-Exact-v0", "TFBind10-Exact-v0"])
parser.add_argument('--task_mode', default="distill", type=str, choices=["oracle", "distill"])
args = parser.parse_args()

#logging.set_verbosity_error()
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = torch.device('cuda:' + str(args.device))
#device = torch.device("cpu")
char2idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
idx2char = ['A', 'T', 'C', 'G']

def build_oracle(args):
    print("building", args.task)
    #load
    seqs = np.load("RRdata/" + args.task + "seqs.npy", allow_pickle=True)
    labels = np.load("RRdata/" + args.task + "labels.npy", allow_pickle=True)
    logits = torch.load("RRdata/" + args.task + "logits.pt")
    assert seqs.shape[0] == logits.shape[0], "loading error"
    #set offline
    labels = torch.from_numpy(labels.squeeze()).reshape(-1, 1).to(torch.float64)
    index = torch.argsort(labels, dim=0).squeeze().cpu().numpy()
    #compute offline
    #offline_index = index[0:int(index.shape[0]/2)]
    offline_index = index[0:5000]
    offline_seqs = seqs[offline_index]
    offline_labels, offline_m, offline_std = normalize(labels[offline_index])
    #build training data
    offline_logits = logits[offline_index]
    offline_logits_t = torch.transpose(offline_logits, 0, 1)
    offline_Kll = torch.matmul(offline_logits.cpu(), offline_logits_t.cpu())
    beta = 1e-6 * torch.trace(offline_Kll) / offline_Kll.shape[0]
    offline_coeffs = (offline_Kll + beta*torch.eye(offline_Kll.shape[0])).to(torch.float64)
    offline_coeffs = torch.solve(offline_labels.cpu(), offline_coeffs.cpu()).solution#.to(device)
    #save
    np.save("RRdata/" + args.task + "offline_seqs.npy", offline_seqs)
    torch.save(offline_logits.cpu(), "RRdata/" + args.task + "offline_logits.pt")
    torch.save(offline_labels.cpu(), "RRdata/" + args.task + "offline_labels.pt")
    torch.save(offline_coeffs.cpu(), "RRdata/" + args.task + "offline_coeffs.pt")
    torch.save(offline_m.cpu(), "RRdata/" + args.task + "offline_m.pt")
    torch.save(offline_std.cpu(), "RRdata/" + args.task + "offline_std.pt")



def distill(args):
    #define model and optimization
    tokenizer = DNATokenizer.from_pretrained("DNABertModel", do_lower_case=False)
    model = MyBert().to(device)
    model.eval()
    #load
    offline_seqs = np.load("RRdata/" + args.task + "offline_seqs.npy", allow_pickle=True)
    offline_logits = torch.load("RRdata/" + args.task + "offline_logits.pt").to(device)
    logits = torch.load("RRdata/" + args.task + "logits.pt").to(device)
    offline_labels = torch.load("RRdata/" + args.task + "offline_labels.pt").to(device)
    offline_coeffs = torch.load("RRdata/" + args.task + "offline_coeffs.pt").to(device)
    offline_m = torch.load("RRdata/" + args.task + "offline_m.pt").to(device)
    offline_std = torch.load("RRdata/" + args.task + "offline_std.pt").to(device)
    #load model
    labels = np.load("RRdata/" + args.task + "labels.npy", allow_pickle=True)
    labels = torch.from_numpy(labels.squeeze()).reshape(-1, 1).to(torch.float64).to(device)
    min_labels, max_labels = torch.min(labels), torch.max(labels)
    print("offline max", torch.max(offline_labels))
    print("best target", (max_labels - offline_m)/offline_std)
    seq2label = np.load("RRdata/" + args.task + "seq2label.npy", allow_pickle=True).item()
    #proxy
    proxy = OneMLP(768, 384, 1).to(device)
    proxys = [proxy for i in range(5)]
    seeds = [1, 10, 100, 1000, 10000]
    for j in range(5):
        proxys[j].load_state_dict(torch.load("model/" + args.task + "_part_oracle_" + str(seeds[j]) +  ".pt"))
    #choose candidate
    candidates = offline_seqs[-128:]
    init_scores = offline_labels[-128:]*offline_std + offline_m
    init_scores = (init_scores.data - min_labels)/(max_labels - min_labels)
    init_scores = init_scores.squeeze().cpu().numpy()
    init_logits = offline_logits[-128:]
    scores = torch.zeros(128).to(device)
    distances = torch.zeros(128)
    if args.task == "TFBind8-Exact-v0":
        distill_len = 8
    elif args.task == "TFBind10-Exact-v0":
        distill_len = 10
    distilled_normal_sample = torch.normal(mean=0.0, std=0.01, size=(distill_len, 4)).to(device)
    print("sample", distilled_normal_sample)
    perfs = torch.zeros(128, args.Tmax + 1).to(device)
    perfs[:, 0] = torch.from_numpy(init_scores).to(device)
    distances = torch.zeros(128, args.Tmax + 1).to(device)
    etas = 0.5*torch.ones(128, args.Tmax + 1).to(device)
    for i in range(128):
        #prepare candidate
        candidate = candidates[i]
        distill_len = len(candidate)
        #distilled = (-args.init) * torch.ones(distill_len, 4).to(device)
        distilled = copy.deepcopy(distilled_normal_sample)
        distilled0 = torch.zeros(distill_len, 4).to(device)
        for k in range(distill_len):
            max_index = torch.argmax(distilled[k])
            max_value = copy.deepcopy(distilled[k, max_index])
            distilled[k, max_index] = copy.deepcopy(distilled[k, char2idx[candidate[k]]])
            distilled[k, char2idx[candidate[k]]] = max_value
            distilled0[k, char2idx[candidate[k]]] = 1.0
        distilled.requires_grad = True
        #handle kmer
        kmer_candidate = seq2kmer(candidate)
        #candidate_input_id, _, _ = prepare_data([kmer_candidate], [1], tokenizer, device)
        distill_seq = {}
        distill_seq['token_type_ids'] = torch.zeros(1, distill_len).to(device)
        distill_seq['attention_mask'] = torch.ones(1, distill_len).to(device)
        #optimizer
        distilled_opt = optim.Adam([distilled], lr=args.lr)#, weight_decay=args.wd)
        if args.gamma_learn:
            eta = torch.Tensor([0.5]).to(device)
        else:
            eta = torch.Tensor([0.5]).to(device)
        eta_m = torch.Tensor([0.0]).to(device)
        eta_v = torch.Tensor([0.0]).to(device)
        eta_beta = torch.Tensor([0.9, 0.999]).to(device)
        #predined target
        distill_label = torch.Tensor([10.0]).to(device).view(1, 1).to(torch.float64)
        for t in range(args.Tmax):
            #get logit
            distill_seq['input_ids'] = torch.zeros(1, distill_len, 69).to(device)
            distill_seq['input_ids'][0, 1:-1, 5:69] = torch.clone(m_func(distilled))
            distill_seq['input_ids'][0, 0, 2] = 1
            distill_seq['input_ids'][0, -1, 3] = 1
            distill_logit = model(input_ids=distill_seq["input_ids"], \
                    attention_mask=distill_seq["attention_mask"]).to(torch.float64)
            if (args.gamma_learn or args.eta_learn) and (args.mode == 'BDI'):
                oracle_score = 0
                for j in range(5):
                    oracle_score = oracle_score + proxys[j](distill_logit.float())*offline_std.data
                oracle_score = oracle_score/5.0 + offline_m.data
                f_out_grad = grad(oracle_score, [distilled], retain_graph=True)[0].reshape(-1, 1).squeeze()
            #forward
            if args.mode in ['BDI', 'forward']:
                Khl = torch.matmul(distill_logit, torch.transpose(offline_logits, 0, 1))
                predictions_l2h = torch.matmul(Khl, offline_coeffs)
                loss_l2h = torch.mean(torch.pow(predictions_l2h - distill_label, 2))
                if (args.gamma_learn or args.eta_learn) and (args.mode == 'BDI'):
                    loss_l2h_grad = torch.autograd.grad(loss_l2h, [distilled], retain_graph=True)[0].reshape(-1, 1).squeeze()
            #backward
            if args.mode in ['BDI', 'backward']:
                Klh = torch.matmul(offline_logits, torch.transpose(distill_logit, 0, 1))
                Khh = torch.matmul(distill_logit, torch.transpose(distill_logit, 0, 1))
                beta = 1e-6*torch.trace(Khh) / Khh.shape[0]
                offline_coeffs_Klh = (Khh + beta*torch.eye(Khh.shape[0]).to(device)).to(torch.float64)
                offline_coeffs_Klh = torch.solve(distill_label, offline_coeffs_Klh).solution
                predictions_h2l = torch.matmul(Klh, offline_coeffs_Klh)
                loss_h2l = torch.mean(torch.pow(predictions_h2l - offline_labels, 2))
                if (args.gamma_learn or args.eta_learn) and (args.mode == 'BDI'):
                    loss_h2l_grad = torch.autograd.grad(loss_h2l, [distilled], retain_graph=True)[0].reshape(-1, 1).squeeze()
            #overall
            if args.mode in ['forward']:
                loss = loss_l2h
            elif args.mode in ['backward']:
                loss = loss_h2l
            elif args.mode in ['BDI']:
                if args.gamma_learn:
                    #compute in grad
                    loss_in_grad = loss_l2h_grad - loss_h2l_grad
                    eta_grad = torch.dot(f_out_grad, args.lr* loss_in_grad).data
                    #adam
                    eta_m = eta_beta[0]*eta_m + (1-eta_beta[0])*eta_grad
                    eta_v = eta_beta[1]*eta_v + (1-eta_beta[1])*torch.pow(eta_grad, 2)
                    eta_m_ = eta_m/(1-torch.pow(eta_beta[0], t+1))
                    eta_v_ = eta_v/(1-torch.pow(eta_beta[1], t+1))
                    eta = torch.clamp(eta - args.lr * eta_m_/(torch.sqrt(eta_v_)+1e-8), 0.0, 1.0)
                    etas[i, t] = eta
                if args.eta_learn and (t >= 1):
                    lr_data = eta * loss_l2h_grad + (1 - eta) * loss_h2l_grad
                    status = distilled_opt.state_dict()
                    param_groups = status['param_groups'][0]
                    current_lr = status['param_groups'][0]['lr']
                    key = param_groups['params'][0]
                    statistics = status['state']
                    content = statistics[key]
                    #compute adam statistics
                    lr_data = lr_data.reshape(content['exp_avg'].shape)
                    lr_m = eta_beta[0]*content['exp_avg'] + (1-eta_beta[0]) * lr_data
                    lr_v = eta_beta[1]*content['exp_avg_sq'] + (1-eta_beta[1])*torch.pow(lr_data, 2) 
                    lr_m_ = lr_m/(1-torch.pow(eta_beta[0], t+1))
                    lr_v_ = lr_v/(1-torch.pow(eta_beta[1], t+1))
                    grad_data = lr_m_/(torch.sqrt(lr_v_) + 1e-8)
                    lr_update = torch.dot(f_out_grad, grad_data.view(-1, 1).squeeze()).data
                    #current_lr = torch.clamp(current_lr - 0.01*lr_update, 1e-4, 0.5)
                    current_lr = torch.clamp(0.1 - 0.01*lr_update, 1e-4, 0.5)
                    status['param_groups'][0]['lr'] = current_lr
                    print("current_lr", current_lr, "eta", eta)
                    distilled_opt.load_state_dict(status)
                loss = eta * loss_l2h + (1 - eta) * loss_h2l
            distilled_opt.zero_grad()
            loss.backward()
            distilled_opt.step()
            #evaluation
            with torch.no_grad():
                max_seq = torch.argmax(distilled.data, dim=1).data.cpu().numpy()
                seq = ""
                for s_i in max_seq:
                    seq = seq + idx2char[s_i]
                scores[i] = seq2label[seq].squeeze().tolist()
                scores[i] = (scores[i] - min_labels)/(max_labels - min_labels)
                perfs[i, t] = scores[i]
                distance = 0
                for s_i in range(distill_len):
                    if seq[s_i] != candidate[s_i]:
                        distance = distance + 1
                distances[i, t] = distance
                print("distance {} score before {:.3} score after {:.3}".format(distance, init_scores[i],  scores[i]))
    print("mean_dist {:.1} max value {:.3} median value {:.3} mean value {:.3}".format(torch.mean(distances), torch.max(scores), torch.median(scores), torch.mean(scores)))
    results = np.load("npy/abalation.npy", allow_pickle=True).item()
    key = args.task + "_" + args.mode  + "_" + str(args.gamma_learn) + "_" + str(args.seed)
    results[key] = [perfs.data.cpu().numpy(), distances.data.cpu().numpy(), etas.data.cpu().numpy()]
    np.save("npy/abalation.npy", results)


if __name__ == "__main__":
    print(args)
    set_seed(args.seed)
    if args.task_mode in ['oracle']:
        build_oracle(args)
    elif args.task_mode in ['distill']:
        distill(args)
