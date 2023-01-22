import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from transformers import BertModel, BertTokenizer
#from transformers import AutoModel, AlbertTokenizer
from transformers import T5Tokenizer, T5EncoderModel
from my_model import *
from transformers import logging
from torch.autograd import grad

import os
import re
import requests
import json
import higher
import random
from utils import *
import argparse
import time

parser = argparse.ArgumentParser(description="sequence distillation")
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--Tmax', default=25, type=int)
parser.add_argument('--tau', default=0.1, type=float)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--init', default=3.7698, type=float)
parser.add_argument('--device', default=7, type=int)
parser.add_argument('--mode', default="BDI", type=str, choices=["BDI", "forward", "backward"])
parser.add_argument('--gamma_learn', default=1, type=int)
parser.add_argument('--eta_learn', default=0, type=int)
parser.add_argument('--task', default="UBE2I", type=str, choices=["avGFP", "AAV", "E4B"])
parser.add_argument('--task_mode', default="distill", type=str, choices=["oracle", "distill"])
args = parser.parse_args()

logging.set_verbosity_error()
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = torch.device('cuda:' + str(args.device))

def compute_logits(args):
    print("computing", args.task)
    #define model
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    model = model.eval()
    #for AMP task
    if args.task == "AMP":
        #load data
        seqs0, _ = load_data("all")
        logits = []
        for i in range(seqs0.shape[0]):
            seq = re.sub(r"[UZOB]", "X", seqs0[i])
            tokenized_seq = tokenizer(seq, return_tensors="pt").to(device)
            input_ids = tokenized_seq.input_ids
            attention_mask = tokenized_seq.attention_mask
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                emb = outputs.last_hidden_state
                logit = torch.mean(emb, dim=1)
            logits.append(logit)
            if i % 1000 == 0:
                print(i)
        #fit KRR
        logits = torch.cat(logits, 0).to(torch.float64)
        torch.save(logits.cpu(), "RRdata/" + args.task + "oracle_logits.pt")
        return
    #for non-AMP task
    logits = []
    with open('data/train_data/' + args.task + "/train.json", 'r') as train_file:
        train_data = json.load(train_file)
    with open('data/train_data/' + args.task + "/valid.json", 'r') as valid_file:
        valid_data = json.load(valid_file)
    data = train_data + valid_data
    L = len(data)
    for i in range(L):
        seq = data[i]['seq']
        seq = re.sub(r"[UZOB]", "X", seq)
        seq = " ".join(seq)
        tokenized_seq = tokenizer(seq, return_tensors="pt").to(device)
        input_ids = tokenized_seq.input_ids
        attention_mask = tokenized_seq.attention_mask
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = outputs.last_hidden_state
            logit = torch.mean(emb, dim=1)
        logits.append(logit.cpu())
        if i%1000 == 0:
            print(i)
    #normalize
    logits = torch.cat(logits, 0).to(torch.float64)
    torch.save(logits.cpu(), "RRdata/" + args.task + "oracle_logits.pt")

def build_oracle(args):
    print("building", args.task)
    #load
    seqs = np.load("RRdata/" + args.task + "seqs.npy", allow_pickle=True)
    m = torch.load("model/" + args.task + "_m.pt")
    std = torch.load("model/" + args.task + "_std.pt")
    min_y = torch.load("model/" + args.task + "_min_y.pt") 
    max_y = torch.load("model/" + args.task + "_max_y.pt")
    logits = torch.load("RRdata/" + args.task + "logits.pt")
    oracle_logits = torch.load("RRdata/" + args.task + "oracle_logits.pt")
    assert seqs.shape[0] == logits.shape[0], "loading error"
    #load oracle to compute labels
    model = SimpleMLP(1024, 512, 1, min_y, max_y).to(device)
    #model = OneMLP(1024, 512, 1).to(device)
    model.load_state_dict(torch.load("model/" + args.task + "_oracle.pt"))
    model.eval()
    with torch.no_grad():
        labels = model(oracle_logits.float().to(device)).double().cpu()
    labels_unnorm = denormalize(labels, m, std)
    #set offline
    index = torch.argsort(labels_unnorm, dim=0).squeeze().cpu().numpy()
    #compute offline
    offline_index = index[0:int(index.shape[0]/2)]
    offline_seqs = seqs[offline_index]
    offline_labels, offline_m, offline_std = normalize(labels_unnorm[offline_index])
    #build training data
    offline_logits = logits[offline_index]
    offline_logits_t = torch.transpose(offline_logits, 0, 1)
    offline_Kll = torch.matmul(offline_logits.cpu(), offline_logits_t.cpu())
    beta = 1e-6 * torch.trace(offline_Kll) / offline_Kll.shape[0]
    print(offline_Kll.shape)
    offline_coeffs = (offline_Kll + beta*torch.eye(offline_Kll.shape[0])).to(torch.float64)
    offline_coeffs = torch.solve(offline_labels.cpu(), offline_coeffs.cpu()).solution.to(device)
    #save
    np.save("RRdata/" + args.task + "offline_seqs.npy", offline_seqs)
    torch.save(offline_logits.cpu(), "RRdata/" + args.task + "offline_logits.pt")
    torch.save(offline_labels.cpu(), "RRdata/" + args.task + "offline_labels.pt")
    torch.save(offline_coeffs.cpu(), "RRdata/" + args.task + "offline_coeffs.pt")
    torch.save(offline_m.cpu(), "RRdata/" + args.task + "offline_m.pt")
    torch.save(offline_std.cpu(), "RRdata/" + args.task + "offline_std.pt")


def distill(args):
    #define model and optimization
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = MyBert()#.to(device)
    model = model.to(device)
    model.eval()
    #oracle
    tokenizer_oracle = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model_oracle = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    model_oracle = model_oracle.eval()
    #load
    offline_seqs = np.load("RRdata/" + args.task + "offline_seqs.npy", allow_pickle=True)
    offline_logits = torch.load("RRdata/" + args.task + "offline_logits.pt").to(device)
    offline_logits = offline_logits.to(torch.float64)
    logits = torch.load("RRdata/" + args.task + "logits.pt").to(device)
    oracle_logits = torch.load("RRdata/" + args.task + "oracle_logits.pt").to(device)
    offline_labels = torch.load("RRdata/" + args.task + "offline_labels.pt").to(device)
    offline_coeffs = torch.load("RRdata/" + args.task + "offline_coeffs.pt").to(device)
    m = torch.load("model/" + args.task + "_m.pt").to(device)
    std = torch.load("model/" + args.task + "_std.pt").to(device)
    min_y = torch.load("model/" + args.task + "_min_y.pt")
    max_y = torch.load("model/" + args.task + "_max_y.pt")
    offline_m = torch.load("RRdata/" + args.task + "offline_m.pt").to(device)
    offline_std = torch.load("RRdata/" + args.task + "offline_std.pt").to(device)
    #load model
    oracle = SimpleMLP(1024, 512, 1, min_y, max_y).to(device)
    oracle.load_state_dict(torch.load("model/" + args.task + "_oracle.pt"))
    oracle.eval()
    with torch.no_grad():
        labels = oracle(oracle_logits.float()).double()
    labels = labels.data*(std + 1e-9) + m
    #min_labels, max_labels = torch.min(labels), torch.max(labels)
    #min_labels, max_labels = min_y, max_y
    min_labels = min_y*(std.data + 1e-9) + m.data
    max_labels = max_y*(std.data + 1e-9) + m.data
    #proxy
    proxy = OneMLP(1024, 512, 1).to(device)
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
    for i in range(128):
        #prepare candidate
        candidate = candidates[i]
        #exit(0)
        distill_len = int((len(candidate)+1)/2)
        candidate_input_id, _, _ = prepare_data([candidate], [1], tokenizer, device)
        distilled = candidate_input_id[:, 1:-1, 5:25].squeeze().data
        distilled0 = copy.deepcopy(distilled)
        distilled = torch.normal(mean=0.0, std=0.01, size=(distill_len, 20)).to(device)
        #assign
        for k in range(distill_len):
            desired_index = torch.argmax(distilled0[k])
            max_index = torch.argmax(distilled[k])
            max_value = copy.deepcopy(distilled[k, max_index])
            distilled[k, max_index] = copy.deepcopy(distilled[k, desired_index])
            distilled[k, desired_index] = max_value
        distilled.requires_grad = True
        
        distill_seq = {}
        distill_seq['token_type_ids'] = torch.zeros(1, distill_len+2).to(device)
        distill_seq['attention_mask'] = torch.ones(1, distill_len+2).to(device)
        #optimizer
        distilled_opt = optim.Adam([distilled], lr=args.lr)#, weight_decay=args.wd)
        if args.gamma_learn:
            eta = torch.Tensor([0.5]).to(device)
        else:
            eta = torch.Tensor([0.5]).to(device)
        #eta = torch.Tensor([1.0]).to(device)
        eta_m = torch.Tensor([0.0]).to(device)
        eta_v = torch.Tensor([0.0]).to(device)
        eta_beta = torch.Tensor([0.9, 0.999]).to(device)
        #predined target
        distill_label = torch.Tensor([10.0]).to(device).view(1, 1).to(torch.float64)
        for t in range(args.Tmax):
            distill_seq['input_ids'] = torch.zeros(1, distill_len+2, 30).to(device)
            distill_seq['input_ids'][0, 1:-1, 5:25] = torch.clone(m_func(distilled))
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
                    distilled_opt.load_state_dict(status)
                loss = eta * loss_l2h + (1 - eta) * loss_h2l
            distilled_opt.zero_grad()
            loss.backward()
            distilled_opt.step()


        max_seq = torch.argmax(distilled.data, dim=1)
        max_seq_one_hot = F.one_hot(max_seq, 20).to(torch.float64)
        distill_seq['input_ids'] = torch.zeros(1, distill_len+2, 30).to(device)
        distill_seq['input_ids'][0, 0, 2] = 1
        distill_seq['input_ids'][0, -1, 3] = 1
        distill_seq['input_ids'][0, 1:-1, 5:25] = max_seq_one_hot
        tmp1 = torch.argmax(distill_seq['input_ids'], dim=2)
        decoded = tokenizer.decode(tmp1[0])
        decoded = " ".join(decoded.split(" ")[1:-1])
        decoded_seq = re.sub(r"[UZOB]", "X", decoded)
        encoded_seq = tokenizer_oracle(decoded_seq, return_tensors='pt').to(device)
        with torch.no_grad():
            output = model_oracle(**encoded_seq)
            distill_logit = torch.mean(output[0], dim=1)
            scores[i] = oracle(distill_logit.float().data) * (std.data + 1e-9) + m.data
            scores[i] = (scores[i] - min_labels)/(max_labels - min_labels)
            print("sample {} \n score before {} \n score after {}".format(i, init_scores[i],  scores[i]))
    print("max value {} median value {}".format(torch.max(scores), torch.median(scores)))
    max_value, median_value = torch.max(scores), torch.median(scores)
    results = np.load("npy/resultsRR.npy", allow_pickle=True).item()
    key = args.task + "_" + args.mode + "_" + str(args.eta_learn) + "_" + str(args.gamma_learn) + "_" + str(args.seed) 
    results[key] = [max_value.data.cpu().numpy(), median_value.data.cpu().numpy()]
    np.save("npy/resultsRR.npy", results)


if __name__ == "__main__":
    print(args)
    set_seed(args.seed)
    if args.task_mode in ['oracle']:
        build_oracle(args)
    elif args.task_mode in ['distill']:
        distill(args)
