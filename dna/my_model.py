import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F

from transformers import BertModel, DNATokenizer
import copy
import numpy as np

config0 = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "do_sample": False,
  "eos_token_ids": 0,
  "finetuning_task": None,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "is_decoder": False,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1
  },
  "layer_norm_eps": 1e-12,
  "length_penalty": 1.0,
  "max_length": 20,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_beams": 1,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "num_return_sequences": 1,
  "num_rnn_layer": 1,
  "output_attentions": False,
  "output_hidden_states": False,
  "output_past": True,
  "pad_token_id": 0,
  "pruned_heads": {},
  "repetition_penalty": 1.0,
  "rnn": "lstm",
  "rnn_dropout": 0.0,
  "rnn_hidden": 768,
  "split": 10,
  "temperature": 1.0,
  "top_k": 50,
  "top_p": 1.0,
  "torchscript": False,
  "type_vocab_size": 2,
  "use_bfloat16": False,
  "vocab_size": 69
}


BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config=config0):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Linear(config["vocab_size"], config["hidden_size"], bias=False)
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])

        self.LayerNorm = BertLayerNorm(config["hidden_size"], eps=1e-12)

    def forward(self, input_ids, token_type_ids=None):
        #input_ids, bs*seq_length*vocab_size
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[:, :, 0])
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[:, :, 0]).long()

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        model = BertModel.from_pretrained("DNABertModel")
        self.embeddings = BertEmbeddings()
        self.embeddings.load_state_dict(torch.load("pretrained_models/bertembeddings.pt"))
        self.encoder = copy.deepcopy(model.encoder)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden_states=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[:, :, 0]).long()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[:, :, 0]).long()

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)#, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                    extended_attention_mask)
        sequence_output = encoded_layers[-1]
        output = torch.mean(sequence_output[:, 1:-1, :], dim=1)
        return output

class Latent2Seq(nn.Module):
    def __init__(self, seq_len, alphabet_len):
        super(Latent2Seq, self).__init__()
        self.seq_len = seq_len
        self.alphabet_len = alphabet_len
        self.latent2seq = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, int(seq_len*alphabet_len))
        )
    def forward(self, latent):
        seq_repre = self.latent2seq(latent).view(self.seq_len, self.alphabet_len)
        return F.softmax(seq_repre, dim=1)

class GenBert(nn.Module):
    def __init__(self):
        super(GenBert, self).__init__()
        model = BertModel.from_pretrained("DNABertModel")
        self.embeddings = BertEmbeddings()
        self.embeddings.load_state_dict(torch.load("pretrained_models/bertembeddings.pt"))
        self.encoder = copy.deepcopy(model.encoder)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_hidden_states=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[:, :, 0]).long()
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[:, :, 0]).long()

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)#, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                    extended_attention_mask)
        sequence_output = encoded_layers[-1]
        #output = torch.mean(sequence_output[:, 1:-1, :], dim=1)
        output = sequence_output[:, 0, :]
        return output

class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 min_y,
                 max_y):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
            nn.Tanh())
        self.w = (max_y - min_y)/2.0
        self.b = (max_y + min_y)/2.0
        
    def forward(self, x):
        return self.w*self.main(x) + self.b

class SimpleMLP1(nn.Module):

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 min_y=0.0,
                 max_y=0.0):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)

class OneMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int):
        super().__init__()
        self.main = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.main(x)

def RBF(X, Y, l=1.0):
    #input: X NxD, Y MxD
    #output: NxM
    #X_norm = torch.sum(X ** 2, dim=-1)
    #Y_norm = torch.sum(Y ** 2, dim=-1)
    #K = torch.exp(-(X_norm[:, None] + Y_norm[None, :] - 2*torch.mm(X, Y.T))/(2*l**2))
    #return K
    return torch.mm(X, Y.T)


if __name__ == "__main__":
    model = BertEmbeddings()
    model.load_state_dict(torch.load("pretrained_models/bertembeddings.pt"))
    print(model)
    exit(0)
    model = model.cuda()
    input_ids = torch.zeros(1, 1, 69).cuda()
    input_ids[0, 0, 12] = 1
    print("info", input_ids.shape, type(input_ids))
    print(model(input_ids.cuda()))
