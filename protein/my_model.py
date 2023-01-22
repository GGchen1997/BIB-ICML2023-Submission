import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import copy
import numpy as np

config0 = {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.0,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "max_position_embeddings": 40000,
  "num_attention_heads": 16,
  "num_hidden_layers": 30,
  "type_vocab_size": 2,
  "vocab_size": 30
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
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.embeddings = BertEmbeddings()
        self.embeddings.load_state_dict(torch.load("pretrained_models/bertembeddings.pt"))
        self.encoder = copy.deepcopy(model.encoder)
        self.pooler = copy.deepcopy(model.pooler)

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
                                      extended_attention_mask,
                                      output_hidden_states=output_hidden_states)
        sequence_output = encoded_layers[-1]
        #output = self.head(torch.mean(sequence_output, dim=1))
        #output = torch.mean(sequence_output, dim=1)
        output = torch.mean(sequence_output[:, 1:-1, :], dim=1)
        return output#, sequence_output[0, 0], sequence_output[0, -1]

class Latent2Seq(nn.Module):
    def __init__(self, seq_len, alphabet_len):
        super(Latent2Seq, self).__init__()
        self.seq_len = seq_len
        self.alphabet_len = alphabet_len
        self.latent2seq = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, int(seq_len*alphabet_len))
        )
    def forward(self, latent):
        seq_repre = self.latent2seq(latent).view(self.seq_len, self.alphabet_len)
        return F.softmax(seq_repre, dim=1)

class GenBert(nn.Module):
    def __init__(self):
        super(GenBert, self).__init__()
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        self.embeddings = BertEmbeddings()
        self.embeddings.load_state_dict(torch.load("pretrained_models/bertembeddings.pt"))
        self.encoder = copy.deepcopy(model.encoder)
        self.pooler = copy.deepcopy(model.pooler)

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
        self.main = nn.Linear(in_dim, out_dim)
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

if __name__ == "__main__":
    device = torch.device('cuda:' + str(3))
    model = BertEmbeddings()
    model.load_state_dict(torch.load("pretrained_models/bertembeddings.pt"))
    print(model)
    exit(0)
    model = model.to(device)
    #model = model.cuda()
    input_ids = torch.zeros(1, 1, 30).cuda()
    input_ids[0, 0, 12] = 1
    print("info", input_ids.shape, type(input_ids))
    print(model(input_ids.to(device)))
    
    #sequences = ["A E T C Z A O"]#, "S K T Z P"]
    #label = [0, 1]
    #device = torch.device('cuda:7')
    #tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    #tmp, input_ids, attention_mask, label_batch = prepare_data(sequences, label, tokenizer, device)
    #logit = model(input_ids=input_ids, attention_mask=attention_mask)
