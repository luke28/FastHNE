import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
import numpy as np
import pickle
import pdb

from utils import common_tools as ct
from params import *

class FastHGNE(nn.Module):
    def __init__(self, params, ancestors):
        super(FastHGNE, self).__init__()
        self.ancestors = ancestors
        self.params = ct.obj_dic(params)
        self.layer_embeddings = nn.ModuleList([nn.Embedding(self.params.num_layer_nodes[i], self.params.dims[i]) for i in range(self.params.num_layers)])
        self.total_dim = np.sum(self.params.dims)
        self.init_var()

    def init_var(self):
        pass
        initrange = 0.5 / self.total_dim
        for embeddings in self.layer_embeddings:
            embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, u, v_pos, v_neg):
        '''
        Dim:
            u = batch_size
            v_pos = batch_size
            v_neg = batch_size * k
        '''
        #pdb.set_trace()
        if torch.cuda.is_available():
            u = [torch.LongTensor(i).cuda() for i in u]
            pos_v = [torch.LongTensor(i).cuda() for i in v_pos]
            neg_v = [torch.LongTensor(i).cuda() for i in v_neg]
        else:
            u = [torch.LongTensor(i) for i in u]
            pos_v = [torch.LongTensor(i) for i in v_pos]
            neg_v = [torch.LongTensor(i) for i in v_neg]

        embed_us = [self.layer_embeddings[i](u[i]) for i in range(num_layers)]
        embed_neg_vs = [self.layer_embeddings[i](neg_v[i]) for i in range(num_layers)]
        embed_pos_vs = [self.layer_embeddings[i](pos_v[i]) for i in range(num_layers)]
        # batch_size * d
        embed_u = torch.cat(embed_us, dim = 1)
        embed_pos_v = torch.cat(embed_pos_vs, dim = 1)
        # batch_size * k * d
        embed_neg_v = torch.cat(embed_neg_vs, dim = 2)

        #print(embed_u.shape, embed_pos_v.shape)
        #pdb.set_trace()
        score  = torch.mul(embed_u, embed_pos_v)
        score = torch.sum(score, dim=1)
        log_target = Func.logsigmoid(score).squeeze()

        neg_score = torch.bmm(embed_neg_v, embed_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = Func.logsigmoid(-1*neg_score).squeeze()

        loss = - log_target - sum_log_sampled
        loss = loss.mean()

        return loss

    def save_embeddings(self, save_path):
        embs = np.zeros((self.params.num_nodes, self.total_dim), dtype = np.float32)
        embs[:, -self.params.dims[-1]:] = self.layer_embeddings[-1].weight.cpu().data.numpy()
        pre = 0
        for i in range(len(self.params.dims) - 1):
            embs[:, pre: pre + self.params.dims[i]] = np.array( \
                    [self.layer_embeddings[i].weight.cpu().data.numpy()[lst[i]] \
                    for lst in self.ancestors], dtype = np.float32)
            pre += self.params.dims[i]
        with open(save_path, "wb") as f:
            pickle.dump({"embeddings": embs}, f)

