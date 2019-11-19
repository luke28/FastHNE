import os
import sys
import json
import argparse
import networkx as nx
import random
import torch
import numpy as np
import pickle
import pdb
from datetime import datetime
import torch.optim as optim
from collections import deque
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


from alias_table_sampling import AliasTable as at
from utils import graph_handler as gh
from utils.lib_ml import MachineLearningLib as mll
from utils.data_handler import DataHandler as dh

from params import * 

from FastHGNE import FastHGNE

def construct_graph(G):
    new_G = nx.Graph()
    new_G.graph['degree'] = 0
    dq = deque()
    for iter in range(num_walks):
        for u in G.nodes():
            dq.clear()
            dq.append(u)
            v = u
            if v not in new_G:
                new_G.add_node(v)
                new_G.node[v]['degree'] = 0
            for t in range(walk_length):
                adj = list(G[v])
                v_id = random.randint(0, len(adj) - 1)
                v = adj[v_id]
                if v not in new_G:
                    new_G.add_node(v)
                    new_G.node[v]['degree'] = 0
                for it in dq:
                    if it in new_G[v]:
                        new_G[v][it]['weight'] += 1
                    else:
                        new_G.add_edge(v, it, weight = 1)
                    new_G.graph['degree'] += 1
                    new_G.node[v]['degree'] += 1
                    new_G.node[it]['degree'] += 1
                dq.append(v)
                if len(dq) > window_size:
                    dq.popleft()
    return new_G


def init_tree(file_path):
    tree, n, m = gh.load_tree(file_path)
    layer_nodes = []
    ancestors = [None for i in range(m)]
    def dfs(u, pre, dep):
        if len(tree[u]) == 0:
            ancestors[u] = [pre[i] for i in range(1, len(pre))]
            return
        if dep >= len(layer_nodes):
            layer_nodes.append([u])
        else:
            layer_nodes[dep].append(u)
        pre.append(u)
        for v in tree[u]:
            dfs(v, pre, dep+1)
        pre.pop()
    dfs(n - 1, [], 0)
    num_layer = len(layer_nodes)
    id2node = [{i : it for i, it in enumerate(lst)} for lst in layer_nodes]
    node2id = [{dic[key] : key for key in dic} for dic in id2node]
    layer_nodes = [[node2id[i][u] for u in lst] for i, lst in enumerate(layer_nodes)]
    ancestors = [[node2id[i+1][u] for i, u in enumerate(lst)] for lst in ancestors]
    Tree = namedtuple('Tree', 'tree id2node node2id layer_nodes ancestors')
    return Tree(tree = tree, layer_nodes = layer_nodes,
            ancestors = ancestors, id2node = id2node, node2id = node2id)

def get_sampler(G):
    n = G.number_of_nodes()
    Sampler = namedtuple('Sampler', 'sampler idx')
    pos_samplers = {i : Sampler(idx = list(G[i]),
        sampler = at([G[i][j]['weight'] for j in G[i]]))  for i in G.nodes()}
    neg_sampler = at([G.node[i]['degree'] for i in G.nodes()])
    return pos_samplers, neg_sampler

def batch_strategy(pos_sampler, neg_sampler, tree):
    def get_batch(maxx):
        ancestors = tree.ancestors
        def father_list(s):
            res = [it for it in ancestors[s]]
            res.append(s)
            return res
        for _ in range(maxx):
            batch_u = [father_list(neg_sampler.sample()) \
                    for i in range(batch_size)]
            batch_pos_v =list(zip(*[father_list(pos_sampler[lst[-1]].idx[\
                    pos_sampler[lst[-1]].sampler.sample()])\
                    for lst in batch_u]))
            batch_u = list(zip(*batch_u))
            batch_neg_v = list(zip(*[father_list(neg_sampler.sample()) \
                    for i in range(batch_size * neg_sample_num)]))
            batch_neg_v = np.array(batch_neg_v).reshape( \
                    (-1, batch_size, neg_sample_num)).tolist()
            yield batch_u, batch_pos_v, batch_neg_v
    return get_batch

def train(get_batch, tree):
    params = {"dims":dims, \
            "num_layers":num_layers, \
            "num_nodes" : num_nodes, \
            "num_layer_nodes" : [len(tree.layer_nodes[i]) \
            for i in range(1, len(tree.layer_nodes))]}
    params["num_layer_nodes"].append(num_nodes)
    model = FastHGNE(params, tree.ancestors)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    average_loss = 0.0
    total_time = 0.0
    for i, (u, pos_v, neg_v) in enumerate(get_batch(epoch_num)):
        #print(u, pos_v, neg_v)
        #pdb.set_trace()
        start_time = datetime.now()
        optimizer.zero_grad()
        loss = model(u, pos_v, neg_v)
        average_loss += loss
        loss.backward()
        optimizer.step()
        end_time = datetime.now()
        total_time += (end_time - start_time).total_seconds()

        if (i + 1) % 300 == 0:
            print(average_loss / 300.0)
            average_loss = 0.0

    model.save_embeddings(save_path)
    print("Finished optimization, total time: " + str(total_time))


def metric(save_path, label_path):
    with open(save_path, "rb") as f:
        X = pickle.load(f)["embeddings"]
    X_scaled = scale(X)
    y = dh.load_ground_truth(label_path)
    y = y[:len(X)]
    acc = 0.0
    for _ in range(metric_times):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, stratify = y)
        clf = mll.logistic(X_train, y_train, {})
        ret = mll.infer(clf, X_test, y_test)
        acc += ret[1]
    acc /= float(metric_times)
    print(acc)


def main():
    random.seed(177)
    np.random.seed(2317)
    torch.cuda.manual_seed(179)
    torch.manual_seed(179)

    is_train = False
    if is_train:
        G = gh.load_unweighted_digraph(graph_path, False)
        new_G = construct_graph(G)
        print("[+] Finished construct the graph.")
        pos_samplers, neg_sampler = get_sampler(new_G)
        tree = init_tree(tree_path)
        print("[+] Finished init the tree.")
        get_batch = batch_strategy(pos_samplers, neg_sampler, tree)
        train(get_batch, tree)
    metric(save_path, label_path)
    #return new_G, pos_samplers, neg_sampler


if __name__ == "__main__":
    main()
