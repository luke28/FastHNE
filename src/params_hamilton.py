from utils.env import *
import sys, os

epoch_num = 1000
batch_size = 32
lr = 0.2

walk_length = 40
num_walks = 2
window_size = 10

dims = [60, 2, 2]
neg_sample_num = 4
num_layers = 3
num_nodes = 2314

graph_path = os.path.join(DATA_PATH, "graph_hamilton")
tree_path = os.path.join(DATA_PATH, "tree2_hamilton")
save_path = os.path.join(RES_PATH, "embeddings_hamilton.pkl")
label_path = os.path.join(DATA_PATH, "flag_hamilton")

metric_times = 3
test_size = 0.1
