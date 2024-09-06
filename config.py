dataset = 'Citeseer' # choice: ['Citeseer', 'squirrel', 'chameleon', "squirrel_filtered", "chameleon_filtered", "amazon_ratings", "minesweeper", "tolokers", "questions", "Cora", "DBLP", "CS", "Physics", "arxiv", "Flickr", "Reddit"]

optimization = 'am' # choice: ['gp' for gaussian process, 'am' for alternating minimization, 'normal' for Adam]

model = 'SLOG_B' # choice: ['SLOG_B', 'SLOG_N']

assert model == 'SLOG_B' or optimization not in ['gp', 'am'], "SLOG_N is not currently supported with gp or am"

gpu = 0

small_graph = dataset not in [
    "arxiv",
    "Reddit",
    "Flickr",
]
if small_graph:
    load_adj_list = True
    recompute_eig = True
    resplit = True
    lr = 1e-2
    epoch_num = 500
    hidden_dim = 128
else: # large graph
    load_adj_list = True
    recompute_eig = False
    resplit = False
    lr = 1e-4
    epoch_num = 100
    hidden_dim = 512

if model == 'SLOG_N':
    layer_num = 3
    res = True
elif model == 'SLOG_B':
    layer_num = 1
    res = False
else:
    raise ValueError("model not supported")


if resplit:
    assert recompute_eig
if recompute_eig:
    assert load_adj_list

matrix_scale = False

weight_decay = 5e-4  # 5e-3
batch_size = 512
sample_num = 2
khop = 25

wandb = False
wandb_project = "tedgcn_{}_{}_ve".format(dataset, lr)

training_ratio = 0.6
valid_ratio = 0.2

import uuid
if small_graph:
    mid = "_" + uuid.uuid4().hex[:8]
else:
    mid = ""
dataset_dump_file = "./datasets/" + dataset + ".pkl"
dataset_index_file = "./datasets/index/" + dataset + mid +  ".pkl"
dataset_train_eig_folder = "./datasets/train_eig/" +  dataset + mid + "/"
dataset_valid_eig_folder = "./datasets/valid_eig/" + dataset + mid + "/"
dataset_test_eig_folder = "./datasets/test_eig/" + dataset + mid + "/"


import os

if recompute_eig:
    if not small_graph:
        user_input = input("Are you sure to remove all the eig files? [Y/n] ")
        if user_input != "Y":
            print("exit")
            exit()
    if os.path.exists(dataset_train_eig_folder):
        os.system("rm -rf " + dataset_train_eig_folder)
    if os.path.exists(dataset_valid_eig_folder):
        os.system("rm -rf " + dataset_valid_eig_folder)
    if os.path.exists(dataset_test_eig_folder):
        os.system("rm -rf " + dataset_test_eig_folder)
    if os.path.exists(dataset_index_file):
        os.system("rm -rf " + dataset_index_file)

if os.path.exists(dataset_train_eig_folder) == False:
    os.makedirs(dataset_train_eig_folder)
if os.path.exists(dataset_valid_eig_folder) == False:
    os.makedirs(dataset_valid_eig_folder)
if os.path.exists(dataset_test_eig_folder) == False:
    os.makedirs(dataset_test_eig_folder)


dataset_info = {
    "arxiv": {"feat_dim": 128, "class_num": 40, "node_num": 169343},
    "chameleon": {"feat_dim": 2325, "class_num": 5, "node_num": 2277, "inductive": True,},
    "squirrel": {"feat_dim": 2089, "class_num": 5, "node_num": 2354, "inductive": True},
    "CS": {"feat_dim": 6805, "class_num": 15, "node_num": 18333, "inductive": True},
    "Physics": {"feat_dim": 8415, "class_num": 5, "node_num": 34493, "inductive": True},
    "DBLP": {"feat_dim": 1639, "class_num": 4, "node_num": 17716, "inductive": True},
    "Cora": {"feat_dim": 1433, "class_num": 7, "node_num": 2708, "inductive": True},
    "Citeseer": {"feat_dim": 3703, "class_num": 6, "node_num": 3327, "inductive": True},
    "Reddit": {"feat_dim": 602, "class_num": 41, "node_num": 232965, "inductive": True},
    "Flickr": {"feat_dim": 500, "class_num": 7, "node_num": 89250, "inductive": True},
    "chameleon_filtered": {"feat_dim": 2325, "class_num": 5, "node_num": 890, "inductive": True},
    "squirrel_filtered": {"feat_dim": 2089, "class_num": 5, "node_num": 2223, "inductive": True},
    "tolokers": {"feat_dim": 10, "class_num": 2, "node_num": 11758, "inductive": True},
    "minesweeper": {"feat_dim": 7, "class_num": 2, "node_num": 10000, "inductive": True,},
    "amazon_ratings": {"feat_dim": 300, "class_num": 5, "node_num": 24492, "inductive": True,},
    "questions": {"feat_dim": 301, "class_num": 2, "node_num": 24492, "inductive": True,},
}

if dataset in dataset_info.keys():
    feat_dim = dataset_info[dataset]["feat_dim"]
    class_num = dataset_info[dataset]["class_num"]

    node_num = dataset_info[dataset]["node_num"]
    if "multi_label" in dataset_info[dataset].keys():
        multi_label = dataset_info[dataset]["multi_label"]
    else:
        multi_label = False
    if "inductive" in dataset_info[dataset].keys():
        inductive = dataset_info[dataset]["inductive"]
    else:
        inductive = False
else:
    inductive = True
    multi_label = False
f1_score = inductive
f1_score = False

print("inductive" if inductive else "transductive")
