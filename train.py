import config
import pickle
from data_preprocessing import *
from model import *
import torch
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
from sklearn.metrics import f1_score
import time
if config.wandb:
    import wandb

def preprocess_eig_file(idx, node_list, node_num, train_adj_list, adj_list, edge_index, mode='train'):
    if mode == 'train':
        tmp_node_file = config.dataset_train_eig_folder + str(idx) + ".pkl"
    elif mode == 'valid':
        tmp_node_file = config.dataset_valid_eig_folder + str(idx) + ".pkl"
    elif mode == 'test':
        tmp_node_file = config.dataset_test_eig_folder + str(idx) + ".pkl"
    else:
        raise ValueError("mode should be valid or test")
    if os.path.isfile(tmp_node_file):
        return
    assert config.load_adj_list, "load_adj_list should be True"
    if mode == 'train':
        sampling_node_list = sampling(node_list, train_adj_list if config.inductive else adj_list, config.sample_num, config.khop, mode)
        batch_node_id_dic, batch_edge_index = convert_sampling_to_index(sampling_node_list, train_adj_list if config.inductive else adj_list, edge_index, node_num)
    else:
        sampling_node_list = sampling(node_list, adj_list, config.sample_num, config.khop, mode)
        batch_node_id_dic, batch_edge_index = convert_sampling_to_index(sampling_node_list, adj_list, edge_index, node_num)
    print(mode, batch_edge_index.shape[1]/len(sampling_node_list)/len(sampling_node_list))
    La, U = eig(len(sampling_node_list), batch_edge_index)
    La, U = torch.tensor(La, dtype=torch.float), torch.tensor(U, dtype=torch.float)
    with open(tmp_node_file, "wb") as f:
        pickle.dump([node_list, sampling_node_list, La, U], f)

def split_dataset(train_adj_list, adj_list, edge_index, directed = True, cur_data_split=None):
    assert cur_data_split is not None if config.dataset in ["squirrel_filtered", "chameleon_filtered", "amazon_ratings", "minesweeper", "tolokers", "roman_empire", "questions"] and not config.inductive else True, "cur_data_split should be None"

    dataset_dump_file = open(config.dataset_dump_file, "rb")
    data = pickle.load(dataset_dump_file)
    dataset_dump_file.close()
    [node_num, edge_index, node_feat, label, train_set, valid_set, test_set] = data

    node_num = edge_index.max() + 1
    print(np.array(edge_index).shape)
    
    if cur_data_split is not None:
        train_set, valid_set, test_set = train_set[cur_data_split], valid_set[cur_data_split], test_set[cur_data_split]
    # print(node_num, len(edge_index[0]))
    print("Start indexing...")
    if config.resplit:
        total_num_list = []
        for i in range(len(train_set)):
            if train_set[i]:
                total_num_list.append(i)
        for i in range(len(valid_set)):
            if valid_set[i]:
                total_num_list.append(i)
        for i in range(len(test_set)):
            if test_set[i]:
                total_num_list.append(i)
        label_node_num = len(total_num_list)
        print(label_node_num, node_num)
        random.shuffle(total_num_list)
        # print(total_num_list)
        train_set = [False for i in range(node_num)]
        valid_set = [False for i in range(node_num)]
        test_set = [False for i in range(node_num)]
        train_index = int(label_node_num * config.training_ratio)
        valid_index = int(label_node_num * (config.training_ratio + config.valid_ratio))
        for i in range(train_index):
            train_set[total_num_list[i]] = True
        for i in range(train_index, valid_index):
            valid_set[total_num_list[i]] = True
        for i in range(valid_index, label_node_num):
            test_set[total_num_list[i]] = True
        train_set = torch.as_tensor(train_set)
        valid_set = torch.as_tensor(valid_set)
        test_set = torch.as_tensor(test_set)

    train_node_set = np.where(train_set)[0]
    valid_node_set = np.where(valid_set)[0]
    test_node_set = np.where(test_set)[0]
    full_node_set = np.where(train_set | valid_set | test_set)[0]
    print("train_node_set", len(train_node_set))
    print("valid_node_set", len(valid_node_set))
    print("test_node_set", len(test_node_set))
    print("full_node_set", len(full_node_set))

    print("==> Start indexing...")

    if config.load_adj_list:
        # if not os.path.exists(config.dataset_index_file):
        if True:
            print("==> Create index file...")
            adj_list = [[] for i in range(node_num)]
            print("==> Start creating adj_list")

            if directed:
                for u, v in tqdm(zip(*edge_index)):
                    adj_list[u].append(v)
                    adj_list[v].append(u)
            else:
                for u, v in tqdm(zip(*edge_index)):
                    adj_list[u].append(v)

            print("==> Start creating train_adj_list")
            train_adj_list = [[] for i in range(node_num)]
            # index1 = np.isin(edge_index[0], train_node_set)
            # index2 = np.isin(edge_index[1], train_node_set)
            index1 = train_set[edge_index[0]]
            index2 = train_set[edge_index[1]]
            index = index1 & index2
            train_edge_index = [[], []]
            train_edge_index[0] = edge_index[0][index]
            train_edge_index[1] = edge_index[1][index]

            train_edge_index = np.array(train_edge_index)

            if directed:
                for u, v in tqdm(zip(*train_edge_index)):
                    train_adj_list[u].append(v)
                    train_adj_list[v].append(u)
            else:
                for u, v in tqdm(zip(*train_edge_index)):
                    train_adj_list[u].append(v)

            if not os.path.exists(os.path.dirname(config.dataset_index_file)):
                os.makedirs(os.path.dirname(config.dataset_index_file))

            print("==> Save index file...")
            with open(config.dataset_index_file, "wb") as f:
                pickle.dump([adj_list, train_adj_list], f)
        else:
            print("==> Load index file...")
            with open(config.dataset_index_file, "rb") as f:
                [adj_list, train_adj_list] = pickle.load(f)

    node_num = edge_index.max() + 1
    print('node_num', node_num)
    return train_node_set, valid_node_set, test_node_set, node_num, label, node_feat, train_adj_list, adj_list, edge_index

def split_batch(train_node_set, valid_node_set, test_node_set):
    print("==> Deciding preprocessed node batches ...")
    batch_list = [[],[],[]]
    for i, (dataset_eig_folder, node_set) in enumerate(zip(
        [config.dataset_train_eig_folder, config.dataset_valid_eig_folder, config.dataset_test_eig_folder], 
        [train_node_set, valid_node_set, test_node_set])):
        
        if not os.path.isfile(dataset_eig_folder + "index.pkl") or config.recompute_eig:
            if os.path.exists(dataset_eig_folder):
                os.system("rm -rf " + dataset_eig_folder)
            os.makedirs(dataset_eig_folder)
            idx = list(node_set)
            random.shuffle(idx)
            batches = math.ceil(len(node_set) / config.batch_size)
            print('batches', batches)
            for j in range(batches):
                batch_list[i].append((j, idx[j*config.batch_size:(j+1)*config.batch_size]))
            with open(dataset_eig_folder + "index.pkl", "wb") as f:
                pickle.dump(batch_list[i], f)
        else:
            with open(dataset_eig_folder + "index.pkl", "rb") as f:
                batch_list[i] = pickle.load(f)
    return batch_list

def cal_eig(batch_list_train, batch_list_valid, batch_list_test, node_num, train_adj_list, adj_list, edge_index):
    # if config.inductive:
    print("==> Start preprocessing train eig, total num {}...".format(len(batch_list_train)))
    for (idx, node_list) in tqdm(batch_list_train):
        preprocess_eig_file(idx, node_list, node_num, train_adj_list, adj_list, edge_index, mode='train')

    print("==> Start preprocessing valid eig, total num {}...".format(len(batch_list_valid)))
    for (idx, node_list) in tqdm(batch_list_valid):
        preprocess_eig_file(idx, node_list, node_num, train_adj_list, adj_list, edge_index, mode='valid')
    
    print("==> Start preprocessing test eig, total num {}...".format(len(batch_list_test)))
    for (idx, node_list) in tqdm(batch_list_test):
        preprocess_eig_file(idx, node_list, node_num, train_adj_list, adj_list, edge_index, mode='test')

def load_eig(batch_list_train, batch_list_valid, batch_list_test):
    print("==> Indexing train eig file...")
    data_list_train = []
    for i, node_list in tqdm(enumerate(batch_list_train)):
        tmp_node_file = config.dataset_train_eig_folder + str(i) + ".pkl"
        with open(tmp_node_file, "rb") as f:
            [node_list, sampling_node_list, La, U] = pickle.load(f)
        node_list = np.array(node_list)
        node_idx = np.where(sampling_node_list == node_list[:, None])[1]
        data_list_train.append([i, node_list, node_idx, sampling_node_list, La, U])

    print("==> Indexing full eig file...")
    data_list_valid, data_list_test = [], []
    for i, node_list in tqdm(enumerate(batch_list_valid)):
        tmp_node_file = config.dataset_valid_eig_folder + str(i) + ".pkl"
        with open(tmp_node_file, "rb") as f:
            [node_list, sampling_node_list, La, U] = pickle.load(f)
        node_list = np.array(node_list)
        node_idx = np.where(sampling_node_list == node_list[:, None])[1]
        data_list_valid.append([i, node_list, node_idx, sampling_node_list, La, U])
        
    for i, node_list in tqdm(enumerate(batch_list_test)):
        tmp_node_file = config.dataset_test_eig_folder + str(i) + ".pkl"
        with open(tmp_node_file, "rb") as f:
            [node_list, sampling_node_list, La, U] = pickle.load(f)
        node_list = np.array(node_list)
        node_idx = np.where(sampling_node_list == node_list[:, None])[1]
        data_list_test.append([i, node_list, node_idx, sampling_node_list, La, U])

    return data_list_train, data_list_valid, data_list_test

@torch.no_grad()
def evaluate(data_list, model, label, node_feat, adj_list, node_num, edge_index, multi_label=False):
    random.shuffle(data_list)
    test_correct_num, test_total_num = 0, 0
    pred_all, label_all = torch.tensor([]), torch.tensor([])
    for _ in range(len(data_list)):
        # tmp_id = node_to_full_list_dict[valid_node_set[_]]
        idx, key_node, key_idx, sampling_node_list, La, U = data_list[_]
        
        batch_label = label[key_node]
        batch_node_feat = node_feat[sampling_node_list]

        La, U = La.cuda(), U.cuda()
        logits, tmp_emd = model(batch_node_feat, La, U)
        logits = logits[key_idx]
        if multi_label:
            pred = torch.where(logits > 0, 1, 0).long()
        else:
            pred = logits.argmax(1)
        if config.f1_score:
            pred_all = torch.cat((pred_all, pred.cpu()))
            label_all = torch.cat((label_all, batch_label.cpu()))
        else:
            test_correct_num += pred.eq(batch_label).sum().item()
            test_total_num += len(key_node)

    if config.f1_score:
        test_acc = f1_score(label_all, pred_all, average='micro')
    else:
        test_acc = test_correct_num / test_total_num
    return test_acc

def remove_eig_file():
    if os.path.exists(config.dataset_train_eig_folder):
        os.system("rm -rf " + config.dataset_train_eig_folder)
    if os.path.exists(config.dataset_valid_eig_folder):
        os.system("rm -rf " + config.dataset_valid_eig_folder)
    if os.path.exists(config.dataset_test_eig_folder):
        os.system("rm -rf " + config.dataset_test_eig_folder)
    if os.path.exists(config.dataset_index_file):
        os.system("rm -rf " + config.dataset_index_file)

def train(resplit=False, sp = 0, directed = True, cur_data_split=None):
    if config.wandb:
        wandb.init(project=config.wandb_project or "tedsage_{}_{}".format(config.dataset, config.lr), config={
            'learning_rate': config.lr,
            'batch_size': config.batch_size,
            'weight_decay': config.weight_decay,
            'hidden_dim': config.hidden_dim,
            'sample_num': config.sample_num,
            'khop': config.khop,
            'dataset': config.dataset,
        } )
    multi_label = config.multi_label
    if config.recompute_eig:
        remove_eig_file()

    train_adj_list, adj_list, edge_index = None, None, None

    train_node_set, valid_node_set, test_node_set, node_num, label, node_feat, train_adj_list, adj_list, edge_index = split_dataset(train_adj_list, adj_list, edge_index, directed=directed, cur_data_split=cur_data_split)
    print('max', edge_index.max())
    print('feat', node_feat.shape)
    batch_list_train, batch_list_valid, batch_list_test = split_batch(train_node_set, valid_node_set, test_node_set)
    random.shuffle(batch_list_train)
    random.shuffle(batch_list_valid)
    random.shuffle(batch_list_test)

    cal_eig(batch_list_train, batch_list_valid, batch_list_test, node_num, train_adj_list, adj_list, edge_index) # store in the eig files
    # exit(0)
    data_list_train, data_list_valid, data_list_test = load_eig(batch_list_train, batch_list_valid, batch_list_test)

    print("==> Start training...")
    node_feat = torch.tensor(node_feat, dtype=torch.float).cuda()
    feat_dim = node_feat.shape[1]
    class_num = label.max().item() + 1

    if config.layer_num == 1 and config.res == False:
        model = SLOG_B(feat_dim, config.hidden_dim, class_num, softmax=not multi_label).cuda()
    else:
        model = SLOG_N(feat_dim, config.hidden_dim, class_num, softmax=not multi_label, layer_num=config.layer_num, res=config.res).cuda()

    label = torch.tensor(label).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_test_acc = 0
    best_valid_acc = 0
    max_test_acc = 0
    ve, ve2 = None, None
    for epoch in tqdm(range(0, config.epoch_num)):
        model.train()
        random.shuffle(data_list_train)
        loss_sum = 0
        for _ in tqdm(range(len(data_list_train))):
            optimizer.zero_grad()
            idx, key_node, key_idx, sampling_node_list, La, U = data_list_train[_]
            batch_label = label[key_node]

            batch_node_feat = node_feat[sampling_node_list]
            La, U = La.cuda(), U.cuda()
            # if model is nested GCN, then we need to add the key_node to the batch_node_feat
            out, _ = model(batch_node_feat, La, U)
            out = out[key_idx]
            if multi_label:
                loss = F.binary_cross_entropy_with_logits(out, batch_label.float())
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            else:
                loss = F.nll_loss(out, batch_label)
            loss.backward()
            loss_sum += loss.item()
            optimizer.step()

        # valid/test
        model.eval()
        # print(model.ve.ve)
        valid_acc = evaluate(data_list_valid, model, label, node_feat, train_adj_list if config.inductive else adj_list, node_num, edge_index, multi_label=multi_label)
        test_acc  = evaluate(data_list_test,  model, label, node_feat, train_adj_list if config.inductive else adj_list, node_num, edge_index, multi_label=multi_label)

        if config.wandb:
            wandb.log({"train_loss": loss_sum, "valid_accuracy": valid_acc, "test_accuracy": test_acc})
        if valid_acc > best_valid_acc:
            max_epoch = epoch
            best_valid_acc = valid_acc
            max_test_acc = test_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
        
        print("==> Epoch: {:03d}, metric: {}, loss: {:04f}, Valid: {:04f}, Test: {:04f}, max_test: {:04f}".format(epoch, 'f1_micro' if config.f1_score else 'acc', loss_sum, valid_acc, test_acc, max_test_acc))
        
    print("==> Max Epoch: {:03d}, Test: {:04f}".format(max_epoch, max_test_acc))
    print("==> Best Epoch: {:03d}, Test: {:04f}".format(best_epoch, best_test_acc))
    return max_test_acc, best_test_acc
