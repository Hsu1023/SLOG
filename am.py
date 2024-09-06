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
from train import split_dataset, split_batch, cal_eig, load_eig
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm

@torch.no_grad()
def evaluate(data_list, model, label, node_feat, multi_label=False):
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

def train(model, optimizer, epochs=config.epoch_num, mode='W'):
    global data_list_train, label, node_feat, data_list_valid, data_list_test, result
    best_test_acc = 0
    best_valid_acc = 0
    max_test_acc = 0
    if mode == 'W':
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad = True
                module.bias.requires_grad = True
            if isinstance(module, Velocity):
                for module2 in module.parameters():
                    module2.requires_grad = False
    elif mode == 'd':
        for name, module in model.named_children():
            if isinstance(module, Velocity):
                if isinstance(module, Velocity):
                    for module2 in module.parameters():
                        module2.requires_grad = True
            elif isinstance(module, nn.Linear):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    

    for epoch in tqdm(range(0, epochs)):
        model.train()
        random.shuffle(data_list_train)
        loss_sum = 0
        for _ in tqdm(range(len(data_list_train))):
            optimizer.zero_grad()
            idx, key_node, key_idx, sampling_node_list, La, U = data_list_train[_]
            batch_label = label[key_node]

            batch_node_feat = node_feat[sampling_node_list]
            La, U = La.cuda(), U.cuda()
            out, _ = model(batch_node_feat, La, U)
            out = out[key_idx]
            loss = F.nll_loss(out, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            loss_sum += loss.item()
            optimizer.step()

        # valid/test
        model.eval()
        valid_acc = evaluate(data_list_valid, model, label, node_feat)
        test_acc = evaluate(data_list_test, model, label, node_feat)

        if valid_acc > best_valid_acc:
            max_epoch = epoch
            best_valid_acc = valid_acc
            max_test_acc = test_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
        
        print("==> Epoch: {:03d}, metric: {}, loss: {:04f}, Valid: {:04f}, Test: {:04f}, max_test: {:04f}".format(epoch, 'f1_micro' if config.f1_score else 'acc', loss_sum, valid_acc, test_acc, max_test_acc))

    result.append((max_test_acc, best_valid_acc))
    return best_valid_acc



def main(resplit=False, sp = 0, directed = True):
    global data_list_train, label, node_feat, data_list_valid, data_list_test, result
    train_adj_list = [[] for i in range(config.node_num)]
    adj_list = [[] for i in range(config.node_num)]
    edge_index = []

    sp = 0 if config.dataset in ["squirrel_filtered", "chameleon_filtered", "amazon_ratings", "minesweeper", "tolokers", "roman_empire", "questions"] else None
    train_node_set, valid_node_set, test_node_set, node_num, label, node_feat, train_adj_list, adj_list, edge_index = split_dataset(train_adj_list, adj_list, edge_index, directed=directed, cur_data_split=sp)

    batch_list_train, batch_list_valid, batch_list_test = split_batch(train_node_set, valid_node_set, test_node_set)
    random.shuffle(batch_list_train)
    random.shuffle(batch_list_valid)
    random.shuffle(batch_list_test)

    cal_eig(batch_list_train, batch_list_valid, batch_list_test, node_num, train_adj_list, adj_list, edge_index) # store in the eig files

    data_list_train, data_list_valid, data_list_test = load_eig(batch_list_train, batch_list_valid, batch_list_test)
    node_feat = torch.Tensor(node_feat).cuda()
    label = torch.LongTensor(label).cuda()
    result = []

    model = SLOG_B(config.feat_dim, config.hidden_dim, config.class_num, softmax=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for i in range(0, int(config.epoch_num/20)):
        train(model, optimizer, epochs=10, mode='W')
        train(model, optimizer, epochs=10, mode='d')

    tests = np.array([i[0] for i in result])
    valids = np.array([i[1] for i in result])
    print(tests[valids.argmax()])
    return tests[valids.argmax()], max(tests)

    
def train_am(respilt=False, sp = 0, directed = True):
    download_data(config.dataset)
    return main(resplit=False, sp = 0, directed = True)

if __name__ == '__main__':
    print('inductive' if config.inductive else 'transductive')
    torch.cuda.set_device(config.gpu)
    download_data(config.dataset)
    main(resplit=False, sp = 0, directed = True)

