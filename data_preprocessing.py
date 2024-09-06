from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork
from torch_geometric.datasets import Coauthor, Amazon, CitationFull, LINKXDataset, Reddit, Yelp, PPI, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import numpy as np
from scipy.sparse import csr_matrix, find, spdiags, linalg, csc_matrix
import pickle
import config
import copy
import random
import time
import os

def row_normalize(A):
    '''
    Perform row-normalization of the given matrix

    inputs
        A : crs_matrix
            (n x n) input matrix where n is # of nodes
    outputs
        nA : crs_matrix
             (n x n) row-normalized matrix

    '''
    n = A.shape[0]

    # do row-wise sum where d is out-degree for each node
    d = A.sum(axis=1)
    d = np.asarray(d).flatten()

    # handle 0 entries in d
    d = np.maximum(d, np.ones(n))
    invd = 1.0 / d
    sqrt_invd = np.sqrt(invd)
    invD = np.outer(sqrt_invd, sqrt_invd)

    # 0.5(I + D^-1 * A) = I - 0.5 L_sym 
    return csc_matrix(np.array(0.5*(np.eye(n) + invD*(A.A))))

def get_real_matrix(U):
    new_U = np.zeros((U.shape[0], U.shape[1]))
    print(list(U[0]))
    print(U[0][0][0])
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            new_U[i][j] = U[i][j].real
    return new_U

def gen_adj_matrix(edge_index, node_num, directed=True,weighted=False):
    edge_weight = [1 for i in range(edge_index.shape[1])]
    A = csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(node_num, node_num))
    if directed:
        A = (A + A.T)
    if not weighted:
        # no redundant edges are allowed for unweighted graphs
        I, J, K = find(A)
        A = csr_matrix((np.ones(len(K)), (I, J)), shape=A.shape)
    return A

def remove_isolated_nodes(A):
    d = A.sum(axis=1)
    d = np.asarray(d).flatten()
    keep_nodes = np.where(d>0)
    return keep_nodes

def sampling(nodes, adj_list, num_sample=20, khop=2, mode='train'):
    if adj_list is None:
        raise ValueError("adj_list is None")
    tmp_nodes = set(nodes)
    if num_sample < 0:
        for _ in range(0, khop):
            to_neighs = [set(adj_list[int(node)]) for node in tmp_nodes]
            tmp_nodes = tmp_nodes | set.union(*to_neighs)
    else:
        for _ in range(0, khop):
            to_neighs = [set(random.sample(adj_list[int(node)], min(num_sample, len(adj_list[int(node)])))) for node in tmp_nodes]
            tmp_nodes = tmp_nodes | set.union(*to_neighs)
    sampling_node_list = list(tmp_nodes)
    return sampling_node_list

def convert_sampling_to_index(sampling_node_list, adj_list, edge_index, node_num):
    # id_dic = dict() # old -> new
    id_dic_array = np.ones((node_num,)) * -1
    for i in range(len(sampling_node_list)):
        id_dic_array[sampling_node_list[i]] = i

    sampling_node_mask = np.zeros((node_num,), dtype=np.bool_)
    sampling_node_mask[sampling_node_list] = True

    sampling_edge_index = [[], []]
    t1 = time.time()
    index1 = sampling_node_mask[edge_index[0]]
    index2 = sampling_node_mask[edge_index[1]]
    index = index1 & index2
    sampling_edge_index[0] = edge_index[0][index]
    sampling_edge_index[0] = id_dic_array[sampling_edge_index[0]]
    sampling_edge_index[1] = edge_index[1][index]
    sampling_edge_index[1] = id_dic_array[sampling_edge_index[1]]

    sampling_edge_index = np.array(sampling_edge_index)
    t2 = time.time()
    # print("build index: {}".format(t2 - t1))
    return id_dic_array, np.array(sampling_edge_index)


def reconstruct_data(edge_index, node_feat, node_num, label):
    A = gen_adj_matrix(edge_index, node_num)
    keep_nodes = remove_isolated_nodes(A)
    if len(keep_nodes[0]) == node_num:
        return None
    else:
        new_node_feat = node_feat[keep_nodes[0]]
        new_label = label[keep_nodes[0]]
        new_node_num = len(keep_nodes[0])
        nodeold2new = dict()
        for i in range(len(list(keep_nodes))):
            nodeold2new[keep_nodes[i]] = i
        new_edge_index = np.array([[], []])
        for i in range(len(edge_index[0])):
            new_edge_index[0].append(nodeold2new[edge_index[0][i]])
            new_edge_index[1].append(nodeold2new[edge_index[1][i]])
        return new_edge_index, new_node_feat, new_node_num, new_label


def download_data(data_name):
    if data_name in ["squirrel_filtered", "chameleon_filtered", "amazon_ratings", "minesweeper", "tolokers", "roman_empire", "questions"] or  'syn_' in data_name:
        data = np.load('./data/' + data_name + '.npz')
        val_masks = torch.tensor(data['val_masks'])
        [torch.where(val_mask)[0] for val_mask in val_masks]
        edges = data['edges'].T
        data = Data(x=torch.from_numpy(data['node_features']).float(), edge_index=torch.from_numpy(edges).long(), y=torch.from_numpy(data['node_labels']).long(), train_mask=torch.from_numpy(data['train_masks']).bool(), val_mask=torch.from_numpy(data['val_masks']).bool(), test_mask=torch.from_numpy(data['test_masks']).bool())
        datasets = [data]
    elif data_name in ["arxiv"]:
        datasets = PygNodePropPredDataset(name = 'ogbn-arxiv', transform=T.ToSparseTensor())
        datasets.data.y = datasets.data.y[:,0]
        dataset = datasets.data
    elif data_name in ["PubMed", "Cora", "Citeseer"]:
        datasets = Planetoid(root='./datasets/' + data_name, name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["computers", "photo"]:
        datasets = Amazon(root='./datasets/' + data_name, name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["CS", "Physics"]:
        datasets = Coauthor(root='./datasets/' + data_name, name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["Actor"]:
        datasets = Actor(root='./datasets/' + data_name, transform=T.NormalizeFeatures())
        #datasets = Actor(root='./datasets/')
    elif data_name in ["Cornell", "Texas", "Wisconsin"]:
        datasets = WebKB(root='./datasets/' + data_name, name=data_name)
    elif data_name in ["chameleon", "squirrel"]:
        datasets = WikipediaNetwork(
            root='./datasets/' + data_name, name=data_name, geom_gcn_preprocess=True)
        """
        datasets = WikipediaNetwork(
            root='./datasets/', name=data_name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
        """
    
    elif data_name in ["crocodile"]:
        datasets = WikipediaNetwork(
            root='./datasets/' + data_name, name=data_name, geom_gcn_preprocess=False)
    elif data_name in ["DBLP"]:
        datasets = CitationFull(root='./datasets/' + data_name, name=data_name)
    elif data_name in ["penn94",  "cornell5"]:
        datasets = LINKXDataset(root='./datasets/' + data_name, name=data_name)
    elif data_name in ["Reddit"]:
        datasets = Reddit(root='./datasets/Reddit/')
    elif data_name in ["Yelp"]:
        datasets = Yelp(root='./datasets/Yelp/', transform=T.NormalizeFeatures())
    elif data_name in ["Flickr"]:
        datasets = Flickr(root='./datasets/Flickr/')

    if data_name not in ["arxiv", "mag"]:
        dataset = datasets[0]
    if data_name in ["", ""]:
        preProcDs = WikipediaNetwork(
            root='../data/', name=data_name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        edge_index = preProcDs[0].edge_index
    elif data_name in ["arxiv"]:
        edge_index = torch.concatenate((dataset.edge_index, torch.flip(dataset.edge_index, [0])), dim=1)
        # self loop
        edge_index = torch.cat([edge_index, torch.arange(0, dataset.num_nodes).repeat(2,1)], dim=1)
        edge_index = np.array(edge_index)
    else:
        edge_index = np.array(dataset.edge_index)
    node_feat = np.array(dataset.x)
    label = np.array(dataset.y)
    print(label.shape)
    node_num = dataset.num_nodes

    print(label)
    print(node_num, node_feat.shape, np.max(label), np.min(label), data_name)
    
    if data_name in ["computers", "photo", "CS", "Physics", "DBLP", "penn94", "cornell5"]:
        total_num_list = [i for i in range(node_num)]
        label_node_num = len(total_num_list)
        print(label_node_num)
        random.shuffle(total_num_list)
        # print(total_num_list)
        train_set = np.array([False for i in range(node_num)])
        valid_set = np.array([False for i in range(node_num)])
        test_set = np.array([False for i in range(node_num)])
        train_index = int(label_node_num * config.training_ratio)
        valid_index = int(label_node_num * (config.training_ratio + config.valid_ratio))
        for i in range(train_index):
            train_set[total_num_list[i]] = True
        for i in range(train_index, valid_index):
            valid_set[total_num_list[i]] = True
        for i in range(valid_index, label_node_num):
            test_set[total_num_list[i]] = True
        print(train_set.shape)
    elif data_name in ["crocodile"]:
        assert config.inductive == True
        train_set = np.array([True for i in range(node_num)])
        valid_set = np.array([False for i in range(node_num)])
        test_set = np.array([False for i in range(node_num)])
    elif data_name in ["chameleon", "squirrel", "Cornell", "Texas", "Wisconsin", "Actor"]:
        train_set = np.array(dataset.train_mask[:, 0])
        valid_set = np.array(dataset.val_mask[:, 0])
        test_set = np.array(dataset.test_mask[:, 0])
        print(sum(train_set), sum(valid_set), sum(test_set))
    elif data_name in ["PubMed", "Cora", "Citeseer", "reed98", "Reddit", "Yelp", "Flickr"] or 'syn_' in data_name:
        train_set = np.array(dataset.train_mask)
        valid_set = np.array(dataset.val_mask)
        test_set = np.array(dataset.test_mask)
    elif data_name in ["arxiv", "mag"]:
        split_idx = datasets.get_idx_split()
        train_node = split_idx["train"]
        valid_node = split_idx["valid"]
        test_node = split_idx["test"]
        train_set = np.array([False for i in range(node_num)])
        valid_set = np.array([False for i in range(node_num)])
        test_set = np.array([False for i in range(node_num)])
        train_set[train_node] = True
        valid_set[valid_node] = True
        test_set[test_node] = True
    elif data_name in ["squirrel_filtered", "chameleon_filtered", "amazon_ratings", "minesweeper", "tolokers", "roman_empire", "questions"]: 
        train_set = np.array(data.train_mask)
        valid_set = np.array(data.val_mask)
        test_set = np.array(data.test_mask)

    if not os.path.isfile(config.dataset_dump_file) or config.resplit:
        dataset_dump_file = open(config.dataset_dump_file, "wb")
        data = [node_num, edge_index, node_feat, label, train_set, valid_set, test_set]
        pickle.dump(data, dataset_dump_file)
        dataset_dump_file.close()
    return node_num, edge_index, node_feat, label, train_set, valid_set, test_set

def check_sym(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] != A[j][i]:
                return False
    return True

def eig(node_num, edge_index, device='cuda:0'):
    adj_matrix = gen_adj_matrix(edge_index, node_num)
    norm_adj_matrix = row_normalize(adj_matrix).A
    adj_matrix = adj_matrix.A
    t1 = time.time()
    if config.matrix_scale:
        print("matrix scale:" , norm_adj_matrix.shape)
    if device != 'cpu':
        try:
            La, U = torch.linalg.eigh(torch.from_numpy(norm_adj_matrix).cuda())
            La, U = La.cpu().numpy(), U.cpu().numpy()
        except Exception as e:
            La, U = np.linalg.eigh(norm_adj_matrix)
    else:
        La, U = np.linalg.eigh(norm_adj_matrix)
    t2 = time.time()

    La = np.real(La)+0.00000001
    return La, U

def pusedo_eig(node_num, edge_index):
    adj_matrix = gen_adj_matrix(edge_index, node_num)
    norm_adj_matrix = row_normalize(adj_matrix).A
    adj_matrix = adj_matrix.A
    U, La, Vh = np.linalg.svd(norm_adj_matrix)
    La_new = []
    index_new = []
    for i in range(len(La)):
        if La[i]>0.000000000001 or La[i]<-0.000000001:
            La_new.append(La[i])
            index_new.append(i)
    print(La_new)
    La_new = np.real(La_new)+0.00000001
    U_new = U[:, index_new]
    Vh_new = Vh[index_new, :]
    return U_new, La_new, Vh_new

def sort_np(La, node_num, appro_ratio):
    l = []
    for i in range(len(La)):
        l.append((La[i],i))
    large_La = []
    small_La = []
    l.sort(key=lambda x:x[0])
    for i in range(min(max(2000, int(node_num*appro_ratio)), int(node_num/2))):
        large_La.append(node_num - i -1)
        small_La.append(i)
    return np.array(small_La), np.array(large_La)

def compare(node_num, edge_index, power_num):
    adj_matrix = gen_adj_matrix(edge_index, node_num)
    norm_adj_matrix = row_normalize(adj_matrix)
    print(check_sym(norm_adj_matrix))
    ori_adj_matrix = copy.deepcopy(norm_adj_matrix)
    print(np.max(ori_adj_matrix))
    La, U = np.linalg.eigh(norm_adj_matrix)
    La = np.real(La)
    print(np.max(La), np.min(La))
    U = np.real(U)
    La = np.diag(La)
    print(np.max(np.dot(U.T, U)))
    print(np.min(np.dot(U.T, U)))
    recover_adj_matrix = np.dot(np.dot(U, La), U.T)
    print(np.max(recover_adj_matrix), np.min(norm_adj_matrix))
    print(np.max(recover_adj_matrix - norm_adj_matrix), np.min(recover_adj_matrix - norm_adj_matrix))

def heat_map(node_num, edge_index, v):
    adj_matrix = gen_adj_matrix(edge_index, node_num)
    norm_adj_matrix = row_normalize(adj_matrix).A
    max_degree_node, max_degree = cal_degree(norm_adj_matrix)
    La, U = np.linalg.eigh(norm_adj_matrix)
    La = np.real(La)+0.00000001
    La_v = np.power(La, v)
    aug_adj_matrix = np.dot(np.dot(U, np.diag(np.power(La, v))), U.T)
    print(np.max(aug_adj_matrix), np.min(aug_adj_matrix))
    print(np.max(norm_adj_matrix), np.min(norm_adj_matrix))
    print(np.max(aug_adj_matrix - norm_adj_matrix), np.min(aug_adj_matrix - norm_adj_matrix))
    print(max_degree_node, max_degree)
    """
    The last line is used to plot La figure
    """
    #return La, La_v
    return aug_adj_matrix, norm_adj_matrix

def cal_degree(adj_matrix):
    node_num = adj_matrix.shape[0]
    max_degree = 0
    max_degree_node = 0
    for i in range(node_num):
        degree = 0
        for j in range(node_num):
            if adj_matrix[i][j] > 0:
                degree += 1
        if degree > max_degree:
            max_degree_node = i
            max_degree = degree
    return max_degree_node, max_degree