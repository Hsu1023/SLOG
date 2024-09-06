from train import train
from data_preprocessing import download_data
import config
import os
import numpy as np
import torch
import random
from gp import train_gp
from am import train_am
test_res_list = []
best_res_list = []
test_v = []
best_res_v = []

print('inductive' if config.inductive else 'transductive')
torch.cuda.set_device(config.gpu)

print("config: lr: {}, weight_decay: {}, hidden_dim: {}, batch_size: {}, sample_num: {}, ratio: {}/{}/{}, res: {}, layer_num: {}".format(config.lr, config.weight_decay, config.hidden_dim, config.batch_size, config.sample_num, config.training_ratio, config.valid_ratio, 1-config.training_ratio-config.valid_ratio, config.res, config.layer_num))
if config.dataset in ["squirrel_filtered", "chameleon_filtered", "amazon_ratings", "minesweeper", "tolokers", "roman_empire", "questions"]:
    download_data(config.dataset)
    for i in range(5):
        if config.optimization == 'gp':
            tmp_res = train_gp(config.resplit, cur_data_split=i)
        elif config.optimization == 'am':
            tmp_res = train_am(config.resplit, cur_data_split=i)
        else:
            tmp_res = train(config.resplit, cur_data_split=i)
        test_res_list.append(tmp_res[0])
        best_res_list.append(tmp_res[1])
        print(test_res_list)
else:
    for i in range(5):
        download_data(config.dataset)
        if config.optimization == 'gp':
            tmp_res = train_gp(config.resplit)
        elif config.optimization == 'am':
            tmp_res = train_am(config.resplit)
        else:
            tmp_res = train(config.resplit)
        test_res_list.append(tmp_res[0])
        best_res_list.append(tmp_res[1])
        print(test_res_list)
print(config.dataset)
print("Test_rest: Mean: {:.3f}, Std: {:.3f}".format(np.array(test_res_list).mean(axis=0),
                            np.array(test_res_list).std(axis=0)))
print("Best_rest: Mean: {:.3f}, Std: {:.3f}".format(np.array(best_res_list).mean(axis=0),
                            np.array(best_res_list).std(axis=0)))

print("config: lr: {}, weight_decay: {}, hidden_dim: {}, batch_size: {}, sample_num: {}, ratio: {}/{}/{}, res: {}, layer_num: {}".format(config.lr, config.weight_decay, config.hidden_dim, config.batch_size, config.sample_num, config.training_ratio, config.valid_ratio, 1-config.training_ratio-config.valid_ratio, config.res, config.layer_num))