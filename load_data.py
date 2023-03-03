import numpy as np
import torch
import scipy.sparse as sp


def load_train_data(i, is_cuda, device):
    exec_adj = sp.load_npz('train_data/execAdj_' + str(i) + '.npz')
    file_adj = sp.load_npz('train_data/fileAdj_' + str(i) + '.npz')
    exec_adj = exec_adj.toarray()
    file_adj = file_adj.toarray()
    exec_feature = np.load('train_data/execFeature_' + str(i) + '.npy')
    file_feature = np.load('train_data/fileFeature_' + str(i) + '.npy')
    exec_adj = torch.tensor(exec_adj).float()
    file_adj = torch.tensor(file_adj).float()
    exec_feature = torch.tensor(exec_feature).float()
    file_feature = torch.tensor(file_feature).float()
    exec_feature = torch.nn.functional.normalize(exec_feature, p=2, dim=1)
    file_feature = torch.nn.functional.normalize(file_feature, p=2, dim=1)
    if is_cuda:
        exec_adj = exec_adj.to(device)
        file_adj = file_adj.to(device)
        exec_feature = exec_feature.to(device)
        file_feature = file_feature.to(device)
    return exec_adj, file_adj, exec_feature, file_feature
