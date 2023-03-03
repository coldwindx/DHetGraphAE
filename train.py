import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.sparse as sp
from load_data import load_train_data
from model_diff_lay import HetGraphAE2 as model

# from model.model import Encoder

data_num = 323538
loss_record = []
model_name = 'HetGraphAE2'
big_graph = [187949,268769]
big_graph = set(big_graph)

exec_feature_dim = 64
file_feature_dim = 64
in_feature_dim = 64
gcn_hid = 64
nn_hid1 = 48
device = torch.device('cuda:0')

class Train:

    def __init__(self, alpha=0.5, learn_rate=0.001, weight_decay=0.001):
        # 获取三个邻接表以及进程的原始特征矩阵
        self.encoder = model(exec_feature_dim, file_feature_dim, in_feature_dim, gcn_hid, nn_hid1)
        self.encoder.train()
        self.alpha = alpha
        if torch.cuda.is_available():
            self.encoder = self.encoder.to(device)
        self.loss_feature = torch.nn.MSELoss(reduction='mean')
        self.loss_adj = torch.nn.BCELoss()
        self.optimizer_encoder = optim.Adam(
            [{'params': self.encoder.parameters(), 'lr': learn_rate}],
            weight_decay=weight_decay)

    def batch_loss(self, i, is_cuda=True):
        exec_adj, file_adj, exec_feature, file_feature = load_train_data(i, is_cuda,device)
        re_exec_adj, re_file_adj, re_exec_feature = self.encoder(exec_feature, file_feature, exec_adj, file_adj)
        raw_exec_adj = exec_adj + exec_adj.t()
        mask = torch.ones((exec_adj.shape[0], exec_adj.shape[1])) - torch.eye(exec_adj.shape[0])
        if is_cuda:
            re_exec_adj = re_exec_adj.cpu()
            re_exec_adj_ = re_exec_adj * mask
            re_exec_adj_ = re_exec_adj_.cuda()
        else:
            re_exec_adj_ = re_exec_adj * mask
        loss_adj_exec = self.loss_adj(re_exec_adj_, raw_exec_adj)
        # print(re_exec_adj_)
        # print(raw_exec_adj)
        loss_adj_file = self.loss_adj(re_file_adj, file_adj)
        loss_feature = self.loss_feature(re_exec_feature, exec_feature)
        # print('loss_adj_exec', loss_adj_exec)
        # print('loss_adj_file', loss_adj_file)
        # print('loss_feature', loss_feature * 100)
        return loss_adj_exec + loss_adj_file + loss_feature * 50

    def train(self, t=1000):
        tree_list = list(range(data_num))
        random.seed(666)
        random.shuffle(tree_list)
        try:
            count = 0
            for times in range(t):
                for i in range(268700, len(tree_list)):
                    count += 1
                    print(i)
                    if i in {18822, 28995, 36918, 61530, 73826, 252977, 275151, 287485, 288875}:
                        continue
                    if i in big_graph:
                        is_cuda = False
                        self.encoder = self.encoder.cpu()
                    else:
                        is_cuda = True
                    loss = self.batch_loss(tree_list[i], is_cuda)
                    self.optimizer_encoder.zero_grad()
                    loss.backward()
                    if not is_cuda:
                        paras = list(self.optimizer_encoder.state.values())
                        for para in paras:
                            para['exp_avg'] = para['exp_avg'].cpu()
                            para['exp_avg_sq'] = para['exp_avg_sq'].cpu()
                    self.optimizer_encoder.step()
                    if i % 100 == 0:
                        print("第%s次迭代, 第%s个batch loss:" % (times, i), flush=True)
                        print(loss, flush=True)
                    # if count == 1000:
                    #     torch.save(self.encoder, 'save_model/' + model_name + str(times))
                    #     count = 0
                    del loss
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available() and not is_cuda:
                        device = torch.device('cuda:0')
                        self.encoder = self.encoder.to(device)
                        paras = list(self.optimizer_encoder.state.values())
                        for para in paras:
                            para['exp_avg'] = para['exp_avg'].cuda()
                            para['exp_avg_sq'] = para['exp_avg_sq'].cuda()
                torch.save(self.encoder, 'save_model/' + model_name + str(times))
        except KeyboardInterrupt or MemoryError or RuntimeError:
            torch.save(self.encoder, 'save_model/' + model_name)
        return


SEED = 666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
train_ = Train(alpha=1.3, learn_rate=0.001, weight_decay=0.00)
train_.train(t=20)
