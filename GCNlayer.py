import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, exec_feature_dim, file_feature_dim, in_feature_dim, out_feature_dim):
        super(GraphConvolution, self).__init__()
        self.exec_feature_dim = exec_feature_dim
        self.file_feature_dim = file_feature_dim
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.dim_trans_exec = nn.Linear(exec_feature_dim, in_feature_dim)
        self.dim_trans_file = nn.Linear(file_feature_dim, in_feature_dim)
        self.linear = nn.Linear(self.in_feature_dim * 4, self.out_feature_dim)

    def forward(self, exec_input, file_input, exec_adj, file_adj):
        exec_input = self.dim_trans_exec(exec_input)
        exec_down = torch.mm(exec_adj, exec_input)
        exec_down = exec_down / (exec_adj.sum(dim=1).unsqueeze(dim=1).add(1e-20))
        exec_adj_converse = exec_adj.t()
        exec_up = torch.mm(exec_adj_converse, exec_input)
        exec_up = exec_up / (exec_adj_converse.sum(dim=1).unsqueeze(dim=1).add(1e-20))

        file_input = self.dim_trans_file(file_input)
        file_down = torch.mm(file_adj, file_input)
        file_down = file_down / (file_adj.sum(dim=1).unsqueeze(dim=1).add(1e-20))
        combine = torch.cat((exec_up, exec_input, exec_down, file_down), dim=1)
        output = self.linear(combine)
        return output

    def adj_process(self, adj):
        if torch.cuda.is_available():
            adj = adj + torch.eye(adj.shape[0]).float().cuda()
        else:
            adj = adj + torch.eye(adj.shape[0]).float()
        d = adj.sum(dim=1)
        D = torch.diag(torch.pow(d, -0.5))
        adj_lap = D.mm(adj).mm(D)
        return adj_lap

