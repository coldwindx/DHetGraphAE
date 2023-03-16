from GCNlayer import GraphConvolution
import torch.nn as nn
import torch

class DHetGraphAE(nn.Module):
    def __init__(self, exec_feature_dim, file_feature_dim, in_feature_dim, gcn_hid, nn_hid1):
        super(DHetGraphAE, self).__init__()

        self.Encoder_lay1 = GraphConvolution(exec_feature_dim, file_feature_dim, in_feature_dim, gcn_hid)
        self.Encoder_lay2 = GraphConvolution(gcn_hid, file_feature_dim, gcn_hid, nn_hid1)
        self.Encoder_lay3 = GraphConvolution(nn_hid1, file_feature_dim, nn_hid1, gcn_hid)
        self.Encoder_lay4 = GraphConvolution(gcn_hid, file_feature_dim, gcn_hid, exec_feature_dim)
        self.linear = nn.Linear(exec_feature_dim, exec_feature_dim)
        self.file_tran = nn.Linear(file_feature_dim, nn_hid1)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, exec_input, file_input, exec_adj, file_adj):
        # 返回的是一个b*dim的向量，每行表示图的表示
        gcn_local_hid_1 = self.act(self.Encoder_lay1(exec_input, file_input, exec_adj, file_adj))
        gcn_local_hid_2 = self.act(self.Encoder_lay2(gcn_local_hid_1, file_input, exec_adj, file_adj))
        file_tran = self.file_tran(file_input)
        re_exec_adj = torch.sigmoid(gcn_local_hid_2 @ gcn_local_hid_2.t())
        re_file_adj = torch.sigmoid(gcn_local_hid_2 @ file_tran.t())

        gcn_local_hid_3 = self.act(self.Encoder_lay3(gcn_local_hid_2, file_input, exec_adj, file_adj))
        gcn_local_hid_4 = self.act(self.Encoder_lay4(gcn_local_hid_3, file_input, exec_adj, file_adj))
        re_exec_feature = self.linear(gcn_local_hid_4)
        return re_exec_adj, re_file_adj, re_exec_feature

