import torch
import scipy.sparse as sp
import numpy as np


def getAUC(loss, label):
    p_sum = sum(label)
    n_sum = len(label) - p_sum

    def f(thres):
        p = 0
        tp = 0
        for i in range(len(loss)):
            if label[i] == 1:
                if loss[i] >= thres:
                    tp += 1
            if loss[i] >= thres:
                p += 1
        return tp / p_sum, (p - tp) / n_sum

    TPR = []
    FPR = []
    i = 0
    f_to_t = dict()
    loss_set = list(set(loss))
    loss_set.sort()
    for i in range(len(loss_set)):
        tpr, fpr = f(loss_set[i])
        TPR.append(tpr)
        FPR.append(fpr)
    FPR.append(0)
    TPR.append(0)
    TPR.reverse()
    FPR.reverse()
    AUC = 0
    pre_x = 0
    for i in range(0, len(FPR)):
        if FPR[i] != pre_x:
            AUC += (FPR[i] - pre_x) * TPR[i]
            pre_x = FPR[i]
    print("AUC:", AUC)

model_name = 'HetGraphAE212'
data_num = 2668
label = np.load('test_data/label.npy')
model = torch.load('save_model/' + model_name)
model = model.cuda()
model = model.eval()
anomaly_score = []
label_new = []
loss_fn = torch.nn.MSELoss(reduction='none')
with torch.no_grad():
    for i in range(data_num):
        exec_adj = sp.load_npz('test_data/execAdj_' + str(i) + '.npz')
        file_adj = sp.load_npz('test_data/fileAdj_' + str(i) + '.npz')
        exec_adj = exec_adj.toarray()
        file_adj = file_adj.toarray()
        exec_feature = np.load('test_data/execFeature_' + str(i) + '.npy')
        file_feature = np.load('test_data/fileFeature_' + str(i) + '.npy')
        exec_adj = torch.tensor(exec_adj).float().cuda()
        file_adj = torch.tensor(file_adj).float().cuda()
        exec_feature = torch.tensor(exec_feature).float().cuda()
        file_feature = torch.tensor(file_feature).float().cuda()
        re_exec_adj, re_file_adj, re_exec_feature = model(exec_feature, file_feature, exec_adj, file_adj)
        loss = loss_fn(re_exec_feature, exec_feature)

        loss = loss.mean(dim=1)
        anomaly_score.extend(loss.cpu().tolist())
        label_new.extend([label[i]] * len(loss))
anomaly_score = np.array(anomaly_score)
np.save('test_anomaly_score/' + model_name, anomaly_score)
np.save('test_anomaly_score/' + 'label', label_new)
getAUC(anomaly_score, label_new)
