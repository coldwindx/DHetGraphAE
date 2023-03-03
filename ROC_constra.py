import matplotlib.pyplot as plt
import numpy as np

# 进程记录中实际的恶意进程
anomy = ['/bin/bash -c ls',
         '/bin/bash -c ls -al /tmp/',
         '/bin/bash -c pwd',
         '/bin/bash -c rm -rf /tmp/ABCDE',
         '/bin/bash -c rm /tmp/ABCDE',
         '/bin/bash -c sosocks5serve add --rhost=127.0.0.1 --rport=8000 --username=abc --password=123 --lhost=0.0.0.0 --lport=8000',
         '/bin/bash -c whoami',
         '/bin/ls',
         '/bin/ls -al /tmp/',
         '/bin/ps -ax -o pid',
         '/bin/rm -rf /tmp/ABCDE',
         '/bin/rm /tmp/ABCDE',
         '/bin/sh -c /tmp/ABCDE',
         '/net/121.40.126.250/mnt/ABC/DEF/ABC.app/Contents/MacOS/ABC',
         '/private/tmp/ABCDE',
         '/sbin/mount -o nobrowse -t nfs -o retrycnt=0 -o nosuid,nodev -o nosuid -o automounted -o nosuid 121.40.126.250:/mnt/ABC/DEF /net/121.40.126.250/mnt/ABC/DEF',
         '/sbin/mount_nfs -o nobrowse -o retrycnt=0 -o nosuid -o nodev -o nosuid -o automounted -o nosuid 121.40.126.250:/mnt/ABC/DEF /net/121.40.126.250/mnt/ABC/DEF',
         '/usr/bin/sw_vers -productVersion',
         '/usr/bin/whoami',
         '/usr/libexec/xpcproxy com.happy.to.you.ABC.4128']

cmd_dict = dict()
# 所有的未归一化cmd
cmd_set = set()
for line in open('/home/ypd-23-teacher-2/hzq/data/exec_for_test.txt'):
    field = line.split('$%')
    # 未归一化cmd
    if len(field) == 5:
        cmd = field[1]
        # pid$%归一化cmd
        cmd_std = field[0] + '$%' + field[2]
        # 构建映射
        cmd_dict[cmd_std] = cmd
        cmd_set.add(cmd)

# 加载所有待测试进程
fp = open('/home/ypd-23-teacher-2/hzq/project/graphAEAD/testData/test_id2exec.txt')
id_new = eval(fp.read())
fp.close()
# fp = open('/mnt/datasets/hzqdata/graphAEAD/testData/test_id2exec_exec.txt')
# id_old = eval(fp.read())
# fp.close()
id_new = list(id_new)
# id_old = list(id_old)
# model_list = ['HetGraphAE_1layer', 'HetGraphAE_2layer', 'HetGraphAE_3layer', 'HetGraphAE_4layer', 'HetGraphAE_5layer']
model_list = ['iForest', 'Autoencoder', 'Dominant']
# model_list = ['iForest', 'Autoencoder', 'Dominant-exec']
# model_list = ['HetGraphAE', 'dominant-exec', 'LOF', 'iForest']
# model_list = ['iForest', 'Autoencoder', 'LOF', 'hetGraphAE_3layer', 'hetGraphAE_4layer', 'NoAdj']
# color_list = ['g', 'r', 'b', 'y']
color_list = ['g', 'r', 'b', 'y', 'c', 'm', 'orange', 'peru']
# model_list = ['LOF', 'iForest', 'Autoencoder', 'graphAEProcess', 'graphAEFile', 'hetGraphAE']
AUC = []
# for k in range(2):
for k in range(len(model_list)):
    loss = np.load('/home/ypd-23-teacher-2/hzq/project/graphAEAD/' + model_list[k] + 'Loss.npy')
    # loss = np.load('HetGraphAELoss0.npy')


    index = []
    # if k < 0:
    #     id = id_old
    # else:
    #     id = id_new
    id = id_new
    for i in range(len(id)):
        if id[i] in cmd_dict.keys() and cmd_dict[id[i]] in anomy:
            index.append(i)

    # 异常进程的异常分数集
    loss_anomy = loss[index]
    loss_anomy.sort()
    cmd_loss = dict()
    for i in range(len(loss)):
        if id[i] == '$%$%$%':
            continue
        if id[i] not in cmd_loss.keys():
            cmd_loss[id[i]] = set()
        # id与loss是一一对应的关系
        cmd_loss[id[i]].add(loss[i])

    for key in cmd_loss.keys():
        if len(cmd_loss[key]) != 0:
            cmd_loss[key] = max(cmd_loss[key])
        else:
            print(1)

    p_sum = len(loss_anomy)
    n_sum = len(id) - len(loss_anomy)


    def f(idx):
        a = idx
        p = 0
        # 模型检出的所有异常
        tp = 0
        for key in cmd_loss.keys():
            if key in cmd_dict.keys() and cmd_dict[key] in anomy:
                # 如果恶意进程的异常分数大于等于阈值则被抛出
                if cmd_loss[key] >= a:
                    tp += 1
            # 进程的异常分数大于等于阈值则被抛出
            if cmd_loss[key] >= a:
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
        # if fpr not in f_to_t.keys():
        #     f_to_t[fpr] = tpr
        # elif f_to_t[fpr] < tpr:
        #     f_to_t[fpr] = tpr
        TPR.append(tpr)
        FPR.append(fpr)
    # FPR = list(f_to_t.keys())
    # FPR.sort()
    # TPR = [f_to_t[i] for i in FPR]
    FPR.append(0)
    TPR.append(0)
    TPR.reverse()
    FPR.reverse()
    auc = 0
    pre_x = 0
    for i in range(0, len(FPR)):
        if FPR[i] != pre_x:
            auc += (FPR[i] - pre_x) * TPR[i]
            pre_x = FPR[i]
    AUC.append(auc)
    plt.plot(FPR, TPR, color_list[k], label=model_list[k]+'=%.3f'%auc)

# 显示图片
label = np.load('test_anomaly_score/label.npy')
p_sum = label.sum()
n_sum = len(label) - p_sum
model = 'HetGraphAE212'
loss = np.load('test_anomaly_score/' + model + '.npy')


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
auc = 0
pre_x = 0
for i in range(0, len(FPR)):
    if FPR[i] != pre_x:
        auc += (FPR[i] - pre_x) * TPR[i]
        pre_x = FPR[i]
AUC.append(auc)
plt.plot(FPR, TPR, color_list[len(model_list)], label='DHetGraphAE' + '=%.3f' % auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.title('The ROC curves')
plt.show()
