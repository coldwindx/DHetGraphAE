# coding=UTF-8
import random
from gensim.models import FastText
import numpy as np
import re
from scipy import sparse

exec_feature_dim = 64
file_feature_dim = 64

exec_model = FastText.load('FastText_exec')
file_model = FastText.load('FastText_file')


def _make_mask(ids, num):
    mask = np.zeros(num)
    mask[ids] = 1
    mask = np.array(mask, dtype=np.bool)
    return mask


def get_fea(input_data, model, dim):
    input_data = re.split(r'[\\/=:,.\-\s]\s*', input_data)
    input_data = [item for item in input_data if item != '' and item != 'RANDOM']
    tmp = np.zeros(dim)
    count = 0
    # 对每个词的词嵌入进行平均
    for j in input_data:
        z = model.wv.get_vector(j)
        if np.isnan(z).sum() > 0:
            continue
        tmp += z
        count += 1
    tmp /= count
    return tmp


def cmd_chain_process(cmd_chain):
    cmd_chain = cmd_chain.split('}{')
    cmd_chain[0] = cmd_chain[0][1:]
    cmd_chain[-1] = cmd_chain[-1][:-1]
    for i in range(len(cmd_chain)):
        tmp = cmd_chain[i].split('>>>')
        cmd_chain[i] = tmp[0][3:] + '$%' + tmp[1]
    cmd_chain.reverse()
    return cmd_chain


exec_child = dict()
exec_child_all = dict()
cmd_chain_set = []
exec_set = list()
print("正在获取进程链集合")
count = 0
for line in open('/home/ypd-23-teacher-2/hzq/data/exec_for_test.txt'):
    field = line.strip('\n').split('$%')
    if len(field) == 5:
        cmd_chain_set.append(field[4])
    count += 1
    if count % 10000 == 0:
        print('进程已处理行数：', count)

print("正在对进程链去重，去重前总量为：", len(cmd_chain_set))
cmd_chain_set = list(set(cmd_chain_set))
if '' in cmd_chain_set:
    cmd_chain_set.remove('')
print("去重完毕，去重后进程链总量：", len(cmd_chain_set))
print("正在获取进程集合")

exec_count = 0
for i in range(len(cmd_chain_set)):
    cmd_chain = cmd_chain_process(cmd_chain_set[i])
    cmd_chain_set[i] = cmd_chain
    for i in cmd_chain:
        exec_set.append(i)
    exec_count += 1
    if exec_count % 10000 == 0:
        print("已从进程链获取进程进度：", str(exec_count) + '/' + str(len(cmd_chain_set)))

exec_set = list(set(exec_set))
print("训练集合计进程数：", len(exec_set))
exec_to_ID = {j: i for i, j in enumerate(exec_set)}
exec_feature = np.zeros((len(exec_set), exec_feature_dim))
print("已生成进程到数字的映射")
print("正在获取进程特征：")
for i in range(len(exec_set)):
    exec_feature[i] = get_fea(exec_set[i], model=exec_model, dim=exec_feature_dim)
    if i % 1000 == 0:
        print("已处理进程:", i)

count = 0
root_node = []
for cmd_chain in cmd_chain_set:
    # 遍历进程链，获取其中所有的进程派生关系
    root_node.append(exec_to_ID[cmd_chain[0]])
    if len(cmd_chain) < 2:
        continue
    for i in range(len(cmd_chain) - 1):
        exec_current = exec_to_ID[cmd_chain[i]]
        # 如果当前进程的父进程或者子进程存在，将其作为值加入字典
        if exec_current not in exec_child.keys():
            exec_child[exec_current] = list()
        if exec_current not in exec_child_all.keys():
            exec_child_all[exec_current] = list()
        exec_child[exec_current].append(exec_to_ID[cmd_chain[i + 1]])
        for j in range(i + 1, len(cmd_chain)):
            exec_child_all[exec_current].append(exec_to_ID[cmd_chain[j]])
    count += 1
    if count % 10000 == 0:
        print("已获取进程链数量", str(count) + '/' + str(len(cmd_chain_set)))

root_node = list(set(root_node))

for key in exec_child.keys():
    exec_child[key] = list(set(exec_child[key]))

for key in exec_child_all.keys():
    exec_child_all[key] = list(set(exec_child_all[key]))

# tree_size = [len(exec_child_all[i]) if i in exec_child_all else 1 for i in root_node]
# tmp = list(set(tree_size))
# tmp.sort()
# tree_size_distribute = [[i, tree_size.count(i)] for i in tmp]
row = []
col = []
for i in exec_child.keys():
    for j in exec_child[i]:
        row.append(i)
        col.append(j)
assert len(row) == len(col)
data = np.ones(len(row))
# 构建稀疏矩阵，形状为对应的实体数目，比如这里为（进程实体数， 套接字实体数）
matrix_exec = sparse.coo_matrix((data, (row, col)), shape=(len(exec_set), len(exec_set)))
matrix_exec = matrix_exec.tocsr()

file_set = list()
print("正在获取文件集合")
count = 0
for line in open('/home/ypd-23-teacher-2/hzq/data/file_for_test.txt'):
    field = line.strip('\n').split('$%')
    if len(field) == 4:
        file_set.append(field[2])
    count += 1
    if count % 10000 == 0:
        print('已处理行数：', count)
print("正在对文件集合去重，去重前总量为：", len(file_set))
file_set = list(set(file_set))
print("去重完毕，去重后文件总量：", len(file_set))

file_to_ID = {j: i for i, j in enumerate(file_set)}
file_feature = np.zeros((len(file_set), file_feature_dim))
for i in range(len(file_set)):
    file_feature[i] = get_fea(file_set[i], model=file_model, dim=file_feature_dim)
    if i % 1000 == 0:
        print("已处理进程:", i)

file_access = dict()

print("正在获取文件访问关系")
count = 0
count_exec = 0
for line in open('/home/ypd-23-teacher-2/hzq/data/file_for_test.txt'):
    field = line.strip('\n').split('$%')
    if len(field) == 4:
        cmd = field[0] + '$%' + field[1]
        if cmd in exec_to_ID:
            cmd = exec_to_ID[cmd]
        else:
            count_exec += 1
            continue
        file_path = file_to_ID[field[2]]
        if cmd not in file_access.keys():
            file_access[cmd] = list()
        file_access[cmd].append(file_path)
    count += 1
    if count % 10000 == 0:
        print("已获取文件行数", str(count), "目前字典总长度：", len(file_access))

for key in file_access.keys():
    file_access[key] = list(set(file_access[key]))

row = []
col = []
for i in file_access.keys():
    for j in file_access[i]:
        row.append(i)
        col.append(j)
assert len(row) == len(col)
data = np.ones(len(row))
matrix_file = sparse.coo_matrix((data, (row, col)), shape=(len(exec_set), len(file_set)))
matrix_file = matrix_file.tocsr()

idx = 0
label = []
normal_node = []
abnormal_node = []
for i in root_node:
    if exec_set[i].split('$%')[1] == '/usr/libexec/xpcproxy com.happy.to.you.ABC.RANDOM':
        abnormal_node.append(i)
    else:
        normal_node.append(i)
random.seed(777)
random.shuffle(normal_node)
random.seed(777)
random.shuffle(abnormal_node)
normal_half_len = len(normal_node)//2
abnormal_half_len = len(abnormal_node)//2
eval_set = normal_node[:normal_half_len] + abnormal_node[:abnormal_half_len]
label_eval = [0]*normal_half_len + [1]*abnormal_half_len
test_set = normal_node[normal_half_len:] + abnormal_node[abnormal_half_len:]
label_test = [0]*len(normal_node[normal_half_len:]) + [1]*len(abnormal_node[abnormal_half_len:])

for i in eval_set:
    if i not in exec_child_all:
        node_set = [i]
    else:
        node_set = [i] + exec_child_all[i]
        node_set = list(set(node_set))
        node_set.sort()
    mask = _make_mask(node_set, len(exec_set))
    exec_sub_adj = matrix_exec[mask]
    exec_sub_adj = exec_sub_adj.T[mask].T
    exec_sub_feature = exec_feature[mask]

    access_file_node = []
    for j in node_set:
        if j in file_access:
            access_file_node.extend(file_access[j])
    access_file_node = list(set(access_file_node))
    access_file_node.sort()
    if len(access_file_node) == 0:
        file_sub_adj = np.zeros((len(node_set), 1))
        file_sub_feature = np.zeros((1, file_feature_dim))
        file_sub_adj = sparse.csr_matrix(file_sub_adj)
    else:
        file_sub_adj = matrix_file[mask]
        file_mask = _make_mask(access_file_node, len(file_set))
        file_sub_adj = file_sub_adj.T[file_mask].T
        file_sub_feature = file_feature[file_mask]
    exec_sub_adj = np.where(exec_sub_adj.toarray() > 0, 1, 0)
    file_sub_adj = np.where(file_sub_adj.toarray() > 0, 1, 0)
    for j in range(exec_sub_adj.shape[0]):
        exec_sub_adj[j][j] = 0
    sparse.save_npz('eval_data/execAdj_' + str(idx), sparse.csr_matrix(exec_sub_adj))
    sparse.save_npz('eval_data/fileAdj_' + str(idx), sparse.csr_matrix(file_sub_adj))
    np.save('eval_data/execFeature_' + str(idx), exec_sub_feature)
    np.save('eval_data/fileFeature_' + str(idx), file_sub_feature)
    if (idx % 1000) == 0:
        print('已存储样本' + str(idx), flush=True)
    idx += 1

label_eval = np.array(label_eval)
np.save('eval_data/label', label_eval)

idx=0
for i in test_set:
    if i not in exec_child_all:
        node_set = [i]
    else:
        node_set = [i] + exec_child_all[i]
        node_set = list(set(node_set))
        node_set.sort()
    mask = _make_mask(node_set, len(exec_set))
    exec_sub_adj = matrix_exec[mask]
    exec_sub_adj = exec_sub_adj.T[mask].T
    exec_sub_feature = exec_feature[mask]

    access_file_node = []
    for j in node_set:
        if j in file_access:
            access_file_node.extend(file_access[j])
    access_file_node = list(set(access_file_node))
    access_file_node.sort()
    if len(access_file_node) == 0:
        file_sub_adj = np.zeros((len(node_set), 1))
        file_sub_feature = np.zeros((1, file_feature_dim))
        file_sub_adj = sparse.csr_matrix(file_sub_adj)
    else:
        file_sub_adj = matrix_file[mask]
        file_mask = _make_mask(access_file_node, len(file_set))
        file_sub_adj = file_sub_adj.T[file_mask].T
        file_sub_feature = file_feature[file_mask]
    exec_sub_adj = np.where(exec_sub_adj.toarray() > 0, 1, 0)
    file_sub_adj = np.where(file_sub_adj.toarray() > 0, 1, 0)
    for j in range(exec_sub_adj.shape[0]):
        exec_sub_adj[j][j] = 0
    sparse.save_npz('test_data/execAdj_' + str(idx), sparse.csr_matrix(exec_sub_adj))
    sparse.save_npz('test_data/fileAdj_' + str(idx), sparse.csr_matrix(file_sub_adj))
    np.save('test_data/execFeature_' + str(idx), exec_sub_feature)
    np.save('test_data/fileFeature_' + str(idx), file_sub_feature)
    if (idx % 1000) == 0:
        print('已存储样本' + str(idx), flush=True)
    idx += 1

label_test = np.array(label_test)
np.save('test_data/label', label_test)
