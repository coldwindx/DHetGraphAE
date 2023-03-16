import argparse

import numpy as np
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)

import torch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cic2018', type=str)
    return parser.parse_args()


def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    print('异常阈值', threshold)
    auc_score = auc(fpr, tpr)
    print('auc值: {:.4f}'.format(auc_score))
    return threshold, auc_score


def eval(labels, pred):
    plot_roc(labels, pred)
    print(confusion_matrix(labels, pred))
    a, b, c, d = accuracy_score(labels, pred), precision_score(
        labels, pred), recall_score(labels, pred), f1_score(labels, pred)
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a, b, c, d))
    return a, b, c, d


def matrix(true_graph_labels, scores):
    t, auc = plot_roc(true_graph_labels, scores)
    true_graph_labels = np.array(true_graph_labels)
    scores = np.array(scores)
    pred = np.ones(len(scores))
    pred[scores < t] = 0
    print(confusion_matrix(true_graph_labels, pred))
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(true_graph_labels, pred), precision_score(
        true_graph_labels, pred), recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)))
    return auc, precision_score(true_graph_labels, pred), recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)


if __name__ == '__main__':
    labels = np.load('./scores/label.npy')
    scores = np.load('./scores/DHetGraphAE12.npy')

    pred = torch.zeros(len(scores))
    idx = scores.argsort()[::-1]  # 从大到小

    for k in [50, 200, 400, 600, 800, 1000]:
        print('============ k=', k)
        nidx = np.ascontiguousarray(idx[:k])
        pred[np.sort(nidx)] = 1  # 异常分数最高的K为样本判定为异常
        a, b, c, d = eval(labels.astype(np.longfloat), pred)
        print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a, b, c, d))
