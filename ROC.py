import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mx

def plot_roc(label,s,name='model', colorname='blue'):
    fpr, tpr, thresholds = mx.roc_curve(label[len(label)-len(s):], s, pos_label=1)
    plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, mx.auc(fpr, tpr)),color = colorname)
    plt.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
    plt.axis('square')
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right',fontsize=8)

anomaly_index = [66, 227, 354, 370, 444, 459, 603, 676, 746, 776, 1088, 1255, 1274, 1288, 1413, 1415, 1453, 1467, 1622, 1701, 1889, 1929, 2023, 2147, 2248, 2298, 2653, 3055, 3076, 3116, 3127, 3145, 3410, 3495, 3574, 3584, 4025, 4152, 4160, 4292, 4315, 4421, 4450, 4452, 4544, 4695, 4714, 4725, 4726, 4827, 4850, 4881, 4976, 5083, 5241, 5509, 5576, 6110, 6210, 6381, 6553, 6871, 7030, 7061, 7073, 7123]

# Isolation Forest
loss = np.load('/home/ypd-23-teacher-2/hzq/project/graphAEAD/iForestLoss.npy')
label = np.zeros(len(loss))
label[anomaly_index] = 1
plot_roc(label, loss,'Isolation Forest', 'green')
# Autoencoder
loss = np.load('/home/ypd-23-teacher-2/hzq/project/graphAEAD/AutoencoderLoss.npy')
label = np.zeros(len(loss))
label[anomaly_index] = 1
plot_roc(label, loss,'Autoencoder', 'red')
# LOF
loss = np.load('/home/ypd-23-teacher-2/hzq/project/graphAEAD/LOFLoss.npy')
label = np.zeros(len(loss))
label[anomaly_index] = 1
plot_roc(label, loss,'LOF', 'yellow')
# Dominant
label = np.load('./scores/DominantLabel.npy')
loss = np.load('./scores/DominantLoss.npy')
plot_roc(label, loss,'Dominant', 'cyan')
# DHetGraphAE
label = np.load('./scores/label.npy')
loss = np.load('./scores/DHetGraphAE12.npy')
plot_roc(label, loss,'DHetGraphAE', 'blue')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.title('The ROC curves')
plt.savefig('ROC.png')
plt.show()