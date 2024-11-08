
import torch
import numpy as np
from torchvision import transforms, datasets
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, classification_report, confusion_matrix, \
    roc_auc_score, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize, StandardScaler
import matplotlib.pyplot as plt

# from CTAFFNet import MainModel as create_model
# from ODstarnet import starnet_s4 as create_model
# from CTAFFnoAFF import MainModel as ce_model
from starnet import StarNet as create_model

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, auc)
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import cycle
from matplotlib.colors import ListedColormap
import timm

def get_image_path(dataset, index):
    return dataset.samples[index][0]

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, figsize=(10, 7), annot_fontsize=18, title_fontsize=20, label_fontsize=16):
    my_cmap = ListedColormap(
        ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'])

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=my_cmap, xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': ''}, linewidths=.5, square=True, annot_kws={"size": annot_fontsize})

    plt.xlabel('Predicted Label', fontsize=label_fontsize)
    plt.ylabel('True Label', fontsize=label_fontsize)
    plt.title('Model1', fontsize=title_fontsize)
    plt.tick_params(labelsize=label_fontsize)  # 设置刻度标签的字体大小

    plt.show()
# def plot_confusion_matrix(cm, class_names):
#     # 定义一个柔和的绿色渐变颜色方案
#     my_cmap = ListedColormap(
#         ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'])
#
#     # 设置图形大小
#     plt.figure(figsize=(10, 7))
#
#     # 绘制热图
#     sns.heatmap(cm, annot=True, fmt='d', cmap=my_cmap, xticklabels=class_names, yticklabels=class_names,
#                 cbar_kws={'label': 'Frequency'}, linewidths=.5, square=True, annot_kws={"size": 12})
#
#     # 设置背景色为柔和的白色
#     plt.rcParams['axes.facecolor'] = '#f7fcf5'
#
#     # 设置x轴和y轴标签
#     plt.xlabel('Predicted Label', fontsize=18, color='#333333')  # 暗灰色字体
#     plt.ylabel('True Label', fontsize=18, color='#333333')  # 暗灰色字体
#
#     # 设置标题
#     plt.title('Confusion Matrix', fontsize=18, color='#333333')  # 暗灰色字体
#
#     # 调整字体和标题样式
#     plt.tick_params(labelsize=18, colors='#333333')  # 暗灰色刻度标签
#
#     # 显示图形
#     plt.show()


def plot_clustermap(cm, class_names):
    # 标准化数据
    scaler = StandardScaler()
    cm_normalized = scaler.fit_transform(cm)

    sns.clustermap(cm_normalized, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=class_names,
                   yticklabels=class_names)
    plt.title('Cluster Map of Confusion Matrix')
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, num_classes):
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'orange'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC curves')
    plt.legend(loc="lower right")
    plt.show()


def plot_pr_curve(precision, recall, pr_auc, num_classes):
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'yellow', 'cyan', 'magenta', 'orange'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR curve of class {0} (area = {1:0.2f})'.format(i, pr_auc[i]))
    plt.plot([0, 1], [1, 0], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall curves')
    plt.legend(loc="lower left")
    plt.show()


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Compute the Quadratic Weighted Kappa (QWK)
    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :return: QWK score
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')
def plot_scatter(features, labels, class_names):
    plt.figure(figsize=(12, 8))
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    markers = cycle(['o', 's', 'D', '^', 'v', 'p', '*', 'X', 'H', '8'])  # 添加不同的标记符号
    for label, color, marker in zip(range(len(class_names)), colors, markers):
        plt.scatter(features[labels == label, 0], features[labels == label, 1], color=color, label=class_names[label], alpha=0.6, marker=marker, edgecolors='w', linewidth=0.5)
    plt.legend()
    plt.title('APTOS 2019')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    num_classes = 4  # 修改为您的类别数
    img_size = 224
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root='H:/grade2023/ryy/datasets/funds/aug/newtest', transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = create_model(num_classes=num_classes).to(device)
    model_weight_path = "./model_weight/RCotbest_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)
    model.eval()

    # writer = SummaryWriter(log_dir='./runs/tensorboard_logs')

    all_preds = []
    all_labels = []
    all_prob = []

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 获取第一个元素
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            # outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # probs = torch.softmax(outputs, dim=1)

            all_prob.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for idx, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    image_path = get_image_path(test_dataset, batch_index * test_loader.batch_size + idx)
                    print(f'Misclassified image: {image_path}, Prediction: {pred.item()}, Actual: {label.item()}')

    all_prob = np.concatenate(all_prob)
    all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    qwk = quadratic_weighted_kappa(all_labels, all_preds)
    # Calculate sensitivity and specificity from confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn = np.diag(cm).sum() - cm.sum(axis=1)  # True Negatives
    tp = np.diag(cm)  # True Positives
    fn = cm.sum(axis=1) - tp  # False Negatives
    fp = cm.sum(axis=0) - tp  # False Positives

    # Sensitivity (Recall) and Specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Average sensitivity and specificity
    average_sensitivity = np.mean(sensitivity)
    average_specificity = np.mean(specificity)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, test_dataset.classes)
    plot_clustermap(cm, test_dataset.classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision_dict = dict()
    recall_dict = dict()
    pr_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(all_labels_bin[:, i], all_prob[:, i])
        pr_auc[i] = auc(recall_dict[i], precision_dict[i])

    plot_roc_curve(fpr, tpr, roc_auc, num_classes)
    plot_pr_curve(precision_dict, recall_dict, pr_auc, num_classes)

    average_auc = np.mean(list(roc_auc.values()))
    average_pr_auc = np.mean(list(pr_auc.values()))

    print(
        f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Sensitivity: {average_sensitivity}, Specificity: {average_specificity}, Kappa: {kappa}, MCC: {mcc}, QWK: {qwk}, AUC (Average): {average_auc:.4f}, PR AUC (Average): {average_pr_auc}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))


if __name__ == '__main__':
    main()
