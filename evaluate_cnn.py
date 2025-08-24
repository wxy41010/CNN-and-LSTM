import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# === 可视化混淆矩阵类 ===
class PlotConfusionMatrix:
    def plot_confusion_matrix(self, labels, cm, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90)
        plt.yticks(tick_marks, labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    def prepare_work(self, labels, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        plt.figure(figsize=(12, 8), dpi=120)
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = cm_normalized[i][j]
                if value > 0.01:
                    plt.text(j, i, f"{value:.2f}", color='red', ha='center', va='center', fontsize=8)

        plt.gca().set_xticks(np.arange(len(labels)) + 0.5, minor=True)
        plt.gca().set_yticks(np.arange(len(labels)) + 0.5, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.2)

        self.plot_confusion_matrix(labels, cm_normalized, title='Normalized Confusion Matrix')
        plt.show()

def plotMatrix(attacks, y_true, y_pred):
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    label_names = [attacks[i] for i in unique_labels]
    p = PlotConfusionMatrix()
    p.prepare_work(label_names, y_true, y_pred)

# === 模型结构 ===
class BotnetCNN(torch.nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, 3)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear((input_len - 2) * 32, 128)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# === 加载数据与模型 ===
X_train, y_train, X_test, y_test = torch.load('cnn_data.pt')
model = BotnetCNN(input_len=X_test.shape[2])
model.load_state_dict(torch.load('model/best_cnn_model1.pth'))
model.eval()

# === 执行预测 ===
y_true, y_pred, y_prob = [], [], []
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=256)

with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb)
        prob = F.softmax(out, dim=1)
        _, pred = torch.max(prob, dim=1)
        y_true.extend(yb.numpy())
        y_pred.extend(pred.numpy())
        y_prob.extend(prob[:, 1].numpy())

# === 输出分类指标 ===
print("输出分类指标:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Botnet']))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n性能指标:")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1-Score   : {f1:.4f}")

try:
    roc_auc = roc_auc_score(y_true, y_prob)
    print("ROC-AUC Score: {:.4f}".format(roc_auc))
except ValueError:
    print("⚠️ ROC-AUC 无法计算（可能测试集中只有一类）")

# === 混淆矩阵可视化 ===
attacks = {0: 'Normal', 1: 'Botnet'}
plotMatrix(attacks, np.array(y_true), np.array(y_pred))
