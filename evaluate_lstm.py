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

# === LSTM 模型结构（与训练中一致） ===
class BotnetLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        h_last = self.bn(h_last)
        h_last = self.dropout(h_last)
        return self.fc(h_last)

# === 加载数据与模型 ===
X_train, y_train, X_test, y_test = torch.load('cnn_data1.pt')
X_test = X_test.squeeze(1).unsqueeze(-1)  # (B, 6) → (B, 6, 1)

model = BotnetLSTM(input_dim=1)
model.load_state_dict(torch.load('model/best_lstm_model1.pth'))
model.eval()

# === 执行预测 ===
y_true, y_pred, y_prob = [], [], []
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=1024)

with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb)
        prob = F.softmax(out, dim=1)
        _, pred = torch.max(prob, dim=1)
        y_true.extend(yb.numpy())
        y_pred.extend(pred.numpy())
        y_prob.extend(prob[:, 1].numpy())

# === 输出分类指标 ===
print("LSTM 输出分类指标:")
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
