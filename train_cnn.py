import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# === Step 1: 加载数据 ===
X_train, y_train, X_test, y_test = torch.load('cnn_data.pt')
batch_size = 8192

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size,  pin_memory=True)

# === Step 2: 定义 CNN 模型 ===
class BotnetCNN(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear((input_len - 2) * 32, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# === Step 3: 初始化训练设置 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BotnetCNN(input_len=X_train.shape[2]).to(device)

# 自动加权处理类别不平衡
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 混合精度工具
amp_scaler = torch.amp.GradScaler(device='cuda')

if not os.path.exists("model"):
    os.makedirs("model")

# === Step 4: 训练循环 ===
best_loss = float("inf")
patience = 5
counter = 0
loss_list = []
acc_list = []

print(f"🖥️ 使用设备: {device}")
print(f"显存占用前: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

for epoch in range(30):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            out = model(xb)
            loss = criterion(out, yb)

        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()


        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    epoch_loss = total_loss
    epoch_acc = correct / total
    loss_list.append(epoch_loss)
    acc_list.append(epoch_acc)

    print(f"📘 Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {time.time() - start_time:.2f}s")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), "model/best_cnn_model1.pth")
        print("  ✅ 模型已保存：best_cnn_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("  🛑 Early stopping triggered.")
            break

print(f"显存占用后: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# === Step 5: 绘图保存 ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list, label="Accuracy", color='green')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("loss_accuracy.png")
plt.show()
print("📈 训练曲线图已保存为 loss_accuracy.png")
