import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch

# === 1. 加载数据 ===
df = pd.read_csv('merged_ctu13.csv', low_memory=False)
df.columns = df.columns.str.strip()  # 清除列名空格

# === 2. 选择训练字段 ===
features = ['Dur', 'Proto', 'Sport', 'Dport', 'TotBytes', 'TotPkts']
df = df[features + ['Label']].dropna()

# === 3. 协议类型编码 ===
df['Proto'] = LabelEncoder().fit_transform(df['Proto'].astype(str))

# === 4. Sport 和 Dport 转换为数值，非法设为 NaN ===
for port in ['Sport', 'Dport']:
    df[port] = pd.to_numeric(df[port], errors='coerce')
df.dropna(subset=['Sport', 'Dport'], inplace=True)

# === 5. 标签二值化（是否为 Botnet）===
df['Label'] = df['Label'].apply(lambda x: 1 if 'Botnet' in str(x) else 0)

# === 6. 提取特征并标准化前检查 ===
X = df[features].values
y = df['Label'].values

print("🔍 特征原始最大值:", np.max(X, axis=0))
print("🔍 特征原始最小值:", np.min(X, axis=0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ 特征标准化后最大值:", np.max(X_scaled, axis=0))
print("✅ 特征标准化后最小值:", np.min(X_scaled, axis=0))

# === 7. 数据集划分 ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# === 8. 标签分布打印 ===
print("🎯 训练集标签分布:", np.bincount(y_train))
print("🎯 测试集标签分布:", np.bincount(y_test))

# === 9. 转换为 PyTorch 张量 ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# === 10. 保存张量文件 ===
torch.save((X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor), 'cnn_data.pt')
print("✅ 数据预处理完成，已保存为 cnn_data.pt")
