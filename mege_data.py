import pandas as pd
import os

# === 配置路径 ===
base_path = 'CTU-13-Dataset'     # 根目录名
output_csv = 'merged_ctu13.csv'  # 输出合并文件名

# === 合并操作 ===
all_data = []
subdirs = [str(i) for i in range(1, 14)]  # 目录 1~13

for subdir in subdirs:
    folder_path = os.path.join(base_path, subdir)
    if not os.path.isdir(folder_path):
        print(f"❌ 跳过不存在目录：{folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith('.binetflow'):
            file_path = os.path.join(folder_path, file)
            try:
                print(f"📥 正在读取：{file_path}")
                df = pd.read_csv(file_path, low_memory=False)  # 不跳过 header！
                
                # 仅第一次设置列名，以防部分 .binetflow 文件缺列
                if 'Label' not in df.columns and len(df.columns) >= 15:
                    df.columns = [
                        "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr", "Dport",
                        "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label"
                    ] + [f"extra{i}" for i in range(len(df.columns) - 15)]

                df['source_file'] = f"{subdir}/{file}"
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ 读取失败：{file_path}，错误：{e}")

# 合并并导出
if all_data:
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"\n✅ 合并完成，保存为 {output_csv}，总计记录数：{len(merged_df)}，列数：{len(merged_df.columns)}")
else:
    print("❌ 没有成功读取任何文件。")
