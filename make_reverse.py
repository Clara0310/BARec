import os

# 設定你的資料集名稱
DATASET = "Grocery_and_Gourmet_Food"
DATA_DIR = f"./data/{DATASET}"

# 定義要轉換的檔案對應關係
files = ['train', 'valid', 'test']

def reverse_interactions(input_path, output_path):
    print(f"正在轉換: {input_path} -> {output_path}")
    data = {}
    
    # 1. 讀取原始資料
    if not os.path.exists(input_path):
        print(f"警告: 找不到 {input_path}，跳過。")
        return

    with open(input_path, 'r') as f:
        for line in f:
            u, i, t = line.strip().split()
            if u not in data: data[u] = []
            data[u].append((i, float(t)))
    
    # 2. 進行倒轉 (依照時間由大到小排序)
    with open(output_path, 'w') as f:
        for u, interactions in data.items():
            # 關鍵：reverse=True 代表時間越新的排越前面
            rev_interactions = sorted(interactions, key=lambda x: x[1], reverse=True)
            for item, time in rev_interactions:
                f.write(f"{u}\t{item}\t{time}\n")

# 執行轉換
for fname in files:
    in_file = os.path.join(DATA_DIR, f"{fname}.txt")
    out_file = os.path.join(DATA_DIR, f"{fname}_reverse.txt")
    reverse_interactions(in_file, out_file)

print("完成！反向資料已生成。")