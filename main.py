import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/coco_yolo26n_Fine-tuning/results.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(12, 6))

plt.plot(df["epoch"], df["metrics/mAP50(B)"])

plt.xticks(range(0, 101, 5))  # X축을 5 Epoch 간격으로 표시

plt.xlabel("Epoch")
plt.ylabel("mAP50")
plt.title("YOLO26n Fine-tuning - mAP50")

plt.grid(True)
plt.show()