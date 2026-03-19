# -*- coding: utf-8 -*-
import pandas as pd
from chronos import Chronos2Pipeline

# 1) 路径与数据
finetuned_dir = "/root/root/autodl-tmp/demo/chronos-2-finetuned/2026-02-07_19-23-50/finetuned-ckpt"
file_path = "data/ceshi23.csv"

df = pd.read_csv(file_path)
df["plantid"] = df["plantid"].astype(str)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["plantid", "date"]).reset_index(drop=True)

target_col = "corrected_scada_power"
max_power = 252000
df[target_col] = df[target_col] / max_power

# 2) 取一段上下文（示例：每个序列取前 90% 作为上下文）
sample_id = df["plantid"].iloc[0]
single_ts_length = len(df[df["plantid"] == sample_id])
pred_steps = int(single_ts_length * 0.1)
val_steps = int(single_ts_length * 0.2)
train_steps = single_ts_length - pred_steps - val_steps

context_df_list = []
for item_id, g in df.groupby("plantid"):
    g = g.sort_values("date")
    g_context = g.iloc[: train_steps + val_steps]
    context_df_list.append(g_context)

context_df = pd.concat(context_df_list).reset_index(drop=True)

# 3) 加载模型并预测
pipeline = Chronos2Pipeline.from_pretrained(
    finetuned_dir,
    device_map="cuda",
    local_files_only=True
)

pred_df = pipeline.predict_df(
    context_df,
    future_df=None,
    prediction_length=pred_steps,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="plantid",
    timestamp_column="date",
    target=target_col,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 4) 评估与画图（示例：第一个 plantid）
item_id = df["plantid"].iloc[0]

pred_one = pred_df[pred_df["plantid"] == item_id].set_index("date").sort_index()
if pred_one.empty:
    raise KeyError(f"No predictions for plantid={item_id}")

pred_mean = pred_one["0.5"] * max_power
pred_01 = pred_one["0.1"] * max_power
pred_09 = pred_one["0.9"] * max_power

# 真实值：取测试段（最后 10%）
y_true_ts = df[df["plantid"] == item_id].set_index("date")[target_col] * max_power
y_true_ts = y_true_ts.iloc[-pred_steps:]

common_index = y_true_ts.index.intersection(pred_mean.index)
y_true_aligned = y_true_ts.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

if len(y_true_aligned) == 0:
    print("Error: no overlapping timestamps for evaluation.")
else:
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    accuracy = (1 - (mae / y_true_aligned.max())) * 100

    print(f"--- Eval (ID: {item_id}) ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_ts.index, y_true_ts.values, label="Ground Truth", color="black", linewidth=1.5)
    plt.plot(pred_mean.index, pred_mean.values, label="Prediction (median)", color="#FF3333", linewidth=1.5, linestyle="--")
    plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    plt.gcf().autofmt_xdate()
    plt.title(f"Forecast for {item_id} (Acc: {accuracy:.2f}%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()
#print(pred_df.head())
