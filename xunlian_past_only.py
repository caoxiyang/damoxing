# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os

# --- 1. 读取数据 ---
file_path = 'data/qqqqqqqq20240103_20251102_1315140401.csv'
print(f"读取数据: {file_path}")
df = pd.read_csv(file_path)
df['plantid'] = df['plantid'].astype(str)
df['date'] = pd.to_datetime(df['date'])

# --- 归一化 ---
target_col = "corrected_scada_power"
max_power = 252000
print(f"最大值 {max_power:.2f} (用于归一化)")
df[target_col] = df[target_col] / max_power  # 直接覆盖原列，省得改 target 名字

# --- 2. 转换格式 ---
data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date"
)
data = data.fill_missing_values()

prediction_length = 192
train_data, test_data = data.train_test_split(prediction_length)

# --- 4. 初始化 ---
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target_col,  # 目标列名没变，但数据已经是 0-1 之间
    eval_metric="MAE",
    freq="h"
)

# --- 5. 训练 ---
print("尝试启动训练...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

predictor.fit(
    train_data=train_data,
    hyperparameters={
        "Chronos2": {
            "fine_tune": True,
            "fine_tune_mode": "full",
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 2000,
            "fine_tune_batch_size": 16,
            "inference_batch_size": 16,
            "batch_size": 16
        }
    },
    time_limit=3000,  # time limit in seconds
    enable_ensemble=False,
)

# --- 6. 预测 ---
print("正在预测...")
predictions = predictor.predict(train_data)

# --- 7. 画图 ---
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

plt.figure(figsize=(12, 6))

# 1. 获取 ID
item_id = train_data.item_ids[0]

# 2. 准备数据 (反归一化)
# 预测值
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# 真实值(只取最后 prediction_length 步)
y_future = test_data.loc[item_id][target_col].tail(prediction_length) * max_power

# 确保索引对齐（虽然理论上是对齐的，加一层保险）
common_index = y_future.index.intersection(pred_mean.index)
y_true_aligned = y_future.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

# 计算 MAE (平均绝对误差) 和 RMSE (均方根误差)
mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))

# 计算准确率(公式: 1 - MAE / 最大装机容量)
# 这是电力行业常用的归一化准确率
accuracy = (1 - (mae / max_power)) * 100

print(f"--- 评估结果 ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"准确率(1 - MAE/Capacity): {accuracy:.2f}%")

# --- 绘图 ---
# 画真实值
plt.plot(y_future.index, y_future.values, label="Ground Truth", color="black", linewidth=2, marker='.', markersize=4)

# 画预测值
plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=2, linestyle="--")

# 画置信区间
plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

# --- 设置格式 ---
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.gcf().autofmt_xdate()

# --- 【关键】在标题显示准确率 ---
title_text = (
    f"Prediction Result - Plant ID: {item_id}\n"
    f"Accuracy: {accuracy:.2f}%  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}"
)
plt.title(title_text, fontsize=12, fontweight='bold')

plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.ylabel("Power")
plt.xlabel("Time")

plt.show()
