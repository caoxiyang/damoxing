# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# --- 配置 ---
model_path = "AutogluonModels/ag-20260204_065423"
file_path = "data/qqqqqqqq2q20240103_20251002_1734099.csv"
df = pd.read_csv(file_path)
df["plantid"] = df["plantid"].astype(str)
df["date"] = pd.to_datetime(df["date"])
target_col = "corrected_scada_power"
max_power = 99000  # ��ѵ��ʱһ��
prediction_length = 24


# --- 1. 读取数据 ---
print(f"读取数据: {file_path}")
df = pd.read_csv(file_path)
df["plantid"] = df["plantid"].astype(str)
df["date"] = pd.to_datetime(df["date"])

# --- 2. 归一化 ---
print(f"最大值 {max_power:.2f} (用于归一化)")
df[target_col] = df[target_col] / max_power

# --- 3. 转换格式 ---
data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date",
)
data = data.fill_missing_values()

# 预测时仅使用历史数据，不使用未来协变量
context_data = data.slice_by_timestep(-prediction_length)

# --- 4. 加载模型 ---
print(f"加载模型: {model_path}")
predictor = TimeSeriesPredictor.load(model_path)

# --- 5. 预测 ---
print("正在预测...")
predictions = predictor.predict(context_data)

# --- 6. 画图 ---
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

plt.figure(figsize=(12, 6))

item_id = context_data.item_ids[0]

# 预测值（反归一化）
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# 真实值（取最后 prediction_length 步）
y_future = data.loc[item_id][target_col].tail(prediction_length) * max_power

# 对齐索引
common_index = y_future.index.intersection(pred_mean.index)
y_true_aligned = y_future.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

# 计算 MAE / RMSE
mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
accuracy = (1 - (mae / max_power)) * 100

print(f"--- 评估结果 ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"准确率(1 - MAE/Capacity): {accuracy:.2f}%")

# 绘图
plt.plot(y_future.index, y_future.values, label="Ground Truth", color="black", linewidth=2, marker=".", markersize=4)
plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=2, linestyle="--")
plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:00"))
plt.gcf().autofmt_xdate()

title_text = (
    f"Prediction Result - Plant ID: {item_id}\n"
    f"Accuracy: {accuracy:.2f}%  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}"
)
plt.title(title_text, fontsize=12, fontweight="bold")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.ylabel("Power")
plt.xlabel("Time")
plt.show()
