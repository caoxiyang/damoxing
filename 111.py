# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1. 读取数据与预处理 ---
file_path = 'data/qqqqqqqq20240103_20251102_1315140401.csv'
print(f"读取数据: {file_path}")
df = pd.read_csv(file_path)
df['plantid'] = df['plantid'].astype(str)
df['date'] = pd.to_datetime(df['date'])

# 按 ID 和 时间 排序
df = df.sort_values(by=['plantid', 'date']).reset_index(drop=True)

# --- 归一化 ---
target_col = "corrected_scada_power"
max_power = 252000
print(f"最大值 {max_power:.2f} (用于归一化)")
df[target_col] = df[target_col] / max_power

# --- 转换格式 ---
full_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date"
).fill_missing_values()

# --- 2. 设定切分长度 (修改部分) ---
# 获取单个序列的总长度用于检查
sample_id = full_data.item_ids[0]
single_ts_length = len(full_data.loc[sample_id])

# ★★★ 修改点：将预测长度固定为 24 小时 ★★★
pred_steps = 24
# 验证集长度建议也设为固定值（例如48小时），而不是总长度的20%，以避免训练集与测试集时间跨度太大
val_steps = 48

# 检查数据是否足够长
if single_ts_length <= (pred_steps + val_steps):
    print(f"错误：数据总长度 ({single_ts_length}) 不足以支持 预测({pred_steps}) + 验证({val_steps})")
    exit()

print(f"--- 数据集划分详情 (Per Series) ---")
print(f"单序列总长度: {single_ts_length}")
print(f"预测长度 (Test): {pred_steps} timesteps (24小时)")
print(f"验证长度 (Val): {val_steps} timesteps")

# --- 3. 切分数据 ---
# 切掉最后 24 小时作为“未来真值”，剩下的作为“历史数据”用于训练和验证
data_history = full_data.slice_by_timestep(None, -pred_steps)

# 再从“历史数据”中切掉最后 48 小时，剩下的作为“纯训练集”
data_train_only = data_history.slice_by_timestep(None, -val_steps)

train_data = data_train_only  # 用于 fit
tune_data = data_history  # 包含验证集，用于 fit 过程中的评估
predict_input_data = data_history  # 预测时的输入（即已知的所有历史）

# --- 4. 初始化 ---
predictor = TimeSeriesPredictor(
    prediction_length=pred_steps,  # ★★★ 这里现在是 24
    target=target_col,
    eval_metric="MAE",
    freq="h"
)

# 设置 huggingface 镜像（如果需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 5. 训练 ---
predictor.fit(
    train_data=train_data,
    tuning_data=tune_data,
    hyperparameters={
        "Chronos2": {
            "fine_tune": True,
            "fine_tune_mode": "full",
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 1500,  # 可以根据需要调整步数
            "fine_tune_batch_size": 32,
            "inference_batch_size": 32,
            "batch_size": 32
        }
    },
    time_limit=3000,
    enable_ensemble=False,
)

# --- 6. 预测 ---
print(f"正在预测未来 {pred_steps} 小时...")
predictions = predictor.predict(predict_input_data)

# --- 7. 画图与评估 ---
plt.figure(figsize=(12, 6))

item_id = full_data.item_ids[0]

# 提取预测结果
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# 提取真实值 (取最后 24 小时)
y_true_ts = full_data.loc[item_id][target_col].tail(pred_steps) * max_power

# 对齐索引
common_index = y_true_ts.index.intersection(pred_mean.index)
y_true_aligned = y_true_ts.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

if len(y_true_aligned) == 0:
    print("错误：无法对齐时间索引。")
else:
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    accuracy = (1 - (mae / max_power)) * 100

    print(f"--- 评估结果 (ID: {item_id}) ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"准确率: {accuracy:.2f}%")

    # --- 绘图 ---
    plt.plot(y_true_ts.index, y_true_ts.values, label="Ground Truth", color="black", linewidth=1.5, marker='.')
    plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=1.5, linestyle="--",
             marker='.')
    plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    plt.gcf().autofmt_xdate()

    plt.title(f"24-Hour Forecast for {item_id} (Acc: {accuracy:.2f}%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()