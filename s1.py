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

# 定义协变量

# ★★★ 关键修正 1: 先转换为 TimeSeriesDataFrame 再切分 ★★★
# 这样可以保证多站点数据结构正确
full_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date"
).fill_missing_values()

# --- 2. 动态计算切分长度 (基于单个时间序列) ---
# 假设所有站点的长度大致相同，我们取第一个站点的长度来计算比例
sample_id = full_data.item_ids[0]
single_ts_length = len(full_data.loc[sample_id])

# 计算每个序列的切分点 (步长)
pred_steps = int(single_ts_length * 0.1)  # 最后 10% 作为预测长度
val_steps = int(single_ts_length * 0.2)   # 中间 20% 作为验证
train_steps = single_ts_length - pred_steps - val_steps # 剩余的前 70%

print(f"--- 数据集划分详情 (Per Series) ---")
print(f"单序列总长度: {single_ts_length}")
print(f"预测长度 (Test, 10%): {pred_steps} timesteps")
print(f"验证长度 (Val, 20%): {val_steps} timesteps")

# ★★★ 关键修正 2: 使用 slice_by_timestep 进行切分 ★★★
# 1. full_data: 0-100%
# 3. train_data: 0-70% (用于 fit)
# 4. tune_data: 0-90% (用于 fit 中的验证，AutoGluon 会自动切分最后部分做验证)

# 切掉最后 10% 作为“历史数据”
data_0_to_90 = full_data.slice_by_timestep(None, -pred_steps)

# 再从 0-90% 中，切掉最后 20% (即原始数据的 70%-90%)，得到纯训练集 0-70%
data_0_to_70 = data_0_to_90.slice_by_timestep(None, -val_steps)

train_data = data_0_to_70     # 0% -> 70%
tune_data = data_0_to_90      # 0% -> 90% (AutoGluon 会用这部分来验证模型)
predict_input_data = data_0_to_90 # 预测时输入 0-90%，让它预测 90-100%

# --- 4. 初始化 ---
predictor = TimeSeriesPredictor(
    prediction_length=pred_steps, # ★★★ 使用基于时间步的长度，而不是行数
    target=target_col,
    eval_metric="MAE",
    freq="h" # 确保频率正确，如果是15分钟数据请改为 "15min"
)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
predictor.fit(
    train_data=train_data,
    tuning_data=tune_data, # ★★★ 传入验证集，实现 7:2 划分的训练与验证
    hyperparameters={
        "Chronos2": {
            "fine_tune": True,
            "fine_tune_mode": "full",
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 1500,
            "fine_tune_batch_size": 32,
            "inference_batch_size": 32,
            "batch_size": 32
        }
    },
    time_limit=3000,
    enable_ensemble=True,
)

# --- 6. 预测 ---
print("正在预测测试集 (最后 10%)...")
predictions = predictor.predict(predict_input_data)

# --- 7. 画图与评估 ---
plt.figure(figsize=(12, 6))

# 获取第一个 ID 进行画图
item_id = full_data.item_ids[0]

# 提取预测结果
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# 提取真实值 (取全量数据的最后 pred_steps 长度)
y_true_ts = full_data.loc[item_id][target_col].tail(pred_steps) * max_power

# 确保索引对齐
common_index = y_true_ts.index.intersection(pred_mean.index)
y_true_aligned = y_true_ts.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

if len(y_true_aligned) == 0:
    print("错误：无法对齐时间索引。")
else:
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    # 避免分母为0
    accuracy = (1 - (mae / max_power)) * 100
    print(f"--- 评估结果 (ID: {item_id}) ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"准确率: {accuracy:.2f}%")

    # --- 绘图 ---
    plt.plot(y_true_ts.index, y_true_ts.values, label="Ground Truth", color="black", linewidth=1.5)
    plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=1.5, linestyle="--")
    plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    plt.gcf().autofmt_xdate()

    plt.title(f"Forecast for {item_id} (Acc: {accuracy:.2f}%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()
