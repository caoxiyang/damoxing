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

# ★★★ 关键步骤：必须按时间排序，否则切分无效 ★★★
df = df.sort_values(by='date').reset_index(drop=True)

# --- 归一化 ---
target_col = "corrected_scada_power"
max_power = 252000
print(f"最大值 {max_power:.2f} (用于归一化)")
df[target_col] = df[target_col] / max_power

# --- 2. 数据集划分 (7:2:1) ---
n = len(df)
train_end_idx = int(n * 0.7)  # 70% 处
val_end_idx = int(n * 0.9)    # 90% 处 (70% + 20%)

# 这里的逻辑是：
# train_df: 0% -> 70% (用于训练权重)
# tune_df:  0% -> 90% (AutoGluon 会用 70%-90% 这部分做验证/调优)
# test_df:  0% -> 100% (实际上我们只需要最后的 10% 做真值对比)

df_train = df.iloc[:train_end_idx]
df_tune = df.iloc[:val_end_idx]   # 包含训练集+验证集，用于fit时的tuning_data
df_test_full = df                 # 全量数据，用于提取最后10%做对比
known_covariates_names = ['u10','v10','u100','v100', 'u200','v200']
# 计算测试集的长度（预测长度）

prediction_length = len(df) - val_end_idx
print(f"--- 数据集划分详情 ---")
print(f"总数据量: {n}")
print(f"训练集(Train): {len(df_train)} (前 70%)")
print(f"验证段(Val)  : {len(df_tune) - len(df_train)} (中间 20%)")
print(f"测试集(Test) : {prediction_length} (最后 10%) -> 这也是本次的预测长度")

# 转换为 AutoGluon 格式
train_data = TimeSeriesDataFrame.from_data_frame(df_train, id_column="plantid", timestamp_column="date").fill_missing_values()
tune_data = TimeSeriesDataFrame.from_data_frame(df_tune, id_column="plantid", timestamp_column="date").fill_missing_values()
# 注意：predict_input 必须包含“历史数据”才能预测“未来”，所以预测测试集时，输入应该是 0-90% 的数据
predict_input_data = tune_data


# --- 4. 初始化 ---
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length, # 动态设置为最后 10% 的长度
    target=target_col,
    known_covariates_names=known_covariates_names,
    eval_metric="MAE",
    freq="h"
)

future_index = predictor.make_future_data_frame(predict_input_data)
future_cov = (
    df.iloc[val_end_idx:][["plantid","date"] + known_covariates_names]
      .rename(columns={"plantid":"item_id","date":"timestamp"})
)
future_known_covariates = future_index.merge(
    future_cov,
    on=["item_id","timestamp"],
    how="left"
)

# --- 5. 训练 ---
print("尝试启动训练...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

predictor.fit(
    train_data=train_data,
    tuning_data=tune_data, # ★★★ 传入验证集，实现 7:2 划分的训练与验证
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
    time_limit=3000,
    enable_ensemble=False,
)

# --- 6. 预测 ---
print("正在预测测试集 (最后 10%)...")


# 使用 tune_data (前90%) 作为历史，预测接下来的未来 (后10%)
predictions = predictor.predict(train_data, known_covariates=future_known_covariates)

# --- 7. 画图与评估 ---
plt.figure(figsize=(12, 6))

item_id = train_data.item_ids[0]

# 准备预测值 (反归一化)
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# 准备真实值 (取最后 10% 的数据)
# 从原始 df 中获取最后 prediction_length 长度的数据
y_future_df = df_test_full.iloc[val_end_idx:]
# 确保索引是 datetime 并且是 Series 格式
y_future = pd.Series(
    data=y_future_df[target_col].values * max_power,
    index=y_future_df['date']
)

# 对齐数据计算误差
common_index = y_future.index.intersection(pred_mean.index)
y_true_aligned = y_future.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

if len(y_true_aligned) == 0:
    print("错误：无法对齐时间索引，请检查数据时间戳格式。")
else:
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    accuracy = (1 - (mae / max_power)) * 100

    print(f"--- 评估结果 (测试集 10%) ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"准确率: {accuracy:.2f}%")

    # --- 绘图 ---
    # 画真实值
    plt.plot(y_future.index, y_future.values, label="Ground Truth (Test Set)", color="black", linewidth=1.5)

    # 画预测值
    plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=1.5, linestyle="--")

    # 画置信区间
    plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

    # 格式化
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    plt.gcf().autofmt_xdate()

    title_text = (
        f"Prediction (Test Set: Last 10%)\n"
        f"Acc: {accuracy:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f}"
    )
    plt.title(title_text, fontsize=12, fontweight='bold')
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.ylabel("Power")
    plt.xlabel("Time")
    plt.show()