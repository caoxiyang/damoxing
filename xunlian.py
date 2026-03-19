import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
# --- 1. ��ȡ���� ---
file_path = 'data/qqqqqqqq20240103_20251102_1315140401.csv'
print(f"��ȡ����: {file_path}")
df = pd.read_csv(file_path)
df['plantid'] = df['plantid'].astype(str)
df['date'] = pd.to_datetime(df['date'])

# --- ��һ??---
target_col = "corrected_scada_power"
max_power = 252000
print(f"���?? {max_power:.2f} (���ڹ�һ??")
df[target_col] = df[target_col] / max_power  # ֱ�Ӹ���ԭ�У�ʡ�ø� target ����

# --- 2. ת����ʽ ---
data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date"
)
data = data.fill_missing_values()


prediction_length = 192
train_data, test_data = data.train_test_split(prediction_length)
known_covariates_names = ['u10','v10','u100','v100', 'u200','v200']

# --- 4. ��ʼ??---
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target_col, # Ŀ������û�䣬�������Ѿ�??0-1 ??
    eval_metric="MAE",
    freq="h",
    known_covariates_names=known_covariates_names
)

# --- 5. ѵ�� ---
print("��������ѵ��...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

predictor.fit(
    train_data=train_data,
    hyperparameters={
        "Chronos2": {"fine_tune": True, "fine_tune_mode": "full", "fine_tune_lr": 1e-4, "fine_tune_steps": 2000,
                     "fine_tune_batch_size": 16,"inference_batch_size": 16,
            "batch_size":16 }
    },
    time_limit=3000,  # time limit in seconds
    enable_ensemble=False,
)

# --- 6. Ԥ�� ---
print("����Ԥ��...")
# Only pass future known covariates to avoid leaking target values.
future_known_covariates = test_data[known_covariates_names]
predictions = predictor.predict(train_data, known_covariates=future_known_covariates)

# --- 7. ��ͼ ---
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

plt.figure(figsize=(12, 6))

# 1. ��ȡ ID
item_id = train_data.item_ids[0]

# 2. ׼������ (����һ??
# Ԥ��??
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# ��ʵ??(ֻȡ��??prediction_length ??
y_future = test_data.loc[item_id][target_col].tail(prediction_length) * max_power

# ȷ���������루��Ȼ�������Ƕ���ģ���һ�㱣�գ�
common_index = y_future.index.intersection(pred_mean.index)
y_true_aligned = y_future.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

# ���� MAE (ƽ���������) ??RMSE (��������??
mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))

# ����׼ȷ??(��ʽ?? - MAE / ���װ����??
# ���ǵ�����ҵ���õĹ�һ��׼ȷ��
accuracy = (1 - (mae / max_power)) * 100

print(f"--- ������� ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"׼ȷ??(1 - MAE/Capacity): {accuracy:.2f}%")

# --- ��ͼ ---
# ����ʵ??
plt.plot(y_future.index, y_future.values, label="Ground Truth", color="black", linewidth=2, marker='.', markersize=4)

# ��Ԥ��??
plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=2, linestyle="--")

# ��������??
plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

# --- ���ø�ʽ ---
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.gcf().autofmt_xdate()

# --- ���ؼ����ڱ�����ʾ׼ȷ??---
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
