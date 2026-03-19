import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pathlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ????? CPU ??????????
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# === 1) ·������ ===
model_path = 'AutogluonModels/ag-20260203_070713'
file_path = 'data/qqqqqqqq2q20240103_20251002_1734099.csv'
# === 2) ��ȡ��Ԥ���� ===
df = pd.read_csv(file_path)
df["plantid"] = df["plantid"].astype(str)
df["date"] = pd.to_datetime(df["date"])
df["corrected_scada_power"] = df["corrected_power"]
target_col = "corrected_scada_power"
max_power = 99000  # ��ѵ��ʱһ��
# Normalize target the same way as training
df[target_col] = df[target_col] / max_power

# === 3) תΪ TimeSeriesDataFrame ===
data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date",
).fill_missing_values()

prediction_length = 192
known_covariates_names = ["u10", "v10", "u100", "v100", "u200", "v200"]

# ��ѵ��ʱһ�µ��з�
train_data, test_data = data.train_test_split(prediction_length)

# === 4) ����ģ�� ===
# Fix for loading pickles created on POSIX systems
if isinstance(pathlib.Path(), pathlib.WindowsPath):
    pathlib.PosixPath = pathlib.WindowsPath
predictor = TimeSeriesPredictor.load(model_path)

# === 5) ??????????? ===
# === 6) Ԥ�⣨ֻ��δ����֪Э����������й©�� ===
# === 6) 预测：强制调用你训练好的 Chronos2 模型 ===
future_known_covariates = test_data[known_covariates_names]

# 1. 自动获取模型文件夹内 Chronos2 的准确名称
all_models = predictor.model_names()
my_chronos_model = [m for m in all_models if "Chronos" in m]

if not my_chronos_model:
    print("错误：在模型路径中未找到 Chronos 模型！")
else:
    target_model = my_chronos_model[0]
    print(f"正在尝试使用自定义模型: {target_model}")

    try:
        # 尝试预测。如果显存报错，尝试减小 batch_size
        predictions = predictor.predict(
            train_data,
            known_covariates=future_known_covariates,
            model=target_model,  # 强制指定你训练的模型
            # 如果依然报错，取消下面一行的注释来限制推理时的上下文长度
            # model_args={"max_ts_length": 512}
        )
    except RuntimeError as e:
        print(f"Chronos2 预测依然失败: {e}")
        print("提示：这通常是显存(VRAM)不足。请尝试重启 Kernel 以释放显存。")
# === 7) �����뻭ͼ����ѵ��ʱһ�£� ===
item_id = train_data.item_ids[0]

pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

y_future = test_data.loc[item_id][target_col].tail(prediction_length) * max_power

common_index = y_future.index.intersection(pred_mean.index)
y_true_aligned = y_future.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
accuracy = (1 - (mae / max_power)) * 100

print("--- ������� ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"׼ȷ��(1 - MAE/Capacity): {accuracy:.2f}%")

plt.figure(figsize=(12, 6))
plt.plot(y_future.index, y_future.values, label="Ground Truth", color="black", linewidth=2, marker='.', markersize=4)
plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=2, linestyle="--")
plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.gcf().autofmt_xdate()
plt.title(
    f"Prediction Result - Plant ID: {item_id}\n"
    f"Accuracy: {accuracy:.2f}%  |  MAE: {mae:.2f}  |  RMSE: {rmse:.2f}",
    fontsize=12,
    fontweight='bold'
)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.ylabel("Power")
plt.xlabel("Time")
plt.show()
