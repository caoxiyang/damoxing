import pandas as pd
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import os
# --- 1. 读取数据 ---
# 如果你已经在 demo 文件夹里运行代码：
file_path = 'data/qqqqqqqq20240103_20251102_1315140401.csv'
print(f"读取数据: {file_path}")
df = pd.read_csv(file_path)
df['plantid'] = df['plantid'].astype(str)
df['date'] = pd.to_datetime(df['date'])

# --- 【关键】归一化 (0-1 Scaling) ---
target_col = "corrected_scada_power"
max_power = df[target_col].max()
print(f"最大值: {max_power:.2f} (用于归一化)")
df[target_col] = df[target_col] / max_power  # 直接覆盖原列，省得改 target 名字

# --- 2. 转换格式 ---
data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="plantid",
    timestamp_column="date"
)
data = data.fill_missing_values()

# --- 3. 切分数据 ---
prediction_length = 192
train_data, test_data = data.train_test_split(prediction_length)
known_covariates_names = ['WS_10', 'WS_100', 'WS_200']

# --- 4. 初始化 ---
predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target=target_col, # 目标列名没变，但数据已经是 0-1 了
    eval_metric="MAE",
    freq="h",
    known_covariates_names=known_covariates_names
)

# --- 5. 训练 (修正参数) ---
print("尝试启动训练...")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

predictor = TimeSeriesPredictor(
    prediction_length=prediction_length,
    target="corrected_scada_power",
    known_covariates_names=['WS_10', 'WS_100', 'WS_200'],
    eval_metric="MASE",
).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos2": [
            # Zero-shot model
            {"ag_args": {"name_suffix": "ZeroShot"}},
            # Fine-tuned model
            {"fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
        ]
    },
    time_limit=300,  # time limit in seconds
    enable_ensemble=False,
)
# --- 6. 预测 ---
print("正在预测...")
predictions = predictor.predict(train_data, known_covariates=test_data)

# --- 7. 画图 (修复 AttributeError 和 TypeError) ---
# --- 7. 画图 (修正版：强制只画最后部分) ---
import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))

# 1. 获取 ID
item_id = train_data.item_ids[0]

# 2. 准备预测数据 (这部分没变)
pred_mean = predictions.loc[item_id]["mean"] * max_power
pred_01 = predictions.loc[item_id]["0.1"] * max_power
pred_09 = predictions.loc[item_id]["0.9"] * max_power

# --- 【核心修改】准备真实值 ---
# 我们只取 test_data 的最后 prediction_length 行 (即最后 192 小时)
# 这样 X 轴就只会显示这 8 天，而不是 2 年
y_future = test_data.loc[item_id][target_col].tail(prediction_length) * max_power

# (可选) 如果你想看预测前一点点的趋势，可以多取一点，比如取最后 300 小时：
# y_plot_range = test_data.loc[item_id][target_col].tail(300) * max_power
# 但为了对比预测，我们先只取 y_future

# --- 绘图 ---
# 画真实值
plt.plot(y_future.index, y_future.values, label="Ground Truth", color="black", linewidth=2, marker='.', markersize=4)

# 画预测值
plt.plot(pred_mean.index, pred_mean.values, label="Prediction", color="#FF3333", linewidth=2, linestyle="--")

# 画置信区间
plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

# --- 设置 X 轴格式，防止重叠 ---
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
plt.gcf().autofmt_xdate() # 自动旋转日期标签

plt.title(f"Prediction Zoom-in (Last {prediction_length} Hours)\nPlant ID: {item_id}", fontsize=12)
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.ylabel("Power")
plt.xlabel("Date/Time")

plt.show()