"""
@description: Chronos-Bolt 风电功率预测 (适配用户数据集)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# 显示中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 加载数据与预处理
# ==========================================
# 请确保路径正确，Windows下建议用绝对路径
file_path = r'D:\damoxing\data\qqqqqqqq20240103_20251102_1315140401.csv'
print(f"正在读取数据: {file_path}")

# 读取数据
wind_date = pd.read_csv(file_path)

# 转换时间列格式
wind_date['date'] = pd.to_datetime(wind_date['date'])

print("原始数据列名:", wind_date.columns.tolist())


# ==========================================
# 2. 数据清洗 (解决 Could not infer frequency 和 length=1 报错的关键！)
# ==========================================


# ==========================================
# ==========================================
# 3. 准备数据集 (加入归一化逻辑)
# ==========================================
prediction_length = 96 * 2

# 1. 初始化归一化器
# 针对目标变量 (corrected_scada_power)
target_scaler = MinMaxScaler(feature_range=(0, 1))

# 针对协变量 (风速 u10, v10 等) - 强烈建议也做归一化
# 假设我们要用 'u10', 'v10', 't2m' 这几列作为协变量
covariate_cols = ['u10', 'v10', 't2m','WS_10','WS_100','WS_200']
cov_scaler = MinMaxScaler(feature_range=(0, 1))

# 2. 划分训练集和测试集 (关键！先划分，再归一化)
# 必须先划分，是为了防止测试集的信息泄露给训练集
train_df = wind_date.iloc[:-prediction_length].copy()
test_df = wind_date.iloc[-prediction_length:].copy()

# 3. Fit (只在训练集上计算最大最小值)
# 注意：fit 的时候只能用 train_df
target_scaler.fit(train_df[['corrected_scada_power']])
if covariate_cols:
    cov_scaler.fit(train_df[covariate_cols])

# 4. Transform (转换训练集和测试集)
# 使用训练集的规则(scaler)去转换测试集
train_df['corrected_scada_power'] = target_scaler.transform(train_df[['corrected_scada_power']])
test_df['corrected_scada_power'] = target_scaler.transform(test_df[['corrected_scada_power']])

if covariate_cols:
    train_df[covariate_cols] = cov_scaler.transform(train_df[covariate_cols])
    test_df[covariate_cols] = cov_scaler.transform(test_df[covariate_cols])

print("归一化完成！")
print(f"训练集目标值范围: {train_df['corrected_scada_power'].min()} - {train_df['corrected_scada_power'].max()}")

# ==========================================
# 4. 加载模型
# ==========================================
model_path = r"D:\damoxing\chronos-2"  # 你的模型路径
pipeline = Chronos2Pipeline.from_pretrained(
    model_path,
    device_map="cpu",  # 如果显存不够报错，请改为 "cpu"
    dtype="auto"
)

# ==========================================
# 5. 场景测试
# ==========================================

# --- 场景 1: 单变量预测 (只给目标值) ---
print("\n>>> 开始场景 1: 单变量预测...")
pred_df0 = pipeline.predict_df(
    train_df[['plantid', 'date', 'corrected_scada_power']],  # 只选这三列
    prediction_length=prediction_length,
    id_column="plantid",  # <--- 修正为 plantid
    timestamp_column="date",  # <--- 修正为 date
    target="corrected_scada_power"  # <--- 修正为 corrected_scada_power
)

# --- 场景 2: 历史协变量预测 (利用 u10, v10 等历史数据) ---
print("\n>>> 开始场景 2: 历史协变量预测...")
pred_df1 = pipeline.predict_df(
    train_df,  # 传入包含所有气象数据的 dateFrame
    prediction_length=prediction_length,
    id_column="plantid",
    timestamp_column="date",
    target="corrected_scada_power"
)

# --- 场景 3: 未来已知协变量预测 (最强模式) ---
print("\n>>> 开始场景 3: 未来协变量预测...")
# 构造未来特征 (去除目标列 corrected_scada_power，保留气象数据)
future_covariates = test_df.drop(columns=['corrected_scada_power'])

pred_df2 = pipeline.predict_df(
    train_df,
    future_df=future_covariates,  # 告诉模型未来的天气
    prediction_length=prediction_length,
    id_column="plantid",
    timestamp_column="date",
    target="corrected_scada_power"
)

# ==========================================
# 6. 评估与绘图
# ==========================================
# 提取预测结果 (取 0.5 分位数即中位数)
y_true = test_df['corrected_scada_power'].values
y_pred0 = pred_df0['0.5'].values
y_pred1 = pred_df1['0.5'].values
y_pred2 = pred_df2['0.5'].values


# 计算指标
def calc_metrics(true, pred, name):
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.4f}")
    return r2, mae


print("\n--- 评估结果 ---")
r2_0, mae_0 = calc_metrics(y_true, y_pred0, "单变量")
r2_1, mae_1 = calc_metrics(y_true, y_pred1, "历史协变量")
r2_2, mae_2 = calc_metrics(y_true, y_pred2, "未来协变量")

# 可视化
plt.figure(figsize=(14, 7))
plt.title(f"Chronos-Bolt 风电功率预测对比 (Plant ID: {test_df['plantid'].iloc[0]})")

# 画真实值
plt.plot(test_df['date'], y_true, label='真实值 (Ground Truth)', color='black', linewidth=2)

# 画预测值
plt.plot(test_df['date'], y_pred0, label=f'单变量 (R2={r2_0:.2f})', linestyle='--')
plt.plot(test_df['date'], y_pred1, label=f'历史气象 (R2={r2_1:.2f})', linestyle='--')
plt.plot(test_df['date'], y_pred2, label=f'未来气象 (R2={r2_2:.2f})', linestyle='-', color='red', linewidth=2)

plt.xlabel('时间')
plt.ylabel('修正功率 (corrected_scada_power)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()