# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.timeseries import TimeSeriesDataFrame
# 直接导入底层模型类
from autogluon.timeseries.models.chronos.chronos2 import Chronos2Model

# ================= 配置区域 =================
# 1. 指定模型所在的具体文件夹 (注意：要精确到 Chronos2 这一层)
#    通常路径是: AutogluonModels/ag-xxx/models/Chronos2
MODEL_FILE_DIR = "AutogluonModels/ag-20260206_070952/models/Chronos2"

# 2. 数据文件
DATA_PATH = "data/ceshi23.csv"


# ===========================================

def main():
    # --- 1. 检查模型文件是否存在 ---
    model_file = os.path.join(MODEL_FILE_DIR, "model.pkl")
    print(f"🔍 检查模型文件: {model_file}")

    if not os.path.exists(model_file):
        print("❌ 错误: 找不到 model.pkl 文件！")
        print("   请确认 MODEL_FILE_DIR 路径是否正确。它应该指向包含 model.pkl 的文件夹。")
        return

    # --- 2. 强行加载底层模型 ---
    print("🚀 正在绕过 Predictor，直接加载 Chronos2 底层模型...")
    try:
        # 直接加载模型对象
        model = Chronos2Model.load(MODEL_FILE_DIR)
        print("✅ 底层模型加载成功！")
        print(f"   - 模型类型: {type(model)}")
        print(f"   - 预测长度: {model.prediction_length}")
    except Exception as e:
        print(f"❌ 底层加载失败: {e}")
        return

    # --- 3. 读取数据 ---
    print(f"📄 读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["plantid"] = df["plantid"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["plantid", "date"]).reset_index(drop=True)

    # 归一化
    target_col = "corrected_scada_power"
    max_power = 252000
    if target_col in df.columns:
        df[target_col] = df[target_col] / max_power

    full_data = TimeSeriesDataFrame.from_data_frame(
        df, id_column="plantid", timestamp_column="date"
    ).fill_missing_values()

    # --- 4. 预测 (使用底层方法) ---
    pred_len = model.prediction_length
    print(f"🔮 正在预测未来 {pred_len} 步...")

    # 准备输入数据 (切掉最后一段，用于回测；或者不切，预测未来)
    # 这里我们切掉最后 pred_len 用于验证效果
    predict_input = full_data.slice_by_timestep(None, -pred_len)

    try:
        # 底层模型的 predict 方法直接接受 TimeSeriesDataFrame
        predictions = model.predict(predict_input)
        print("✅ 预测完成！")
    except Exception as e:
        print(f"❌ 预测出错: {e}")
        return

    # --- 5. 简单的画图验证 ---
    item_id = full_data.item_ids[0]
    pred_mean = predictions.loc[item_id]["mean"] * max_power

    # 获取真实值对比
    y_true = full_data.loc[item_id][target_col].tail(pred_len) * max_power

    # 打印前几个预测值
    print("\n--- 预测值预览 (前5个) ---")
    print(pred_mean.head())

    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true.values, label="真实值 (Ground Truth)", color="black")
    plt.plot(pred_mean.index, pred_mean.values, label="预测值 (Prediction)", color="red", linestyle="--")
    plt.title("强制加载模型预测结果")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_img = "direct_prediction_plot.png"
    plt.savefig(out_img)
    print(f"\n🖼️ 图表已保存至: {out_img}")
    plt.show()


if __name__ == "__main__":
    main()