# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

from chronos import Chronos2Pipeline
model_dir = "/root/root/autodl-tmp/demo/chronos-2"
print("exists:", os.path.isdir(model_dir))
print("config:", os.path.isfile(os.path.join(model_dir, "config.json")))
print("weights:", os.path.isfile(os.path.join(model_dir, "model.safetensors")))
# --- 1. Read data ---
file_path = "data/qqqqqqqq20240103_20251102_1315140401.csv"
print(f"Reading: {file_path}")
df = pd.read_csv(file_path)
df["plantid"] = df["plantid"].astype(str)
df["date"] = pd.to_datetime(df["date"])

# sort by id/time
df = df.sort_values(by=["plantid", "date"]).reset_index(drop=True)

# --- 2. Normalize target ---
target_col = "corrected_scada_power"
max_power = 252000
print(f"Max power: {max_power:.2f} (for normalization)")
df[target_col] = df[target_col] / max_power

# --- 3. Split by timesteps (per series length) ---
sample_id = df["plantid"].iloc[0]
single_ts_length = len(df[df["plantid"] == sample_id])

pred_steps = int(single_ts_length * 0.1)   # last 10% for test
val_steps = int(single_ts_length * 0.2)    # middle 20% for val
train_steps = single_ts_length - pred_steps - val_steps

print("--- Per series split ---")
print(f"series length: {single_ts_length}")
print(f"pred_steps (10%): {pred_steps}")
print(f"val_steps (20%): {val_steps}")
print(f"train_steps (70%): {train_steps}")

# split: 0-90% for context, 0-70% for train
def slice_by_steps(group, end_idx):
    # end_idx is exclusive
    return group.iloc[:end_idx]

train_df_list = []
context_df_list = []
test_df_list = []

for item_id, g in df.groupby("plantid"):
    g = g.sort_values("date")
    g_train = slice_by_steps(g, train_steps)
    g_context = slice_by_steps(g, train_steps + val_steps)
    g_test = g.iloc[-pred_steps:]

    train_df_list.append(g_train)
    context_df_list.append(g_context)
    test_df_list.append(g_test)

train_df = pd.concat(train_df_list).reset_index(drop=True)
context_df = pd.concat(context_df_list).reset_index(drop=True)
test_df = pd.concat(test_df_list).reset_index(drop=True)

# --- 4. Build fine-tuning inputs (no covariates) ---
train_inputs = []
for item_id, g in train_df.groupby("plantid"):
    train_inputs.append({
        "target": g[target_col].values
    })

# --- 5. Load model and fine-tune ---
# choose device_map="cuda" if you have GPU
pipeline = Chronos2Pipeline.from_pretrained(
    r"/root/root/autodl-tmp/demo/chronos-2",
    device_map="cuda",
    local_files_only=True
)

finetuned_pipeline = pipeline.fit(
    inputs=train_inputs,
    prediction_length=pred_steps,
    finetune_mode="full",
    learning_rate=2e-6,
    num_steps=1000,
    batch_size=32,
    logging_steps=100,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,


)

finetuned_pipeline.save_pretrained("/root/root/autodl-tmp/demo/chronos-2-finetuned")

# --- 6. Predict ---
pred_df = finetuned_pipeline.predict_df(
    context_df,

    future_df=None,
    prediction_length=pred_steps,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="plantid",
    timestamp_column="date",
    target=target_col,
)

# --- 7. Evaluate and plot (first id) ---
item_id = df["plantid"].iloc[0]

# pred_df is usually a flat frame with id_column; filter by id instead of .loc
pred_one = pred_df[pred_df["plantid"] == item_id].set_index("date").sort_index()
if pred_one.empty:
    raise KeyError(f"No predictions for plantid={item_id}")

pred_mean = pred_one["0.5"] * max_power
pred_01 = pred_one["0.1"] * max_power
pred_09 = pred_one["0.9"] * max_power

y_true_ts = test_df[test_df["plantid"] == item_id].set_index("date")[target_col] * max_power

common_index = y_true_ts.index.intersection(pred_mean.index)
y_true_aligned = y_true_ts.loc[common_index]
y_pred_aligned = pred_mean.loc[common_index]

if len(y_true_aligned) == 0:
    print("Error: no overlapping timestamps for evaluation.")
else:
    mae = mean_absolute_error(y_true_aligned, y_pred_aligned)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    accuracy = (1 - (mae / max_power)) * 100

    print(f"--- Eval (ID: {item_id}) ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(12, 6))
    plt.plot(y_true_ts.index, y_true_ts.values, label="Ground Truth", color="black", linewidth=1.5)
    plt.plot(pred_mean.index, pred_mean.values, label="Prediction (median)", color="#FF3333", linewidth=1.5, linestyle="--")
    plt.fill_between(pred_mean.index, pred_01, pred_09, color="#FF3333", alpha=0.15, label="CI (0.1-0.9)")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    plt.gcf().autofmt_xdate()
    plt.title(f"Forecast for {item_id} (Acc: {accuracy:.2f}%)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.show()
