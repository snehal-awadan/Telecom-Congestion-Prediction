# Modified Congestion prediction code:

from curses import meta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import time
import matplotlib.pyplot as plt

LOOKBACK = 50
TEST_SPLIT = 0.2
EPOCHS = 30
BATCH_SIZE = 32
PRED_HORIZON = 3  # predict 3 time steps ahead

ALPHA = 0.85         # congestion threshold
BASELINE_WIN = 50    # rolling baseline window
PERSIST_K = 3        # consecutive violations
DELTA = 3            # prediction horizon

# Function to infer congestion states
def infer_congestion(
    df,
    y_pred_inv,
    lookback,
    delta,
    alpha=0.85,
    baseline_win=50,
    persist_k=3
):
    df = df.copy()
    df["pred_throughput"] = np.nan

    # align predictions with time
    start_idx = lookback + delta
    df.loc[df.index[start_idx:start_idx + len(y_pred_inv)],
           "pred_throughput"] = y_pred_inv

    results = []

    for link_id, link_df in df.groupby("Link_ID"):
        link_df = link_df.sort_values("Time")

        # rolling baseline
        link_df["baseline"] = (
            link_df["Moving_Average_throughput"]
            .rolling(baseline_win, min_periods=baseline_win)
            .mean()
        )

        # raw congestion trigger
        link_df["cong_raw"] = (
            link_df["pred_throughput"] < alpha * link_df["baseline"]
        ).astype(int)

        # persistence filter
        link_df["congestion"] = (
            link_df["cong_raw"]
            .rolling(persist_k, min_periods=persist_k)
            .sum()
            .ge(persist_k)
            .astype(int)
        )

        results.append(link_df)

    return pd.concat(results)


data = pd.read_csv("/home/snehal/AI_ML/NWDAF/Congestion_prediction/Dataset.csv")

data = data.sort_values(["Link_ID", "Time"]).reset_index(drop=True)

feature_cols = [
    "Instantaneous_throughput",
    "Moving_Average_throughput",
    "Time_average_throughput"
]

# create time delta feature
data["delta_t"] = data.groupby("Link_ID")["Time"].diff().fillna(0)
feature_cols.append("delta_t")

# function to create sequences per link:
def create_sequences_per_link(df, lookback):
    X_all, y_all, meta = list(), list(), list()



# iterate over each link:
    for link_id, link_df in df.groupby("Link_ID"):
        values = link_df[feature_cols].values
        target = link_df["Moving_Average_throughput"].values
        times = link_df["Time"].values

        for i in range(len(link_df) - lookback - PRED_HORIZON):
            X_all.append(values[i:i+lookback])
            y_all.append(target[i + lookback + PRED_HORIZON - 1])
            meta.append(link_id)

    return np.array(X_all), np.array(y_all), meta

train_frames = []
test_frames = []

for link_id, link_df in data.groupby("Link_ID"):
    split_idx = int((1 - TEST_SPLIT) * len(link_df))
    train_frames.append(link_df.iloc[:split_idx])
    test_frames.append(link_df.iloc[split_idx:])

train_df = pd.concat(train_frames)
test_df = pd.concat(test_frames)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_df[feature_cols] = scaler_X.fit_transform(train_df[feature_cols])
test_df[feature_cols] = scaler_X.transform(test_df[feature_cols])

train_df["Moving_Average_throughput"] = scaler_y.fit_transform(
    train_df[["Moving_Average_throughput"]]
)
test_df["Moving_Average_throughput"] = scaler_y.transform(
    test_df[["Moving_Average_throughput"]]
)

X_train, y_train, train_meta = create_sequences_per_link(train_df, LOOKBACK)
X_test, y_test, test_meta = create_sequences_per_link(test_df, LOOKBACK)


# model definition
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

# model compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

start = time.time()

# model training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ],
    verbose=1
)

print(f"Training time: {time.time() - start:.2f}s")

# model evaluation
y_pred = model.predict(X_test)

# inverse transform
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()


# congestion logic:
congestion_df = infer_congestion(
    df=test_df,
    y_pred_inv=y_pred_inv,
    lookback=LOOKBACK,
    delta=DELTA,
    alpha=ALPHA,
    baseline_win=BASELINE_WIN,
    persist_k=PERSIST_K
)


# calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"RMSE: {rmse:.6f}")
print(f"MAE : {mae:.6f}")
print(f"Training for prediction horizon Δ = {PRED_HORIZON}")


# to debug:
print(congestion_df["congestion"].value_counts())
congestion_df.groupby("Link_ID")["congestion"].mean()


# prepare output DataFrame for visualization
output =  pd.DataFrame({
    "Link_ID": test_meta,
    "Time": data["Time"].values[-len(y_test_inv):],
    "actual_MA": y_test_inv,
    "Predicted_MA": y_pred_inv
})

# per-link threshold (from training data)
alpha = 0.85  # congestion sensitivity

link_thresholds = (
    train_df
    .groupby("Link_ID")["Moving_Average_throughput"]
    .median()
    * alpha
)

# predicted congestion (future)
congestion_pred = []

for pred, link_id in zip(y_pred_inv, test_meta):
    threshold = link_thresholds.loc[link_id]
    congestion_pred.append(1 if pred < threshold else 0)

congestion_pred = np.array(congestion_pred)

# true congestion (for evaluation):
true_congestion = []

for true_val, link_id in zip(y_test_inv, test_meta):
    threshold = link_thresholds.loc[link_id]
    true_congestion.append(1 if true_val < threshold else 0)

true_congestion = np.array(true_congestion)

print(f"the length is : {len(y_test_inv)}, {len(y_pred_inv)}, {len(test_meta)}")  # debug : All must be equal.


for link_id in output["Link_ID"].unique():
    link_data = output[output["Link_ID"] == link_id]

    # if len(link_data) < 20:
    #     continue  # skip tiny test windows

    plt.figure(figsize=(10, 5))
    plt.plot(
        link_data["Time"],
        link_data["actual_MA"],
        label="Actual",
        linewidth=2
    )
    plt.plot(
        link_data["Time"],
        link_data["Predicted_MA"],
        label="Predicted",
        linestyle="--"
    )

    plt.title(f"Link {link_id} – Future Throughput Prediction")
    plt.xlabel("Time")
    plt.ylabel("Moving Average Throughput")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/link_{link_id}.png")
    plt.close()

# evaluation metrics :

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(true_congestion, congestion_pred) # Precision = TP / (TP + FP)
recall = recall_score(true_congestion, congestion_pred) # Recall = TP / (TP + FN)
f1 = f1_score(true_congestion, congestion_pred)
cm = confusion_matrix(true_congestion, congestion_pred)

print("\n--- Congestion Prediction Metrics (Δ = 3) ---")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)



# model.save("congestion_future_lstm.h5")
# joblib.dump(scaler_X, "scaler_X.pkl")
# joblib.dump(scaler_y, "scaler_y.pkl")

# print("Model and scalers saved.")


