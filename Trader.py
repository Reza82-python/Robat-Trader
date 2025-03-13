import yfinance as yf
import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import keras_tuner as kt


# تابع تبدیل داده به 4 ساعت
def resample_to_4h(data):
    return data.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


# تابع پیدا کردن سطوح کلیدی
def find_key_levels(data, tolerance=0.002):  # 0.2% تحمل
    all_prices = np.concatenate([data["Open"], data["Close"], data["High"], data["Low"]])
    price_levels = {}
    for price in all_prices:
        hits = np.sum(np.abs(all_prices - price) / price < tolerance)
        if hits >= 5:  # حداقل 5 برخورد
            price_levels[price] = hits

    sorted_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)
    return sorted_levels


# تابع آماده‌سازی داده برای Transformer
def prepare_data(data, lookback=50):
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    features = data[["Open", "Close", "High", "Low", "Volume"]].dropna()
    target = data["Target"].dropna()

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i - lookback:i])
        y.append(target.iloc[i])

    X = np.array(X)
    y = np.array(y)
    return X, y, scaler, features.columns


# تابع ساخت مدل برای Keras Tuner
def build_model(hp):
    inputs = Input(shape=(50, 5))  # lookback=50, 5 ویژگی
    x = MultiHeadAttention(
        num_heads=hp.Int("num_heads", 2, 16, step=2),
        key_dim=hp.Int("key_dim", 32, 256, step=32)
    )(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(hp.Float("dropout_1", 0.1, 0.5, step=0.1))(x)
    x = MultiHeadAttention(
        num_heads=hp.Int("num_heads_2", 2, 16, step=2),
        key_dim=hp.Int("key_dim_2", 32, 256, step=32)
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(hp.Float("dropout_2", 0.1, 0.5, step=0.1))(x)
    x = Dense(hp.Int("dense_units", 32, 256, step=32), activation="relu")(x[:, -1, :])
    x = Dropout(hp.Float("dropout_3", 0.1, 0.5, step=0.1))(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# تابع آموزش مدل
def train_transformer_model(data):
    X, y, scaler, feature_cols = prepare_data(data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=15,  # تست بیشتر
        executions_per_trial=1,
        directory="tuner_dir",
        project_name="transformer_4h"
    )

    tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    best_model = tuner.get_best_models(num_models=1)[0]

    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"دقت مدل Transformer روی داده تست: {accuracy:.2f}")
    print(f"بهترین پارامترها: {tuner.get_best_hyperparameters()[0].values}")
    return best_model, X_test, y_test, scaler, feature_cols


# بخش اصلی
symbol = "^DJI"
dow_jones = yf.Ticker(symbol)

print("آماده‌سازی داده‌ها...")
raw_data_4h = dow_jones.history(period="2y", interval="4h")  # 2 سال برای داده بیشتر
data_4h = resample_to_4h(raw_data_4h)

# پیدا کردن سطوح کلیدی
key_levels = find_key_levels(data_4h)
print("\nمهم‌ترین سطوح قیمتی داوجونز (بیشترین برخورد به کمترین):")
for level, hits in key_levels[:20]:  # 20 سطح برتر
    print(f"سطح: {level:.2f} | تعداد برخورد: {hits}")

# آموزش مدل ML
print("\nآموزش مدل Transformer...")
model, X_test, y_test, scaler, feature_cols = train_transformer_model(data_4h)

# اجرای لحظه‌ای
print("\nشروع اجرای لحظه‌ای (هر 4 ساعت)...")
while True:
    try:
        raw_data_4h = dow_jones.history(period="30d", interval="4h")
        data_4h = resample_to_4h(raw_data_4h)

        last_close = data_4h["Close"].iloc[-1]
        last_features = data_4h[feature_cols].iloc[-50:]
        scaled_features = scaler.transform(last_features)
        last_input = np.expand_dims(scaled_features, axis=0)
        ml_pred = (model.predict(last_input, verbose=0) > 0.5).astype(int)[0]
        ml_signal = "صعود" if ml_pred == 1 else "نزول"

        # پیدا کردن نزدیک‌ترین سطح کلیدی
        nearest_level = min(key_levels, key=lambda x: abs(x[0] - last_close))[0]
        print(
            f"{time.ctime()} | قیمت: {last_close:.2f} | نزدیک‌ترین سطح: {nearest_level:.2f} | پیش‌بینی ML: {ml_signal}")
        time.sleep(14400)  # 4 ساعت
    except Exception as e:
        print(f"خطا: {e}")
        time.sleep(60)