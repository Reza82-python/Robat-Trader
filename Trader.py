import yfinance as yf
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import keras_tuner as kt


# تابع تبدیل داده به 1 ساعت
def resample_to_1h(data):
    return data.resample("1h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna()


# تابع محاسبه MFI
def calculate_mfi(data, period=14):
    data["Typical_Price"] = (data["High"] + data["Low"] + data["Close"]) / 3
    data["Raw_Money_Flow"] = data["Typical_Price"] * data["Volume"]
    data["Price_Diff"] = data["Typical_Price"].diff()
    data["Positive_Flow"] = np.where(data["Price_Diff"] > 0, data["Raw_Money_Flow"], 0)
    data["Negative_Flow"] = np.where(data["Price_Diff"] < 0, data["Raw_Money_Flow"], 0)
    data["Positive_Sum"] = data["Positive_Flow"].rolling(window=period).sum()
    data["Negative_Sum"] = data["Negative_Flow"].rolling(window=period).sum()
    data["Money_Ratio"] = data["Positive_Sum"] / data["Negative_Sum"]
    data["MFI"] = 100 - (100 / (1 + data["Money_Ratio"]))
    data["MFI_Buy_Signal"] = np.where(data["MFI"] < 20, 1, 0)
    data["MFI_Sell_Signal"] = np.where(data["MFI"] > 80, 1, 0)
    return data


# تابع محاسبه SMMA
def calculate_smma(data, period=100):
    data["SMMA"] = data["Close"].rolling(window=period).mean()
    for i in range(period, len(data)):
        data.loc[data.index[i], "SMMA"] = (
                (data["SMMA"].iloc[i - 1] * (period - 1) + data["Close"].iloc[i]) / period
        )
    data["Price_Change"] = data["Close"].pct_change()
    data["Price_Diff"] = data["Close"].diff()
    data["Volatility"] = data["High"] - data["Low"]
    data["High_Change"] = data["High"].pct_change()
    data["Low_Change"] = data["Low"].pct_change()
    return data


# تابع آماده‌سازی داده
def prepare_data(data, lookback=50):
    data = calculate_smma(data.copy())
    data = calculate_mfi(data)
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    features = data[["Close", "SMMA", "MFI", "Price_Change", "Price_Diff", "Volume", "Volatility", "High_Change",
                     "Low_Change"]].dropna()
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
    inputs = Input(shape=(50, 9))  # lookback=50, 9 ویژگی
    x = MultiHeadAttention(
        num_heads=hp.Int("num_heads", 2, 8, step=2),
        key_dim=hp.Int("key_dim", 32, 128, step=32)
    )(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(hp.Float("dropout_1", 0.1, 0.5, step=0.1))(x)
    x = MultiHeadAttention(
        num_heads=hp.Int("num_heads_2", 2, 8, step=2),
        key_dim=hp.Int("key_dim_2", 32, 128, step=32)
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(hp.Float("dropout_2", 0.1, 0.5, step=0.1))(x)
    x = Dense(hp.Int("dense_units", 32, 128, step=32), activation="relu")(x[:, -1, :])
    x = Dropout(hp.Float("dropout_3", 0.1, 0.5, step=0.1))(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# تابع آموزش مدل با Keras Tuner
def train_transformer_model(data):
    X, y, scaler, feature_cols = prepare_data(data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,  # تعداد تست‌ها
        executions_per_trial=1,
        directory="tuner_dir",
        project_name="transformer_tuning"
    )

    tuner.search(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    best_model = tuner.get_best_models(num_models=1)[0]

    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"دقت مدل Transformer روی داده تست: {accuracy:.2f}")
    print(f"بهترین پارامترها: {tuner.get_best_hyperparameters()[0].values}")
    return best_model, X_test, y_test, scaler, feature_cols


# تابع بک‌تست
def backtest(data_1h, model, X_test, y_test, scaler, feature_cols):
    data_1h = calculate_smma(data_1h.copy())
    data_1h = calculate_mfi(data_1h)
    initial_cash = 10000
    cash = initial_cash
    position = 0
    trades = []
    ml_correct = 0
    ml_total = 0
    test_start_idx = len(data_1h) - len(X_test)

    for i in range(100, len(data_1h)):
        close = data_1h["Close"].iloc[i]
        smma = data_1h["SMMA"].iloc[i]
        mfi_buy = data_1h["MFI_Buy_Signal"].iloc[i]
        mfi_sell = data_1h["MFI_Sell_Signal"].iloc[i]

        if i >= test_start_idx and i - test_start_idx < len(X_test):
            ml_pred = (model.predict(X_test[i - test_start_idx:i - test_start_idx + 1], verbose=0) > 0.5).astype(int)[0]
            ml_actual = y_test[i - test_start_idx]
            ml_total += 1
            if ml_pred == ml_actual:
                ml_correct += 1

        # خرید
        if mfi_buy == 1 and close > smma and position == 0:
            risk_amount = initial_cash * 0.01  # 1% ریسک
            stop_loss = close * 0.99  # 1% ضرر
            take_profit = close * 1.02  # 2% سود (1:2)
            position_size = risk_amount / (close - stop_loss)
            position = position_size
            cash -= position * close
            trades.append(("خرید", close, stop_loss, take_profit, data_1h.index[i]))

        # فروش
        elif position > 0:
            if mfi_sell == 1 and close < smma:
                cash += position * close
                position = 0
                trades.append(("فروش (MFI)", close, data_1h.index[i]))
            elif close >= take_profit or close <= stop_loss:
                cash += position * close
                position = 0
                trades.append(("فروش (حد)", close, data_1h.index[i]))

    final_value = cash + (position * data_1h["Close"].iloc[-1]) if position > 0 else cash
    profit = final_value - initial_cash
    ml_accuracy = ml_correct / ml_total * 100 if ml_total > 0 else 0

    print(f"بک‌تست: سرمایه اولیه: {initial_cash} | نهایی: {final_value:.2f} | سود: {profit:.2f}")
    print(f"دقت پیش‌بینی ML در بک‌تست: {ml_accuracy:.2f}%")
    print(f"تعداد معاملات: {len(trades)} | معاملات: {trades[:5]}")


# بخش اصلی
symbol = "^DJI"
dow_jones = yf.Ticker(symbol)

print("آماده‌سازی داده‌ها و آموزش مدل...")
raw_data_1h = dow_jones.history(period="1y", interval="1h")  # 1 سال برای داده بیشتر
data_1h = resample_to_1h(raw_data_1h)
model, X_test, y_test, scaler, feature_cols = train_transformer_model(data_1h)

print("\nشروع بک‌تست...")
backtest(data_1h, model, X_test, y_test, scaler, feature_cols)

# اجرای لحظه‌ای
print("\nشروع اجرای لحظه‌ای (هر 1 ساعت)...")
while True:
    try:
        raw_data_1h = dow_jones.history(period="7d", interval="1h")
        data_1h = resample_to_1h(raw_data_1h)
        data_1h = calculate_smma(data_1h)
        data_1h = calculate_mfi(data_1h)

        last_close = data_1h["Close"].iloc[-1]
        smma = data_1h["SMMA"].iloc[-1]
        mfi_buy = data_1h["MFI_Buy_Signal"].iloc[-1]
        mfi_sell = data_1h["MFI_Sell_Signal"].iloc[-1]

        last_features = data_1h[feature_cols].iloc[-50:]
        scaled_features = scaler.transform(last_features)
        last_input = np.expand_dims(scaled_features, axis=0)
        ml_pred = (model.predict(last_input, verbose=0) > 0.5).astype(int)[0]
        ml_signal = "صعود" if ml_pred == 1 else "نزول"

        signal = "نگه‌داری"
        if mfi_buy == 1 and last_close > smma:
            signal = "خرید (1% ریسک، 1:2)"
        elif mfi_sell == 1 and last_close < smma:
            signal = "فروش (1% ریسک، 1:2)"

        print(
            f"{time.ctime()} | قیمت: {last_close:.2f} | SMMA: {smma:.2f} | سیگنال: {signal} | پیش‌بینی ML: {ml_signal}")
        time.sleep(3600)  # 1 ساعت
    except Exception as e:
        print(f"خطا: {e}")
        time.sleep(60)