import yfinance as yf
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# تابع تبدیل داده به 4 ساعته
def resample_to_4h(data):
    return data.resample("4h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


# تابع پیدا کردن سطوح کلیدی
def find_key_levels(data):
    all_prices = np.concatenate([data["High"], data["Low"]])
    price_levels = {}
    tolerance = 0.002
    for price in all_prices:
        hits = np.sum(np.abs(all_prices - price) / price < tolerance)
        if hits >= 2:  # حداقل برخورد رو کم کردیم
            price_levels[price] = hits
    key_levels = sorted(price_levels.items(), key=lambda x: x[1], reverse=True)[:5]
    return [level[0] for level in key_levels]


# تابع محاسبه اندیکاتورها
def calculate_indicators(data):
    data["SMA10"] = data["Close"].rolling(window=10).mean()
    data["SMA30"] = data["Close"].rolling(window=30).mean()
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    return data


# تابع تشخیص روند با 20 کندل
def detect_trend(data, window=20):
    recent = data["Close"].tail(window)
    if len(recent) < window:
        return "نامشخص"
    trend = (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0] * 100
    return "صعودی" if trend > 0 else "نزولی"


# تابع چک کردن الگوی اینگالف
def check_engulfing(data_30m):
    if len(data_30m) < 2:
        return None
    last_candle = data_30m.iloc[-1]
    prev_candle = data_30m.iloc[-2]

    if (prev_candle["Close"] < prev_candle["Open"] and
            last_candle["Close"] > last_candle["Open"] and
            last_candle["Close"] > prev_candle["Open"] and
            last_candle["Open"] < prev_candle["Close"]):
        return "صعودی"
    elif (prev_candle["Close"] > prev_candle["Open"] and
          last_candle["Close"] < last_candle["Open"] and
          last_candle["Close"] < prev_candle["Open"] and
          last_candle["Open"] > prev_candle["Close"]):
        return "نزولی"
    return None


# تابع آموزش مدل ML
def train_ml_model(data):
    data = calculate_indicators(data.copy())
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data["Volume"] = data["Volume"]
    data_clean = data[["Close", "SMA10", "SMA30", "RSI", "MACD", "Signal", "Volume", "Target"]].dropna()
    features = data_clean[["Close", "SMA10", "SMA30", "RSI", "MACD", "Signal", "Volume"]]
    target = data_clean["Target"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"دقت مدل ML روی داده تست: {accuracy:.2f}")
    return model, X_test, y_test


# تابع بک‌تست
def backtest(data_4h, data_30m, key_prices, model, X_test, y_test):
    cash = 10000
    position = 0
    trades = []
    buy_price = 0
    ml_correct = 0
    ml_total = 0
    test_start_idx = len(data_4h) - len(X_test)  # شروع داده‌های تست

    for i in range(30, len(data_4h)):
        close = data_4h["Close"].iloc[i]
        near_key_level = any(abs(close - key) / key < 0.01 for key in key_prices)
        trend = detect_trend(data_4h.iloc[:i + 1])

        # پیش‌بینی ML فقط برای داده‌های تست
        if i >= test_start_idx and i - test_start_idx < len(X_test):
            ml_pred = model.predict(X_test.iloc[i - test_start_idx].values.reshape(1, -1))[0]
            ml_actual = y_test.iloc[i - test_start_idx]
            ml_total += 1
            if ml_pred == ml_actual:
                ml_correct += 1

        # چک کردن اینگالف
        engulfing = None
        if near_key_level:
            current_time = data_4h.index[i]
            window_30m = data_30m.loc[:current_time].tail(2)
            engulfing = check_engulfing(window_30m)

        # خرید (فقط روند و سطح کلیدی، اینگالف اختیاری)
        if near_key_level and trend == "صعودی" and position == 0:
            if engulfing == "صعودی" or engulfing is None:  # اینگالف تأییدیه اختیاری
                position = cash / close
                cash = 0
                buy_price = close
                trades.append(("خرید", close, data_4h.index[i]))

        # فروش
        elif position > 0:
            profit_loss = (close - buy_price) / buy_price * 100
            if near_key_level and trend == "نزولی" and (engulfing == "نزولی" or engulfing is None):
                cash = position * close
                position = 0
                trades.append(("فروش", close, data_4h.index[i]))
            elif profit_loss > 2 or profit_loss < -1:
                cash = position * close
                position = 0
                trades.append(("فروش (حد سود/ضرر)", close, data_4h.index[i]))

    final_value = cash + (position * data_4h["Close"].iloc[-1]) if position > 0 else cash
    profit = final_value - 10000
    ml_accuracy = ml_correct / ml_total * 100 if ml_total > 0 else 0

    print(f"بک‌تست: سرمایه اولیه: 10000 | نهایی: {final_value:.2f} | سود: {profit:.2f}")
    print(f"دقت پیش‌بینی ML در بک‌تست (فقط داده تست): {ml_accuracy:.2f}%")
    print("معاملات:", trades[:5])


# ---- بخش اصلی ----
symbol = "^DJI"
dow_jones = yf.Ticker(symbol)

# گرفتن داده‌ها
print("آماده‌سازی داده‌ها و آموزش مدل...")
raw_data_1h = dow_jones.history(period="1y", interval="1h")
data_4h = resample_to_4h(raw_data_1h)
data_30m = dow_jones.history(period="1y", interval="30m")
key_prices = find_key_levels(data_4h)
model, X_test, y_test = train_ml_model(data_4h)

# بک‌تست
print("\nشروع بک‌تست...")
backtest(data_4h, data_30m, key_prices, model, X_test, y_test)

# اجرای لحظه‌ای
print("\nشروع اجرای لحظه‌ای (هر 5 دقیقه)...")
while True:
    try:
        raw_data_1h = dow_jones.history(period="30d", interval="1h")
        data_4h = resample_to_4h(raw_data_1h)
        data_30m = dow_jones.history(period="5d", interval="30m")
        key_prices = find_key_levels(data_4h)

        last_close = data_4h["Close"].iloc[-1]
        near_key_level = any(abs(last_close - key) / key < 0.01 for key in key_prices)
        trend = detect_trend(data_4h)
        engulfing = check_engulfing(data_30m.tail(2))

        last_features = calculate_indicators(data_4h.copy())[X_test.columns].iloc[-1].to_frame().T
        ml_pred = model.predict(last_features)[0]
        ml_signal = "صعود" if ml_pred == 1 else "نزول"

        signal = "نگه‌داری"
        if near_key_level:
            if trend == "صعودی" and (engulfing == "صعودی" or engulfing is None):
                signal = "خرید"
            elif trend == "نزولی" and (engulfing == "نزولی" or engulfing is None):
                signal = "فروش"

        print(
            f"{time.ctime()} | قیمت: {last_close:.2f} | روند: {trend} | اینگالف: {engulfing} | سیگنال: {signal} | پیش‌بینی ML: {ml_signal}")
        time.sleep(300)
    except Exception as e:
        print(f"خطا: {e}")
        time.sleep(60)