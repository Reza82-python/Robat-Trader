import yfinance as yf
import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error

# تابع پیدا کردن مناطق کلیدی
def find_most_hit_ranges(data, range_size=250, tolerance=0.001):
    all_prices = np.concatenate([data["Open"], data["Close"], data["High"], data["Low"]])
    price_ranges = {}
    for price in all_prices:
        range_start = int(price // range_size) * range_size
        range_end = range_start + range_size
        hits = np.sum(np.abs(all_prices - price) / price < tolerance)
        if (range_start, range_end) in price_ranges:
            price_ranges[(range_start, range_end)] += hits
        else:
            price_ranges[(range_start, range_end)] = hits
    return sorted(price_ranges.items(), key=lambda x: x[1], reverse=True)[:5]

def find_nearby_ranges(data, current_price, range_size=250, tolerance=0.001, range_percent=0.05):
    all_prices = np.concatenate([data["Open"], data["Close"], data["High"], data["Low"]])
    min_price = current_price * (1 - range_percent)
    max_price = current_price * (1 + range_percent)
    nearby_prices = all_prices[(all_prices >= min_price) & (all_prices <= max_price)]
    price_ranges = {}
    for price in nearby_prices:
        range_start = int(price // range_size) * range_size
        range_end = range_start + range_size
        hits = np.sum(np.abs(all_prices - price) / price < tolerance)
        if (range_start, range_end) in price_ranges:
            price_ranges[(range_start, range_end)] += hits
        else:
            price_ranges[(range_start, range_end)] = hits
    return sorted(price_ranges.items(), key=lambda x: x[1], reverse=True)[:5]

# تابع آماده‌سازی داده (رگرسیون)
def prepare_data_regression(data, lookback=50):
    data["Price_Change"] = data["Close"].pct_change()
    data["High_Change"] = data["High"].pct_change()
    data["Low_Change"] = data["Low"].pct_change()
    data["Open_Change"] = data["Open"].pct_change()
    data["MA5"] = data["Close"].rolling(window=5).mean()

    data["Next_Open"] = data["Open"].shift(-1)
    data["Next_Low"] = data["Low"].shift(-1)
    data["Next_Close"] = data["Close"].shift(-1)
    data["Next_High"] = data["High"].shift(-1)

    features = data[["Open", "Close", "High", "Low", "Volume", "Price_Change", "High_Change", "Low_Change", "Open_Change", "MA5"]].dropna()
    targets = data[["Next_Open", "Next_Low", "Next_Close", "Next_High"]].dropna()

    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(features)
    scaled_targets = scaler_targets.fit_transform(targets)

    X, y = [], []
    for i in range(lookback, len(scaled_features)-1):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_targets[i])

    return np.array(X), np.array(y), scaler_features, scaler_targets, features.columns

# تابع آماده‌سازی داده (طبقه‌بندی)
def prepare_data_classification(data, lookback=50):
    data["Price_Change"] = data["Close"].pct_change()
    data["High_Change"] = data["High"].pct_change()
    data["Low_Change"] = data["Low"].pct_change()
    data["Open_Change"] = data["Open"].pct_change()
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    features = data[["Open", "Close", "High", "Low", "Volume", "Price_Change", "High_Change", "Low_Change", "Open_Change", "MA5"]].dropna()
    target = data["Target"].dropna()

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i-lookback:i])
        y.append(target.iloc[i])

    return np.array(X), np.array(y), scaler, features.columns

# تابع آموزش مدل رگرسیون
def train_regression_model(data):
    X, y, scaler_features, scaler_targets, feature_cols = prepare_data_regression(data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    inputs = Input(shape=(50, 10))
    x = Bidirectional(GRU(units=128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(units=128))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(4)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    y_pred = model.predict(X_test)
    y_test_inv = scaler_targets.inverse_transform(y_test)
    y_pred_inv = scaler_targets.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv, multioutput="raw_values")
    print(f"خطای MSE برای پیش‌بینی‌ها: Open={mse[0]:.2f}, Low={mse[1]:.2f}, Close={mse[2]:.2f}, High={mse[3]:.2f}")
    return model, scaler_features, scaler_targets, feature_cols

# تابع آموزش مدل طبقه‌بندی
def train_classification_model(data):
    X, y, scaler, feature_cols = prepare_data_classification(data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    inputs = Input(shape=(50, 10))
    x = Bidirectional(GRU(units=128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(units=128))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stopping], verbose=1)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"دقت مدل طبقه‌بندی روی داده تست: {accuracy:.2f}")
    return model, scaler, feature_cols

# تابع بک‌تست اصلی (بدون ML)
def backtest(data_1h, nearby_ranges):
    initial_cash = 1000
    cash = initial_cash
    position = 0
    trades = []

    for i in range(1, len(data_1h)):
        close = data_1h["Close"].iloc[i]
        prev_close = data_1h["Close"].iloc[i-1]
        timestamp = data_1h.index[i]

        for (start, end), _ in nearby_ranges:
            mid_point = (start + end) / 2
            penetration = (close - start) / (end - start)
            if start <= close <= end and penetration >= 0.5:
                if prev_close < start and position == 0:  # از پایین
                    stop_loss = end
                    entry_price = close
                    risk = stop_loss - entry_price
                    take_profit = entry_price - 2 * risk
                    position = -1
                    cash += entry_price
                    trades.append(("فروش", entry_price, stop_loss, take_profit, timestamp))

                elif prev_close > end and position == 0:  # از بالا
                    stop_loss = start
                    entry_price = close
                    risk = entry_price - stop_loss
                    take_profit = entry_price + 2 * risk
                    position = 1
                    cash -= entry_price
                    trades.append(("خرید", entry_price, stop_loss, take_profit, timestamp))

            if position == 1 and (close >= take_profit or close <= stop_loss):
                cash += close
                position = 0
                trades.append(("بستن خرید", close, timestamp))
            elif position == -1 and (close <= take_profit or close >= stop_loss):
                cash -= close
                position = 0
                trades.append(("بستن فروش", close, timestamp))

    final_value = cash + (position * data_1h["Close"].iloc[-1]) if position != 0 else cash
    profit = final_value - initial_cash
    print(f"\nبک‌تست اصلی: سرمایه اولیه: {initial_cash} | نهایی: {final_value:.2f} | سود: {profit:.2f}")
    print(f"تعداد معاملات: {len(trades)} | معاملات: {trades[:5]}")

# تابع تست دقت ML (رگرسیون)
def test_regression_accuracy(data_1h, model_reg, scaler_features_reg, scaler_targets_reg, feature_cols_reg):
    X, y, _, _, _ = prepare_data_regression(data_1h)
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]

    y_pred = model_reg.predict(X_test, verbose=0)
    y_test_inv = scaler_targets_reg.inverse_transform(y_test)
    y_pred_inv = scaler_targets_reg.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv, multioutput="raw_values")
    print(f"\nتست دقت رگرسیون در بک‌تست: MSE Open={mse[0]:.2f}, Low={mse[1]:.2f}, Close={mse[2]:.2f}, High={mse[3]:.2f}")

# تابع تست دقت ML (طبقه‌بندی)
def test_classification_accuracy(data_1h, model_clf, scaler_clf, feature_cols_clf):
    X, y, _, _ = prepare_data_classification(data_1h)  # فقط 4 مقدار برگشتی
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    y_test = y[train_size:]

    y_pred = (model_clf.predict(X_test, verbose=0) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nتست دقت طبقه‌بندی در بک‌تست: Accuracy={accuracy:.2f}")

# تابع ذخیره داده‌ها در CSV
def save_to_csv(data, filename="dow_jones_data.csv", append=True):
    mode = 'a' if append else 'w'  # اضافه کردن یا بازنویسی
    data.to_csv(filename, mode=mode, header=(not append), index=True)
    print(f"داده‌ها در {filename} ذخیره شد (append={append})")

# بخش اصلی
symbol = "^DJI"
dow_jones = yf.Ticker(symbol)

# داده 1 ساعته اولیه
print("آماده‌سازی داده‌ها (1 ساعته)...")
raw_data_1h = dow_jones.history(period="730d", interval="1h")
data_1h = raw_data_1h.copy()
save_to_csv(raw_data_1h, append=False)  # ذخیره اولیه با بازنویسی

# مناطق کلیدی
current_price = data_1h["Close"].iloc[-1]
most_hit_ranges = find_most_hit_ranges(data_1h)
nearby_ranges = find_nearby_ranges(data_1h, current_price)

print("\nمناطق با بیشترین برخورد (کل داده):")
for (start, end), hits in most_hit_ranges:
    print(f"بازه: {start:.0f}-{end:.0f} | تعداد برخورد: {hits}")

print("\nمناطق نزدیک به قیمت لحظه‌ای:")
for (start, end), hits in nearby_ranges:
    print(f"بازه: {start:.0f}-{end:.0f} | تعداد برخورد: {hits}")

# آموزش مدل رگرسیون
print("\nآموزش مدل رگرسیون (پیش‌بینی 4 قیمت)...")
model_reg, scaler_features_reg, scaler_targets_reg, feature_cols_reg = train_regression_model(data_1h)

# آموزش مدل طبقه‌بندی
print("\nآموزش مدل طبقه‌بندی (صعود/نزول)...")
model_clf, scaler_clf, feature_cols_clf = train_classification_model(data_1h)

# بک‌تست اصلی (بدون ML)
print("\nشروع بک‌تست اصلی (بدون ML)...")
backtest(data_1h, nearby_ranges)

# تست دقت ML
print("\nتست دقت مدل‌های ML در پیش‌بینی کندل بعدی...")
test_regression_accuracy(data_1h, model_reg, scaler_features_reg, scaler_targets_reg, feature_cols_reg)
test_classification_accuracy(data_1h, model_clf, scaler_clf, feature_cols_clf)

# اجرای لحظه‌ای
print("\nشروع اجرای لحظه‌ای (هر ساعت)...")
while True:
    try:
        raw_data_1h = dow_jones.history(period="7d", interval="5m")
        data_1h = raw_data_1h.copy()
        save_to_csv(raw_data_1h, append=True)  # ذخیره هر بار به صورت الحاقی

        data_1h["Price_Change"] = data_1h["Close"].pct_change()
        data_1h["High_Change"] = data_1h["High"].pct_change()
        data_1h["Low_Change"] = data_1h["Low"].pct_change()
        data_1h["Open_Change"] = data_1h["Open"].pct_change()
        data_1h["MA5"] = data_1h["Close"].rolling(window=5).mean()

        last_close = data_1h["Close"].iloc[-1]
        prev_close = data_1h["Close"].iloc[-2] if len(data_1h) > 1 else last_close
        timestamp = data_1h.index[-1]

        most_hit_ranges = find_most_hit_ranges(data_1h)
        nearby_ranges = find_nearby_ranges(data_1h, last_close)

        # پیش‌بینی رگرسیون
        if len(data_1h[feature_cols_reg]) >= 50:
            last_features_reg = data_1h[feature_cols_reg].iloc[-50:]
            scaled_features_reg = scaler_features_reg.transform(last_features_reg)
            last_input_reg = np.expand_dims(scaled_features_reg, axis=0)
            pred_scaled_reg = model_reg.predict(last_input_reg, verbose=0)
            pred_reg = scaler_targets_reg.inverse_transform(pred_scaled_reg)[0]
            ml_reg_str = f"پیش‌بینی رگرسیون: Open={pred_reg[0]:.2f}, Low={pred_reg[1]:.2f}, Close={pred_reg[2]:.2f}, High={pred_reg[3]:.2f}"
        else:
            ml_reg_str = "پیش‌بینی رگرسیون: داده کافی نیست"

        # پیش‌بینی طبقه‌بندی
        if len(data_1h[feature_cols_clf]) >= 50:
            last_features_clf = data_1h[feature_cols_clf].iloc[-50:]
            scaled_features_clf = scaler_clf.transform(last_features_clf)
            last_input_clf = np.expand_dims(scaled_features_clf, axis=0)
            pred_clf = (model_clf.predict(last_input_clf, verbose=0) > 0.5).astype(int)[0]
            ml_clf_str = f"پیش‌بینی طبقه‌بندی: {'صعود' if pred_clf == 1 else 'نزول'}"
        else:
            ml_clf_str = "پیش‌بینی طبقه‌بندی: داده کافی نیست"

        signal = "نگه‌داری"
        for (start, end), _ in nearby_ranges:
            mid_point = (start + end) / 2
            penetration = (last_close - start) / (end - start)
            if start <= last_close <= end and penetration >= 0.5:
                if prev_close < start:
                    signal = f"فروش (استاپ: {end:.2f}, تیک پروفیت: {last_close - 2 * (end - last_close):.2f})"
                elif prev_close > end:
                    signal = f"خرید (استاپ: {start:.2f}, تیک پروفیت: {last_close + 2 * (last_close - start):.2f})"

        print(f"\n{time.ctime()} | قیمت: {last_close:.2f} | سیگنال: {signal}")
        print(ml_reg_str)
        print(ml_clf_str)
        print("مناطق با بیشترین برخورد:")
        for (start, end), hits in most_hit_ranges:
            print(f"بازه: {start:.0f}-{end:.0f} | تعداد برخورد: {hits}")
        print("مناطق نزدیک به قیمت لحظه‌ای:")
        for (start, end), hits in nearby_ranges:
            print(f"بازه: {start:.0f}-{end:.0f} | تعداد برخورد: {hits}")

        time.sleep(3600)
    except Exception as e:
        print(f"خطا: {e}")
        time.sleep(60)