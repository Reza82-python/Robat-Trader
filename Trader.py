from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from matplotlib.patches import Rectangle
from finta import TA

# اتصال به Bybit (در حالت تست‌نت)
url = HTTP(testnet=True)
response = url.get_index_price_kline(
    category="linear",
    symbol="BTCUSDT",
    interval=240,  # تایم‌فریم 4 ساعته
    limit=80,
)

# تبدیل داده به DataFrame
if 'result' in response:
    kline_data = response['result']
    df = pd.DataFrame(kline_data)
    df[['timestamp', 'open', 'high', 'low', 'close']] = pd.DataFrame(df['list'].to_list(), index=df.index)
    df.drop(columns=['list', 'category'], inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['EMA_50'] = TA.EMA(df, 50)
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

# پیدا کردن سطوح کلیدی
def find_strong_levels(df, sensitivity=0.001, min_hits=10, merge_distance=0.002):
    levels = {}

    for price in df['high'].tolist() + df['low'].tolist():  # بررسی تمامی سقف و کف‌ها
        close_prices = df[['open', 'close', 'high', 'low']].values.flatten()  # همه قیمت‌ها
        close_prices = close_prices.astype(float)
        price = float(price)
        hits = np.sum(np.abs(close_prices - price) / price < sensitivity)  # شمارش برخوردها
        if hits >= min_hits:
            levels[price] = hits

    # ترکیب سطوح نزدیک به هم
    zones = []
    sorted_levels = sorted(levels.keys())

    merged_zone = [sorted_levels[0]]  # شروع اولین منطقه
    for level in sorted_levels[1:]:
        if abs(level - merged_zone[-1]) < sensitivity * level:
            merged_zone.append(level)
        else:
            zones.append((min(merged_zone), max(merged_zone)))  # ذخیره منطقه قبلی
            merged_zone = [level]  # شروع منطقه جدید

    if merged_zone:
        zones.append((min(merged_zone), max(merged_zone)))  # ذخیره آخرین منطقه

    # **مرج کردن مناطق نزدیک به هم**
    merged_zones = []
    merged_zone = zones[0]

    for zone in zones[1:]:
        if abs(zone[0] - merged_zone[1]) / zone[0] < merge_distance:
            merged_zone = (merged_zone[0], zone[1])  # ترکیب مناطق
        else:
            merged_zones.append(merged_zone)
            merged_zone = zone

    merged_zones.append(merged_zone)  # ذخیره آخرین منطقه

    return merged_zones

zones = find_strong_levels(df)

# **تنظیم `timestamp` به عنوان index برای mplfinance**
df.set_index("timestamp", inplace=True)

# **ایجاد مستطیل‌های مناطق کلیدی برای `mplfinance`**
rectangles = []
for zone in zones:
    bottom, top = zone
    rect = Rectangle(
        (df.index.min(), bottom),  # شروع از اولین تایم‌استمپ
        df.index.max() - df.index.min(),  # طول مستطیل تا آخرین تایم‌استمپ
        top - bottom,  # ارتفاع منطقه
        linewidth=1.5, edgecolor="blue", facecolor="blue", alpha=0.2
    )
    rectangles.append(rect)

# **تنظیم `mplfinance` برای نمایش کندل‌ها همراه با مناطق کلیدی**
fig, ax = plt.subplots(figsize=(14, 7))

# رسم نمودار کندل‌استیک
mpf.plot(
    df,
    type="candle",         # نمودار شمعی
    style="charles",       # سبک نمایش
    ax=ax,                 # استفاده از محور مشترک
    ylabel="Price (USDT)",
    volume=False,          # عدم نمایش حجم
)

# افزودن مستطیل‌ها به نمودار
for rect in rectangles:
    ax.add_patch(rect)

plt.title("Bitcoin Price Chart with Key Levels")
plt.show()

# نمایش مناطق کلیدی در کنسول
print("🔹 مناطق کلیدی شناسایی‌شده:")
for zone in zones:
    print(f"🟦 منطقه بین {zone[0]:.2f} تا {zone[1]:.2f}")
