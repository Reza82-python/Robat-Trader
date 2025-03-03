from pybit.unified_trading import HTTP
import pandas as pd
import mplfinance as mpf

# اتصال به Bybit (در حالت تست‌نت)
url = HTTP(testnet=True)
response = url.get_index_price_kline(
    category="linear",
    symbol="BTCUSDT",
    interval=240,  # تایم‌فریم 4 ساعته
    limit=100,
)

# تبدیل داده به DataFrame
if 'result' in response:
    kline_data = response['result']
    df = pd.DataFrame(kline_data)
    df[['timestamp', 'open', 'high', 'low', 'close']] = pd.DataFrame(df['list'].to_list(), index=df.index)
    df.drop(columns=['list', 'category'], inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

    # تبدیل DataFrame به فرمت مورد نیاز برای mplfinance
    df.set_index("timestamp", inplace=True)  # تنظیم ستون timestamp به عنوان index

    # رسم نمودار شمعی (Candlestick Chart)
    mpf.plot(
        df,
        type="candle",         # نمودار شمعی
        style="charles",       # سبک نمایش
        title="Bitcoin Price Chart",
        ylabel="Price (USDT)",
        volume=False,          # نمایش ندادن حجم معاملات
        figratio=(14, 7),      # نسبت ابعاد
    )
