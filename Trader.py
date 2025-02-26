from pybit.unified_trading import HTTP
import pandas as pd
import time

# اتصال به Bybit (در حالت تست‌نت)
url = HTTP(testnet=True)
# گرفتن اطلاعات آخرین قیمت
response = url.get_index_price_kline(
    category="linear",  # نوع قرارداد (linear برای فیوچرز)
    symbol="BTCUSDT",  # نماد بازار (BTC/USDT)
    interval=240,  # تایم‌فریم (60 دقیقه)
    limit=500,  # تعداد داده‌ها، فقط یک کندل آخر
    start=int(time.time()) - 86400,
)
if 'result' in response:
    kline_data = response['result']
    df = pd.DataFrame(kline_data)
    df[['timestamp', 'open', 'high', 'low', 'close']] = pd.DataFrame(df['list'].to_list(), index=df.index)
    df.drop(columns = ['list', 'category'] , inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'] , unit = 'ms')

    df.to_csv("bybit_btc_price.csv",index=True)

    print(df.head())