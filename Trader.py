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
    limit=200,  # تعداد داده‌ها، فقط یک کندل آخر
    start=int(time.time()) - 86400,
)
if 'result' in response:
    kline_data = response['result']

    df = pd.DataFrame(kline_data)
    print(df.head())