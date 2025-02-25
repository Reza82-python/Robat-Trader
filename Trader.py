import requests
import time
import hmac
import hashlib

# ---- اطلاعات API ----
api_key = "GVEK8T68lnSAcCZ16L"
api_secret = "XYLzLl7h8feWWJ7Mved9vAncZSarJ6RfB1mC"

# ---- تایم‌استمپ (برای احراز هویت) ----
timestamp = str(int(time.time() * 1000))

# ---- پارامترهای درخواست ----
params = {
    "symbol": "BTCUSDT"
}

# ---- ساختن امضا برای درخواست ----
query_string = "&".join([f"{key}={value}" for key, value in params.items()])
signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

# ---- ارسال درخواست ----
url = f"https://api.bybit.com/v5/market/tickers?{query_string}&api_key={api_key}&timestamp={timestamp}&sign={signature}"

response = requests.get(url)
data = response.json()

# ---- بررسی نتیجه ----
if "result" in data:
    btc_price = data["result"]["list"][0]["lastPrice"]
    print(f"📌 قیمت بیت‌کوین: {btc_price} USDT")
else:
    print("❌ خطا در دریافت اطلاعات", data)
