import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from datetime import datetime, timezone, timedelta  # 👈 新增

# 取得台北時間（UTC+8）
tz_taipei = timezone(timedelta(hours=8))
today     = datetime.now(tz_taipei)
current_month   = today.month               # 1–12
current_weekday = today.weekday() + 1       # Python 0–6 ➜ 1–7

st.set_page_config(page_title="司機人數預測", page_icon="🚚", layout="centered")

@st.cache_resource
def load_model(path: Path):
    m = CatBoostRegressor()
    m.load_model(path)
    return m

MODEL_PATH = Path("driver_catboost_v2.cbm")
model = load_model(MODEL_PATH)

st.title("🚚 司機人數預測")

a, b = st.columns(2)
with a:
    # 預設為今天的月份
    month = st.number_input("今天的月份", min_value=1, max_value=12, value=current_month)

    # selectbox 的 index 從 0 開始，因此要用 current_weekday-1
    weekday = st.selectbox(
        "今天星期幾 (1=週一 … 7=週日)",
        options=list(range(1, 8)),
        index=current_weekday - 1
    )

    weather_display = st.selectbox("明日天氣", ("晴天 (0)", "雨天 (1)"))
    weather = "0" if weather_display.startswith("晴天") else "1"

with b:
    sites      = st.number_input("明日工地",            0, 100, 6)
    trips      = st.number_input("明日車次",            0, 200, 30)
    far_trips  = st.number_input("明日遠距離車次(>留茂安)", 0, 200, 2)
    super_far  = st.number_input("明日超遠(其中超遠台數)",  0, 200, 1)

c, d = st.columns(2)
with c:
    today_staff = st.number_input("今日在職", 0, 100, 7)
with d:
    today_temp  = st.number_input("今日臨時", 0, 100, 2)

if st.button("預測"):
    df = pd.DataFrame({
        "月份":        [month],
        "星期":        [str(weekday)],
        "明日工地":    [sites],
        "明日車次":    [trips],
        "明日遠距離":  [far_trips],
        "明日超遠":    [super_far],
        "今日在職":    [today_staff],
        "今日臨時":    [today_temp],
        "明日天氣":    [str(weather)],
    })

    # 衍生特徵
    df["車次每司機"] = df["明日車次"] / (df["今日在職"] + 1e-5)
    df["遠距比例"]   = df["明日遠距離"] / (df["明日車次"] + 1e-5)
    df["月份_sin"]   = np.sin(2 * np.pi * df["月份"] / 12)
    df["月份_cos"]   = np.cos(2 * np.pi * df["月份"] / 12)

    # 類別型轉字串
    for c in ["星期", "明日天氣"]:
        df[c] = df[c].astype(str)

    # 按模型所需欄位排序
    required_cols = model.feature_names_
    df = df.reindex(columns=required_cols, fill_value=0)

    pool = Pool(df, cat_features=[df.columns.get_loc(c) for c in ["星期", "明日天氣"]])
    pred = model.predict(pool)[0]
    pred_round = int(np.round(pred))

    st.metric("建議司機人數", f"{pred_round} ±1")

st.divider()

st.caption(
    """**5-Fold CV 平均結果**  
Strict  : 25.64 % ± 4.74  
±1      : 69.61 % ± 4.13  
±2      : 88.35 % ± 3.36"""
)
