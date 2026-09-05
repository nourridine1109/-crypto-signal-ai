
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

st.set_page_config(
    page_title="Crypto Signal AI",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 5rem; max-width: 760px;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 16px;
    padding: 12px;
}
.signal-card {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 20px;
    padding: 16px;
    margin: 10px 0;
}
.bigscore {font-size: 2rem; font-weight: 800;}
.small {opacity: .72; font-size: .9rem;}
h1 {font-size: 1.85rem !important;}
</style>
""", unsafe_allow_html=True)
BASE = "https://api.binance.com"
BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

DEFAULT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "KASUSDT"]

@st.cache_data(ttl=45)
def klines(symbol, interval, limit=500):
    last_error = None

    for base in BASES:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                },
                timeout=12
            )

            r.raise_for_status()

            data = r.json()

            if not data:
                continue

            cols = [
                "time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]

            df = pd.DataFrame(data, columns=cols)

            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

            return df

        except Exception as e:
            last_error = str(e)

    st.warning(f"{symbol}: Datenfehler – {last_error}")
    return None

def indicators(df):
    d=df.copy()
    for n in [20,50,200]:
        d[f"ema{n}"]=d.close.ewm(span=n,adjust=False).mean()
    delta=d.close.diff()
    up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    dn=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["rsi"]=100-(100/(1+up/dn))
    pc=d.close.shift()
    tr=pd.concat([(d.high-d.low),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    d["vma"]=d.volume.rolling(20).mean()
    d["hh"]=d.high.rolling(20).max().shift(1)
    d["ll"]=d.low.rolling(20).min().shift(1)
    return d

def analyze(symbol):
    d15=indicators(klines(symbol,"15m"))
    d1=indicators(klines(symbol,"1h"))
    d4=indicators(klines(symbol,"4h"))
    a,b,c=d15.iloc[-1],d1.iloc[-1],d4.iloc[-1]
    score=50; why=[]

    if c.ema20>c.ema50>c.ema200:
        score+=18; why.append("4H Trend bullish")
    elif c.ema20<c.ema50<c.ema200:
        score-=18; why.append("4H Trend bearish")

    if b.close>b.ema20>b.ema50:
        score+=12; why.append("1H über EMA20/50")
    elif b.close<b.ema20<b.ema50:
        score-=12; why.append("1H unter EMA20/50")

    if 52<=b.rsi<=68:
        score+=8; why.append("1H Momentum bullish")
    elif 32<=b.rsi<=48:
        score-=8; why.append("1H Momentum bearish")

    if a.close>a.hh:
        score+=10; why.append("15M Breakout")
    elif a.close<a.ll:
        score-=10; why.append("15M Breakdown")

    if pd.notna(a.vma) and a.volume>1.25*a.vma:
        if a.close>=a.open:
            score+=7; why.append("Käufer-Volumen")
        else:
            score-=7; why.append("Verkäufer-Volumen")

    if b.rsi>74: score-=10; why.append("Überkauft")
    if b.rsi<26: score+=10; why.append("Überverkauft")

    score=max(0,min(100,int(round(score))))
    long_score=score
    short_score=100-score
    direction="LONG" if long_score>=55 else "SHORT" if short_score>=55 else "NEUTRAL"
    strength=max(long_score,short_score) if direction!="NEUTRAL" else 50

    if b.ema20>b.ema50>b.ema200:
        regime="Aufwärtstrend"
    elif b.ema20<b.ema50<b.ema200:
        regime="Abwärtstrend"
    else:
        regime="Range / Übergang"

    price=float(a.close); atr=float(a.atr)
    lows=d1.tail(80).low
    highs=d1.tail(80).high
    support=float(lows.min())
    resistance=float(highs.max())

    if direction=="LONG":
        sl=min(price-1.5*atr,support-.15*atr)
        risk=max(price-sl,atr)
        entry=(price-.25*atr,price+.1*atr)
        tps=(price+1.5*risk,price+2.5*risk,price+4*risk)
    elif direction=="SHORT":
        sl=max(price+1.5*atr,resistance+.15*atr)
        risk=max(sl-price,atr)
        entry=(price-.1*atr,price+.25*atr)
        tps=(price-1.5*risk,price-2.5*risk,price-4*risk)
    else:
        sl=np.nan; entry=(np.nan,np.nan); tps=(np.nan,np.nan,np.nan)

    return dict(symbol=symbol,price=price,direction=direction,score=strength,
                long=long_score,short=short_score,regime=regime,
                support=support,resistance=resistance,entry=entry,sl=sl,tps=tps,
                why=why[-5:],chart=d1.tail(120).set_index("time")[["close","ema20","ema50","ema200"]])

def f(x):
    if pd.isna(x): return "–"
    if abs(x)>=1000: return f"{x:,.2f}"
    if abs(x)>=1: return f"{x:.4f}"
    return f"{x:.8f}"

st.title("📈 Crypto Signal AI")
st.caption("iPhone Dashboard · 4H Trend → 1H Setup → 15M Entry")

with st.expander("⚙️ Watchlist & Filter"):
    txt=st.text_input("Paare",",".join(DEFAULT))
    threshold=st.slider("Starkes Signal ab",50,95,70)
    refresh=st.button("🔄 Jetzt aktualisieren",use_container_width=True)

symbols=[x.strip().upper() for x in txt.split(",") if x.strip()]

results=[]; errors=[]
with st.spinner("Markt wird analysiert …"):
    for s in symbols:
        try: results.append(analyze(s))
        except Exception as e: errors.append(f"{s}: keine Daten")

results=sorted(results,key=lambda x:x["score"],reverse=True)

if results:
    strong=[r for r in results if r["score"]>=threshold and r["direction"]!="NEUTRAL"]
    c1,c2,c3=st.columns(3)
    c1.metric("Coins",len(results))
    c2.metric("Starke Setups",len(strong))
    c3.metric("Top Score",max(r["score"] for r in results))

    st.subheader("🎯 Signal Radar")
    for r in results:
        icon="🟢" if r["direction"]=="LONG" else "🔴" if r["direction"]=="SHORT" else "⚪"
        st.markdown(
            f"""<div class="signal-card">
            <b>{icon} {r['symbol'].replace('USDT','')}</b>
            <span style="float:right" class="bigscore">{r['score']}/100</span><br>
            <span>{r['direction']} · {r['regime']}</span><br>
            <span class="small">Preis: {f(r['price'])}</span>
            </div>""", unsafe_allow_html=True
        )
        with st.expander(f"Details {r['symbol'].replace('USDT','')}"):
            a,b,c=st.columns(3)
            a.metric("Support",f(r["support"]))
            b.metric("Preis",f(r["price"]))
            c.metric("Resistance",f(r["resistance"]))
            st.write(f"**Entry:** {f(r['entry'][0])} – {f(r['entry'][1])}")
            st.write(f"**Stop-Loss:** {f(r['sl'])}")
            st.write(f"**TP1:** {f(r['tps'][0])}  ·  **TP2:** {f(r['tps'][1])}  ·  **TP3:** {f(r['tps'][2])}")
            st.progress(r["score"]/100,text=f"Signalqualität {r['score']}/100")
            st.write("**Warum?**")
            for w in r["why"]: st.write("• "+w)
            st.write("**3-I-Check:** impulsiv? irrational? inkonsequent? → bei Ja kein Trade.")
            st.line_chart(r["chart"])

if errors:
    st.warning(" · ".join(errors))

st.caption("Technischer Signal-Assistent für Paper-Trading. Keine Anlageberatung. Signale können fehlschlagen.")
