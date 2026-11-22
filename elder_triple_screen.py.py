#!/usr/bin/env python3
"""
Elder Triple Screen Trading System (Chapter 39)
===============================================
Implements Dr. Alexander Elder's Triple Screen with his exact rules:
  * Screen 1 (Tide)  : 13-period EMA on the higher timeframe.
  * Screen 2 (Wave)  : 2-period Force Index on the lower timeframe.
  * Screen 3 (Ripple): Entry 1 tick beyond the previous bar's extreme.

Usage:
    python elder_triple_screen.py --mode weekly_daily --tickers AAPL,MSFT,NVDA --confirm --once
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

# ════════════════════════════ CONFIG ════════════════════════════
EMA_TREND = 13
FORCE_N = 2
STATE_FILE = Path("ts_state.json")

TIMEFRAMES = {
    "weekly_daily": {"tide": "1wk", "wave": "1d", "history_tide": "10y", "history_wave": "2y"},
    "daily_hourly": {"tide": "1d", "wave": "60m", "history_tide": "2y", "history_wave": "60d"},
}

def fp(t: str, x: float) -> str:
    if t.endswith("=X"): return f"{x:.3f}" if "JPY" in t else f"{x:.5f}"
    return f"{x:.2f}"

def tick_size(t: str, price: float) -> float:
    if t.endswith("=X"): return 0.01 if "JPY" in t else 0.0001
    return 0.01 if price >= 1 else 0.001

def avoid_round(stop: float, tick: float, sgn: int) -> float:
    stop = round(stop / tick) * tick
    if round(stop / tick) % 50 == 0:
        stop -= sgn * 6 * tick
    return stop

# ════════════════════════════ TELEGRAM ════════════════════════════
def creds() -> tuple[str | None, str | None]:
    tok = os.getenv("TS_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TS_TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")
    return tok, chat

def send_telegram(text: str) -> None:
    tok, chat = creds()
    if not tok or not chat:
        print(f"[no telegram creds] {text.replace(chr(10), ' | ')}")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": text[:4000]}, timeout=15)
        if not r.ok:
            print(f"[telegram error] {r.status_code} {r.text}")
        else:
            print("[telegram] Message sent successfully.")
    except Exception as exc:
        print(f"[telegram error] {exc}")
    time.sleep(1.0)

# ════════════════════════════ DATA ════════════════════════════
def download(tickers: list[str], interval: str, period: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out = {}
    try:
        raw = yf.download(tickers, period=period, interval=interval,
                          group_by="ticker", auto_adjust=True, progress=False, threads=True)
        for t in tickers:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub.dropna(subset=["Close", "High", "Low", "Volume"])
            if len(sub) > 30: out[t] = sub
    except Exception as exc:
        print(f"[warn] download failed for {interval}: {exc}")
    return out

def load_state() -> set[str]:
    try: return set(json.loads(STATE_FILE.read_text()))
    except: return set()

def save_state(sent: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(sent)[-5000:]))

# ════════════════════════════ INDICATORS ════════════════════════════
def force_index(df: pd.DataFrame, n: int = FORCE_N) -> pd.Series:
    raw_force = df["Close"].diff() * df["Volume"]
    return raw_force.ewm(span=n, adjust=False).mean()

def weekly_trend_ema(df_tide: pd.DataFrame) -> pd.Series:
    return df_tide["Close"].ewm(span=EMA_TREND, adjust=False).mean()

# ════════════════════════════ SCANNER ════════════════════════════
def scan_triple_screen(tickers: list[str], mode: str, confirm: bool, sent: set[str]) -> list[str]:
    cfg = TIMEFRAMES[mode]
    tide_data = download(tickers, cfg["tide"], cfg["history_tide"])
    wave_data = download(tickers, cfg["wave"], cfg["history_wave"])
    
    alerts = []
    for t in tickers:
        if t not in tide_data or t not in wave_data: continue
        wdf = tide_data[t]
        df = wave_data[t]
        
        ema = weekly_trend_ema(wdf)
        tide_up = ema.iloc[-1] > ema.iloc[-2]
        tide_dn = ema.iloc[-1] < ema.iloc[-2]
        
        fi = force_index(df, FORCE_N)
        fi_now = float(fi.iloc[-1])
        pullback_long = fi_now < 0
        pullback_short = fi_now > 0
        
        prev_bar = df.iloc[-2]
        curr_bar = df.iloc[-1]
        
        trigger_px_long = prev_bar["High"] + tick_size(t, prev_bar["High"])
        trigger_px_short = prev_bar["Low"] - tick_size(t, prev_bar["Low"])
        
        breakout_long = curr_bar["High"] >= trigger_px_long
        breakout_short = curr_bar["Low"] <= trigger_px_short
        
        if confirm:
            breakout_long = breakout_long and (curr_bar["Close"] >= trigger_px_long)
            breakout_short = breakout_short and (curr_bar["Close"] <= trigger_px_short)
        
        if tide_up and pullback_long and breakout_long:
            key = f"{mode}_long|{t}|{df.index[-1]}"
            if key not in sent:
                swing_low = df["Low"].iloc[-5:].min()
                stop = avoid_round(swing_low - tick_size(t, swing_low), tick_size(t, swing_low), 1)
                confirm_text = "and CLOSED beyond it" if confirm else "(tick breakout)"
                alert = (f"🟢 BUY {t} | Triple Screen LONG\n"
                         f"Timeframes: {cfg['tide']} Tide, {cfg['wave']} Wave\n"
                         f"Screen 1: Trend UP ✅\n"
                         f"Screen 2: Force Index Pullback ✅ (FI: {fi_now:.1f})\n"
                         f"Screen 3: Broke {fp(t, trigger_px_long)} {confirm_text} ✅\n"
                         f"Stop Loss: {fp(t, stop)}")
                alerts.append(alert)
                sent.add(key)
                
        elif tide_dn and pullback_short and breakout_short:
            key = f"{mode}_short|{t}|{df.index[-1]}"
            if key not in sent:
                swing_high = df["High"].iloc[-5:].max()
                stop = avoid_round(swing_high + tick_size(t, swing_high), tick_size(t, swing_high), -1)
                confirm_text = "and CLOSED beyond it" if confirm else "(tick breakout)"
                alert = (f"🔴 SHORT {t} | Triple Screen SHORT\n"
                         f"Timeframes: {cfg['tide']} Tide, {cfg['wave']} Wave\n"
                         f"Screen 1: Trend DOWN ✅\n"
                         f"Screen 2: Force Index Pullback ✅ (FI: {fi_now:.1f})\n"
                         f"Screen 3: Broke {fp(t, trigger_px_short)} {confirm_text} ✅\n"
                         f"Stop Loss: {fp(t, stop)}")
                alerts.append(alert)
                sent.add(key)
    return alerts

# ════════════════════════════ MAIN ════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Elder Triple Screen Scanner")
    ap.add_argument("--tickers", default="AAPL,MSFT,NVDA,SPY", help="Comma separated list")
    ap.add_argument("--mode", choices=["weekly_daily", "daily_hourly"], default="weekly_daily")
    ap.add_argument("--confirm", action="store_true", help="Require the breakout bar to close beyond the trigger price")
    ap.add_argument("--once", action="store_true", help="Run once and exit")
    args = ap.parse_args()
    
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    sent = load_state()
    
    # 1. Send Start Message
    start_msg = (f"🔄 Elder Triple Screen Scan Started\n"
                 f"Mode: {args.mode.replace('_', ' ').title()}\n"
                 f"Tickers: {len(tickers)}\n"
                 f"Confirmation: {'Require CLOSE' if args.confirm else 'Tick Breakout'}")
    send_telegram(start_msg)
    print("Scan started. Downloading data...")
    
    # 2. Run Scan
    alerts = scan_triple_screen(tickers, args.mode, args.confirm, sent)
    
    # 3. Send Alerts
    if alerts:
        chunk = ""
        for a in alerts:
            if len(chunk) + len(a) > 3800:
                send_telegram(chunk)
                chunk = ""
            chunk += a + "\n\n"
        if chunk: send_telegram(chunk)
        
    save_state(sent)
    
    # 4. Send Finish Message
    finish_msg = f"✅ Scan Complete\nFound {len(alerts)} valid setup(s)."
    send_telegram(finish_msg)
    print(f"Done. {len(alerts)} alerts sent.")

if __name__ == "__main__":
    main()
