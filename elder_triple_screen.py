#!/usr/bin/env python3
"""
Elder Triple Screen Trading System (Chapter 39) - fixed version
================================================================
Your logic, kept:
  Screen 1 (tide)  slope of the 13-EMA on the long-term chart (weekly, or daily in intraday mode)
  Screen 2 (wave)  2-day Force Index against the tide (negative in an uptrend, positive in a downtrend)
  Screen 3 (entry) the next bar breaks the previous bar's high (long) / low (short); --confirm also needs a CLOSE beyond it
  Stop             beyond the 5-bar swing low / high, never at a round number
Same files and settings: watchlists/stocks.json, watchlists/forex.json, ts_state.json, TS_* / TELEGRAM_* variables.

WHAT WAS FIXED
  1. ONE request for every ticker  ->  batches of 60 with a retry. One huge request is fragile (if it fails, nothing is
     scanned); batches fail and retry separately.
  2. FOREX NEVER SIGNALLED. Yahoo reports forex volume as 0, so Force Index (price change x volume) was always 0 and
     "FI < 0" / "FI > 0" could never be true. As Elder advises, the volume of the matching currency future (6E=F for
     EURUSD ...) is now used as the volume proxy. Crosses without a future use a price-only force (volume = 1) and say so.
  3. FORCE INDEX WAS READ ON THE BREAKOUT BAR ITSELF. A bar that breaks the previous high normally closes up, so its Force
     Index is rarely negative and the signal (and the "pullback" label) contradicted itself. Elder's rule: the pullback is
     the bar where Force Index is negative; the NEXT bar's break above that bar's high is the entry. Force Index is now
     read on the bar before the breakout bar.
  4. THE STILL-FORMING BAR WAS USED (today's partial candle, this hour's partial candle, this week's partial week). Alerts
     could appear and vanish. Now only completed bars are used; --live puts the forming bars back for tick-breakout alerts.
  5. "Scanned N tickers" was the number REQUESTED. It now counts what actually had data, per market.
  6. Stocks and forex are separate messages (and can use separate bots), as in the other scanners.
  7. Watchlist files may be a list or a {"TICKER": "sector"} dict for both stocks and forex; "BRK.B" becomes "BRK-B".

Usage
    python elder_triple_screen.py --mode weekly_daily --confirm --once
    python elder_triple_screen.py --mode daily_hourly --confirm --once
    python elder_triple_screen.py --mode daily_hourly --live              # tick breakouts on the forming hourly bar
    python elder_triple_screen.py --intraday-limit 0                      # no cap on stocks in intraday mode

Telegram routing (first one found wins): TS_STOCKS_TELEGRAM_BOT_TOKEN / TS_STOCKS_TELEGRAM_CHAT_ID (or ..._FOREX_...),
then TS_TELEGRAM_BOT_TOKEN / TS_TELEGRAM_CHAT_ID, then TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ════════════════════════════ CONFIG ════════════════════════════
EMA_TREND = 13
FORCE_N = 2
STATE_FILE = Path("ts_state.json")
INTRADAY_LIMIT = 30          # max stocks in daily_hourly mode (0 = no limit); forex is never capped
CHUNK = 60                   # tickers per Yahoo request

TIMEFRAMES = {
    "weekly_daily": {"tide": "1wk", "wave": "1d", "history_tide": "10y", "history_wave": "2y"},
    "daily_hourly": {"tide": "1d", "wave": "60m", "history_tide": "2y", "history_wave": "60d"},
}
BAR_LENGTH = {"1wk": "5D", "1d": "1D", "60m": "60min"}      # a weekly bar is complete once Friday is over
FX_FUT = {"EURUSD=X": "6E=F", "GBPUSD=X": "6B=F", "AUDUSD=X": "6A=F", "NZDUSD=X": "6N=F",
          "USDJPY=X": "6J=F", "JPY=X": "6J=F", "USDCAD=X": "6C=F", "USDCHF=X": "6S=F"}
ICON = {"stocks": "📈", "forex": "💱"}


def market_of(t: str) -> str:
    return "forex" if t.endswith("=X") else "stocks"


def fp(t: str, x: float) -> str:
    if t.endswith("=X"):
        return f"{x:.3f}" if "JPY" in t else f"{x:.5f}"
    return f"{x:.2f}"


def tick_size(t: str, price: float) -> float:
    if t.endswith("=X"):
        return 0.01 if "JPY" in t else 0.0001
    return 0.01 if price >= 1 else 0.001


def avoid_round(stop: float, tick: float, sgn: int) -> float:
    stop = round(stop / tick) * tick
    if round(stop / tick) % 50 == 0:
        stop -= sgn * 6 * tick
    return stop


# ════════════════════════════ TELEGRAM (one route per market) ════════════════════════════
def creds(market: str | None) -> tuple[str | None, str | None]:
    m = market.upper() if market else None
    tok = ((os.getenv(f"TS_{m}_TELEGRAM_BOT_TOKEN") if m else None) or os.getenv("TS_TELEGRAM_BOT_TOKEN")
           or os.getenv("TELEGRAM_BOT_TOKEN"))
    chat = ((os.getenv(f"TS_{m}_TELEGRAM_CHAT_ID") if m else None) or os.getenv("TS_TELEGRAM_CHAT_ID")
            or os.getenv("TELEGRAM_CHAT_ID"))
    return tok, chat


def send_telegram(text: str, market: str | None = None) -> None:
    tok, chat = creds(market)
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
    except Exception as exc:                                     # noqa: BLE001
        print(f"[telegram error] {exc}")
    time.sleep(1.0)


# ════════════════════════════ WATCHLIST LOADING ════════════════════════════
def clean(t) -> str:
    return str(t).strip().upper().replace(".", "-") if not str(t).strip().upper().endswith("=X") else str(t).strip().upper()


def read_list(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:                                     # noqa: BLE001
        print(f"[warn] Could not load {path.name}: {exc}")
        return []
    items = list(data.keys()) if isinstance(data, dict) else list(data)
    return [clean(x) for x in items if str(x).strip()]


def load_watchlists(mode: str, intraday_limit: int) -> list[str]:
    base = Path(__file__).parent / "watchlists"
    stocks, forex = read_list(base / "stocks.json"), read_list(base / "forex.json")
    if mode == "daily_hourly" and intraday_limit > 0:
        print(f"Intraday mode: limiting stocks to the first {intraday_limit} of {len(stocks)}.")
        stocks = stocks[:intraday_limit]
    return list(dict.fromkeys(stocks + forex))


# ════════════════════════════ DATA (batched, completed bars) ════════════════════════════
def drop_forming(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    ts = df.index[-1]
    now = pd.Timestamp.now(tz=ts.tz) if getattr(ts, "tzinfo", None) else pd.Timestamp.now()
    return df.iloc[:-1] if ts + pd.Timedelta(BAR_LENGTH[interval]) > now else df


def download(tickers: list[str], interval: str, period: str, live: bool = False) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i:i + CHUNK]
        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(batch, period=period, interval=interval, group_by="ticker",
                                  auto_adjust=True, progress=False, threads=True)
                break
            except Exception as exc:                             # noqa: BLE001
                print(f"[warn] {interval} batch {i // CHUNK + 1} attempt {attempt + 1} failed: {exc}")
                time.sleep(3)
        if raw is None or raw.empty:
            continue
        for t in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[t]
                else:
                    sub = raw
                sub = sub.dropna(subset=["Close", "High", "Low"])
                sub = sub if live else drop_forming(sub, interval)
                if len(sub) > 30:
                    out[t] = sub
            except Exception as exc:                             # noqa: BLE001
                print(f"[warn] {t}: {exc}")
    return out


def load_state() -> set[str]:
    try:
        return set(json.loads(STATE_FILE.read_text()))
    except Exception:                                            # noqa: BLE001
        return set()


def save_state(sent: set[str]) -> None:
    STATE_FILE.write_text(json.dumps(sorted(sent)[-5000:]))


# ════════════════════════════ VOLUME (forex proxy) AND INDICATORS ════════════════════════════
def align_volume(df: pd.DataFrame, fut: pd.DataFrame, interval: str) -> pd.Series:
    """Volume of a currency future mapped onto the forex bars (Elder's suggested proxy)."""
    v = fut["Volume"].astype(float)
    if interval in ("1d", "1wk"):
        by_day = v.groupby(v.index.date).sum()
        return pd.Series([by_day.get(d, np.nan) for d in df.index.date], index=df.index).ffill().fillna(0)
    fv = v.copy()
    fv.index = fv.index.tz_localize("UTC") if fv.index.tz is None else fv.index.tz_convert("UTC")
    di = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return pd.Series(fv.reindex(di).to_numpy(), index=df.index).fillna(0)


def volume_for(t: str, df: pd.DataFrame, futs: dict, interval: str) -> tuple[pd.Series, str]:
    if market_of(t) == "stocks":
        v = df["Volume"].fillna(0).astype(float) if "Volume" in df else pd.Series(0.0, index=df.index)
        return (v, "volume") if float(v.sum()) > 0 else (pd.Series(1.0, index=df.index), "price-only force (no volume data)")
    code = FX_FUT.get(t)
    f = futs.get(code) if code else None
    if f is not None and "Volume" in f:
        v = align_volume(df, f, interval)
        if float(v.sum()) > 0:
            return v, f"{code} futures volume"
    return pd.Series(1.0, index=df.index), "price-only force (no volume for this pair)"


def fmt_fi(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.5f}"


def force_index(df: pd.DataFrame, vol: pd.Series, n: int = FORCE_N) -> pd.Series:
    return (df["Close"].diff() * vol).fillna(0).ewm(span=n, adjust=False).mean()


def weekly_trend_ema(df_tide: pd.DataFrame) -> pd.Series:
    return df_tide["Close"].ewm(span=EMA_TREND, adjust=False).mean()


# ════════════════════════════ SCANNER ════════════════════════════
def scan_triple_screen(tickers: list[str], mode: str, confirm: bool, sent: set[str], live: bool):
    """Returns (alerts per market, counters per market)."""
    cfg = TIMEFRAMES[mode]
    tide_data = download(tickers, cfg["tide"], cfg["history_tide"], live)
    wave_data = download(tickers, cfg["wave"], cfg["history_wave"], live)
    fx_codes = sorted({FX_FUT[t] for t in tickers if t in FX_FUT})
    futs = download(fx_codes, cfg["wave"], cfg["history_wave"], live) if fx_codes else {}

    alerts: dict[str, list[str]] = {"stocks": [], "forex": []}
    stats = {m: {"requested": 0, "scanned": 0, "proxy": 0, "signals": 0} for m in alerts}
    for t in tickers:
        stats[market_of(t)]["requested"] += 1
        if t not in tide_data or t not in wave_data:
            continue
        wdf, df = tide_data[t], wave_data[t]
        if len(df) < 6 or len(wdf) < 15:
            continue
        mkt = market_of(t)
        stats[mkt]["scanned"] += 1

        ema = weekly_trend_ema(wdf)
        tide_up, tide_dn = ema.iloc[-1] > ema.iloc[-2], ema.iloc[-1] < ema.iloc[-2]

        vol, vlabel = volume_for(t, df, futs, cfg["wave"])
        if vlabel.startswith("price-only") and mkt == "forex":
            stats[mkt]["proxy"] += 1
        fi = force_index(df, vol)
        fi_sig = float(fi.iloc[-2])                    # Force Index on the pullback bar = the bar BEFORE the breakout bar
        pullback_long, pullback_short = fi_sig < 0, fi_sig > 0

        prev_bar, curr_bar = df.iloc[-2], df.iloc[-1]
        trigger_px_long = prev_bar["High"] + tick_size(t, prev_bar["High"])
        trigger_px_short = prev_bar["Low"] - tick_size(t, prev_bar["Low"])
        breakout_long = curr_bar["High"] >= trigger_px_long
        breakout_short = curr_bar["Low"] <= trigger_px_short
        if confirm:
            breakout_long = breakout_long and (curr_bar["Close"] >= trigger_px_long)
            breakout_short = breakout_short and (curr_bar["Close"] <= trigger_px_short)

        confirm_text = "and CLOSED beyond it" if confirm else "(tick breakout)"
        when = f"{df.index[-1]:%Y-%m-%d %H:%M}"
        if tide_up and pullback_long and breakout_long:
            key = f"{mode}_long|{t}|{df.index[-1]}"
            if key not in sent:
                tk = tick_size(t, float(df["Low"].iloc[-5:].min()))
                stop = avoid_round(float(df["Low"].iloc[-5:].min()) - tk, tk, 1)
                alerts[mkt].append(f"🟢 BUY {t} | Triple Screen LONG\n"
                                   f"Timeframes: {cfg['tide']} Tide, {cfg['wave']} Wave (bar {when})\n"
                                   f"Screen 1: Trend UP ✅\n"
                                   f"Screen 2: Force Index Pullback ✅ (FI on the pullback bar: {fmt_fi(fi_sig)}; {vlabel})\n"
                                   f"Screen 3: Broke {fp(t, trigger_px_long)} {confirm_text} ✅\n"
                                   f"Stop Loss: {fp(t, stop)}")
                sent.add(key)
                stats[mkt]["signals"] += 1
        elif tide_dn and pullback_short and breakout_short:
            key = f"{mode}_short|{t}|{df.index[-1]}"
            if key not in sent:
                tk = tick_size(t, float(df["High"].iloc[-5:].max()))
                stop = avoid_round(float(df["High"].iloc[-5:].max()) + tk, tk, -1)
                alerts[mkt].append(f"🔴 SHORT {t} | Triple Screen SHORT\n"
                                   f"Timeframes: {cfg['tide']} Tide, {cfg['wave']} Wave (bar {when})\n"
                                   f"Screen 1: Trend DOWN ✅\n"
                                   f"Screen 2: Force Index Pullback ✅ (FI on the pullback bar: {fmt_fi(fi_sig)}; {vlabel})\n"
                                   f"Screen 3: Broke {fp(t, trigger_px_short)} {confirm_text} ✅\n"
                                   f"Stop Loss: {fp(t, stop)}")
                sent.add(key)
                stats[mkt]["signals"] += 1
    return alerts, stats


# ════════════════════════════ MAIN ════════════════════════════
def run(args) -> None:
    if args.tickers:
        tickers = [clean(t) for t in args.tickers.split(",") if t.strip()]
    else:
        tickers = load_watchlists(args.mode, args.intraday_limit)
        if not tickers:
            tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
    n_st = sum(1 for t in tickers if market_of(t) == "stocks")
    sent = load_state()

    send_telegram(f"🔄 Elder Triple Screen Scan Started\n"
                  f"Mode: {args.mode.replace('_', ' ').title()}\n"
                  f"Tickers: {len(tickers)} (stocks {n_st}, forex {len(tickers) - n_st})\n"
                  f"Confirmation: {'Require CLOSE' if args.confirm else 'Tick Breakout'}"
                  f"{' | live forming bars' if args.live else ''}")
    print(f"Scan started. Downloading data for {len(tickers)} tickers...")

    alerts, stats = scan_triple_screen(tickers, args.mode, args.confirm, sent, args.live)

    for m in ("stocks", "forex"):                              # one message stream per market, never mixed
        chunk = ""
        for a in alerts[m]:
            if len(chunk) + len(a) > 3800:
                send_telegram(chunk, m)
                chunk = ""
            chunk += a + "\n\n"
        if chunk:
            send_telegram(chunk, m)
    save_state(sent)

    for m in ("stocks", "forex"):
        s = stats[m]
        if not s["requested"]:
            continue
        note = f"\n⚠️ {s['proxy']} pair(s) use a price-only force (no futures volume proxy)." if s["proxy"] else ""
        low = "\n⚠️ Many tickers returned no data - Yahoo may have limited this run." if s["scanned"] < 0.7 * s["requested"] else ""
        send_telegram(f"✅ {ICON[m]} {m.upper()} Scan Complete\nScanned {s['scanned']} of {s['requested']} tickers "
                      f"(the rest had no usable data).\nFound {s['signals']} valid setup(s).{note}{low}", m)
    print("Done. " + ", ".join(f"{m}: {stats[m]['scanned']}/{stats[m]['requested']} scanned, {stats[m]['signals']} alert(s)" for m in stats))


def main() -> None:
    ap = argparse.ArgumentParser(description="Elder Triple Screen Scanner")
    ap.add_argument("--tickers", help="Comma separated list (overrides watchlists)")
    ap.add_argument("--mode", choices=["weekly_daily", "daily_hourly"], default="weekly_daily")
    ap.add_argument("--confirm", action="store_true", help="Require the breakout bar to close beyond the trigger price")
    ap.add_argument("--live", action="store_true", help="include the still-forming bars (tick breakouts as they happen)")
    ap.add_argument("--intraday-limit", type=int, default=INTRADAY_LIMIT, help="max stocks in daily_hourly mode (0 = no limit)")
    ap.add_argument("--once", action="store_true", help="Run once and exit (this is also the default)")
    ap.add_argument("--every", type=int, default=0, help="seconds between scans (0 = run once)")
    args = ap.parse_args()
    while True:
        run(args)
        if args.once or args.every <= 0:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
