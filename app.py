import os
import sys
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Model state ────────────────────────────────────────────────────────────────
_predictor = None
_loaded_model_id = None

MODEL_CONFIGS = {
    "mini":  {"model_id": "NeoQuasar/Kronos-mini",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base", "context": 2048, "params": "4.1M"},
    "small": {"model_id": "NeoQuasar/Kronos-small", "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base", "context": 512,  "params": "24.7M"},
    "base":  {"model_id": "NeoQuasar/Kronos-base",  "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base", "context": 512,  "params": "102.3M"},
}

# ── Instrument catalogue ───────────────────────────────────────────────────────
INSTRUMENTS = {
    "Commodities": [
        {"label": "Gold",        "ticker": "GC=F"},
        {"label": "Silver",      "ticker": "SI=F"},
        {"label": "Crude Oil",   "ticker": "CL=F"},
        {"label": "Natural Gas", "ticker": "NG=F"},
        {"label": "Copper",      "ticker": "HG=F"},
        {"label": "Platinum",    "ticker": "PL=F"},
        {"label": "Wheat",       "ticker": "ZW=F"},
        {"label": "Corn",        "ticker": "ZC=F"},
    ],
    "Forex": [
        {"label": "EUR/USD", "ticker": "EURUSD=X"},
        {"label": "GBP/USD", "ticker": "GBPUSD=X"},
        {"label": "USD/JPY", "ticker": "JPY=X"},
        {"label": "USD/CHF", "ticker": "CHF=X"},
        {"label": "AUD/USD", "ticker": "AUDUSD=X"},
        {"label": "USD/CAD", "ticker": "CAD=X"},
        {"label": "NZD/USD", "ticker": "NZDUSD=X"},
    ],
    "Crypto": [
        {"label": "Bitcoin",  "ticker": "BTC-USD"},
        {"label": "Ethereum", "ticker": "ETH-USD"},
        {"label": "Solana",   "ticker": "SOL-USD"},
        {"label": "BNB",      "ticker": "BNB-USD"},
        {"label": "XRP",      "ticker": "XRP-USD"},
    ],
    "Stocks": [
        {"label": "Apple",     "ticker": "AAPL"},
        {"label": "Microsoft", "ticker": "MSFT"},
        {"label": "NVIDIA",    "ticker": "NVDA"},
        {"label": "Tesla",     "ticker": "TSLA"},
        {"label": "Amazon",    "ticker": "AMZN"},
        {"label": "Meta",      "ticker": "META"},
        {"label": "Google",    "ticker": "GOOGL"},
    ],
    "Indices": [
        {"label": "S&P 500",  "ticker": "^GSPC"},
        {"label": "NASDAQ",   "ticker": "^IXIC"},
        {"label": "Dow Jones","ticker": "^DJI"},
        {"label": "Russell 2000", "ticker": "^RUT"},
        {"label": "VIX",      "ticker": "^VIX"},
        {"label": "FTSE 100", "ticker": "^FTSE"},
        {"label": "Nikkei 225", "ticker": "^N225"},
    ],
}


def get_predictor(model_key: str):
    global _predictor, _loaded_model_id
    if _predictor is not None and _loaded_model_id == model_key:
        return _predictor
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        cfg = MODEL_CONFIGS[model_key]
        logger.info(f"Loading tokenizer {cfg['tokenizer_id']} ...")
        tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer_id"])
        logger.info(f"Loading model {cfg['model_id']} ...")
        model = Kronos.from_pretrained(cfg["model_id"])
        _predictor = KronosPredictor(model, tokenizer, max_context=cfg["context"])
        _loaded_model_id = model_key
        logger.info("Model loaded.")
        return _predictor
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise


def fetch_ohlcv(ticker: str, interval: str, bars: int) -> pd.DataFrame:
    """Fetch OHLCV from yfinance and return lowercase-column DataFrame."""
    period_map = {
        "1d":  f"{max(bars * 2, 500)}d",
        "1h":  "730d",
        "4h":  "730d",
    }
    period = period_map.get(interval, "730d")

    # yfinance 4h workaround: fetch 1h then resample
    fetch_interval = "1h" if interval == "4h" else interval
    df = yf.download(ticker, period=period, interval=fetch_interval, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    if interval == "4h":
        df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()

    return df.tail(bars)


def make_future_timestamps(last_ts: pd.Timestamp, n: int, freq_td: timedelta) -> pd.DatetimeIndex:
    """Generate n future bar timestamps spaced by freq_td."""
    if last_ts.tzinfo is not None:
        last_ts = last_ts.tz_localize(None)
    return pd.DatetimeIndex([last_ts + freq_td * (i + 1) for i in range(n)])


def infer_freq(ts_index: pd.DatetimeIndex) -> timedelta:
    if len(ts_index) < 2:
        return timedelta(days=1)
    diffs = ts_index[1:] - ts_index[:-1]
    return pd.Series(diffs).median().to_pytimedelta()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", instruments=INSTRUMENTS, model_configs=MODEL_CONFIGS)


@app.route("/api/instruments")
def api_instruments():
    return jsonify(INSTRUMENTS)


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    body = request.get_json(force=True)
    ticker      = body.get("ticker", "GC=F").upper()
    interval    = body.get("interval", "1d")
    context_bars = int(body.get("context_bars", 300))
    pred_bars   = int(body.get("pred_bars", 30))
    model_key   = body.get("model", "mini")
    temperature = float(body.get("temperature", 1.0))
    top_p       = float(body.get("top_p", 0.9))
    sample_count = int(body.get("sample_count", 3))

    if model_key not in MODEL_CONFIGS:
        return jsonify({"error": f"Unknown model '{model_key}'"}), 400

    # ── Fetch data ─────────────────────────────────────────────────────────────
    try:
        df = fetch_ohlcv(ticker, interval, context_bars + 50)
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {e}"}), 500

    if len(df) < 30:
        return jsonify({"error": f"Not enough data for {ticker} ({len(df)} bars)"}), 400

    df = df.tail(context_bars)
    x_ts = pd.Series(df.index)
    if hasattr(x_ts.dt, "tz") and x_ts.dt.tz is not None:
        x_ts = x_ts.dt.tz_localize(None)
    x_ts = pd.Series(pd.to_datetime(x_ts).values)

    freq = infer_freq(pd.DatetimeIndex(x_ts))
    y_ts = pd.Series(make_future_timestamps(x_ts.iloc[-1], pred_bars, freq))

    # ── Run Kronos ─────────────────────────────────────────────────────────────
    try:
        predictor = get_predictor(model_key)
        pred_df = predictor.predict(
            df, x_ts, y_ts,
            pred_len=pred_bars,
            T=temperature, top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    def candles(d, label):
        ts = pd.DatetimeIndex(d.index)
        if hasattr(ts, "tz") and ts.tz is not None:
            ts = ts.tz_localize(None)
        return {
            "label": label,
            "x":     [str(t) for t in ts],
            "open":  d["open"].tolist(),
            "high":  d["high"].tolist(),
            "low":   d["low"].tolist(),
            "close": d["close"].tolist(),
        }

    return jsonify({
        "ticker":   ticker,
        "interval": interval,
        "model":    model_key,
        "historical": candles(df, "Historical"),
        "forecast":   candles(pred_df, "Forecast"),
    })


@app.route("/api/price")
def api_price():
    ticker = request.args.get("ticker", "GC=F").upper()
    try:
        t = yf.Ticker(ticker)
        df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return jsonify({"error": "no data"}), 404
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        close = df["close"].dropna()
        price  = float(close.iloc[-1])
        prev   = float(close.iloc[-2]) if len(close) >= 2 else price
        change = price - prev
        pct    = (change / prev) * 100 if prev else 0
        decimals = 2 if price > 10 else (4 if price > 0.1 else 6)
        return jsonify({"price": price, "change": change, "pct": pct, "decimals": decimals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
