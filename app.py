import os
import sys
import json
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests
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
    "Energy": [
        {"label": "Crude Oil (WTI)", "ticker": "CL=F",  "unit": "USD/bbl"},
        {"label": "Natural Gas",     "ticker": "NG=F",  "unit": "USD/MMBtu"},
        {"label": "Heating Oil",     "ticker": "HO=F",  "unit": "USD/gal"},
        {"label": "Gasoline (RBOB)", "ticker": "RB=F",  "unit": "USD/gal"},
        {"label": "Brent Crude",     "ticker": "BZ=F",  "unit": "USD/bbl"},
    ],
    "Metals": [
        {"label": "Gold",     "ticker": "GC=F", "unit": "USD/oz"},
        {"label": "Silver",   "ticker": "SI=F", "unit": "USD/oz"},
        {"label": "Copper",   "ticker": "HG=F", "unit": "USD/lb"},
        {"label": "Platinum", "ticker": "PL=F", "unit": "USD/oz"},
        {"label": "Palladium","ticker": "PA=F", "unit": "USD/oz"},
    ],
    "Agriculture": [
        {"label": "Wheat",   "ticker": "ZW=F", "unit": "USD/bu"},
        {"label": "Corn",    "ticker": "ZC=F", "unit": "USD/bu"},
        {"label": "Soybeans","ticker": "ZS=F", "unit": "USD/bu"},
        {"label": "Sugar",   "ticker": "SB=F", "unit": "USD/lb"},
        {"label": "Coffee",  "ticker": "KC=F", "unit": "USD/lb"},
        {"label": "Cotton",  "ticker": "CT=F", "unit": "USD/lb"},
        {"label": "Cocoa",   "ticker": "CC=F", "unit": "USD/mt"},
    ],
    "Forex": [
        {"label": "EUR/USD", "ticker": "EURUSD=X", "unit": ""},
        {"label": "GBP/USD", "ticker": "GBPUSD=X", "unit": ""},
        {"label": "USD/JPY", "ticker": "JPY=X",    "unit": ""},
        {"label": "USD/CHF", "ticker": "CHF=X",    "unit": ""},
        {"label": "AUD/USD", "ticker": "AUDUSD=X", "unit": ""},
        {"label": "USD/CAD", "ticker": "CAD=X",    "unit": ""},
    ],
    "Crypto": [
        {"label": "Bitcoin",  "ticker": "BTC-USD", "unit": "USD"},
        {"label": "Ethereum", "ticker": "ETH-USD", "unit": "USD"},
        {"label": "Solana",   "ticker": "SOL-USD", "unit": "USD"},
        {"label": "BNB",      "ticker": "BNB-USD", "unit": "USD"},
    ],
    "Stocks & Indices": [
        {"label": "S&P 500",    "ticker": "^GSPC", "unit": ""},
        {"label": "NASDAQ",     "ticker": "^IXIC", "unit": ""},
        {"label": "Dow Jones",  "ticker": "^DJI",  "unit": ""},
        {"label": "Apple",      "ticker": "AAPL",  "unit": "USD"},
        {"label": "NVIDIA",     "ticker": "NVDA",  "unit": "USD"},
        {"label": "Tesla",      "ticker": "TSLA",  "unit": "USD"},
        {"label": "Microsoft",  "ticker": "MSFT",  "unit": "USD"},
        {"label": "Gold Miners","ticker": "GDX",   "unit": "USD"},
    ],
}

# ── ENSO / El Niño data ────────────────────────────────────────────────────────
_enso_cache = {"data": None, "fetched": None}

def get_enso_status():
    """Fetch latest ONI from NOAA and classify ENSO phase."""
    now = datetime.utcnow()
    if _enso_cache["data"] and _enso_cache["fetched"] and (now - _enso_cache["fetched"]).seconds < 3600 * 6:
        return _enso_cache["data"]
    try:
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        resp = requests.get(url, timeout=8)
        lines = [l for l in resp.text.strip().split("\n") if l.strip() and not l.startswith("SEAS")]
        last = lines[-1].split()
        oni = float(last[2])  # ANOM column
        season = last[0]
        if oni >= 1.5:
            phase, strength = "El Niño", "strong"
        elif oni >= 0.5:
            phase, strength = "El Niño", "moderate" if oni >= 1.0 else "weak"
        elif oni <= -1.5:
            phase, strength = "La Niña", "strong"
        elif oni <= -0.5:
            phase, strength = "La Niña", "moderate" if oni <= -1.0 else "weak"
        else:
            phase, strength = "Neutral", "neutral"
        result = {"phase": phase, "strength": strength, "oni": oni, "season": season}
    except Exception as e:
        logger.warning(f"ENSO fetch failed: {e}")
        result = {"phase": "Unknown", "strength": "unknown", "oni": None, "season": ""}
    _enso_cache["data"] = result
    _enso_cache["fetched"] = now
    return result


# ── Macro signal knowledge base ────────────────────────────────────────────────
def get_macro_signals(ticker: str, enso: dict, forecast_pct: float) -> list:
    """Return a list of contextual signal dicts for this instrument."""
    month = datetime.now().month
    signals = []

    # ── ENSO signals ──────────────────────────────────────────────────────────
    enso_phase = enso.get("phase", "Unknown")
    enso_str   = enso.get("strength", "")
    oni        = enso.get("oni")

    ENSO_EFFECTS = {
        # Energy
        "NG=F":  {"El Niño": ("bearish", "El Niño typically brings warmer winters to North America and Europe, reducing heating demand for natural gas. Historically, NG prices drop 15–25% during strong El Niño winters."),
                  "La Niña": ("bullish", "La Niña favors colder winters in key demand regions, historically pushing natural gas prices up 20–30% through the heating season.")},
        "HO=F":  {"El Niño": ("bearish", "Heating oil demand falls during El Niño's warmer winters. Refiners and traders typically price in lower seasonal demand premium."),
                  "La Niña": ("bullish", "Colder La Niña winters drive heating oil demand higher. Distillate inventories draw sharply in Jan–Mar.")},
        "CL=F":  {"El Niño": ("neutral", "El Niño has mixed effects on crude — lower heating demand but higher air-conditioning load. Net impact is usually modest."),
                  "La Niña": ("bearish", "La Niña can reduce global refinery throughput and slow emerging market demand, creating modest downward pressure on crude.")},
        "ZW=F":  {"El Niño": ("bearish", "El Niño brings above-average rainfall to key wheat regions (Argentina, Australia), boosting yields and pressuring prices."),
                  "La Niña": ("bullish", "La Niña triggers drought conditions in wheat-growing regions, historically pushing prices 10–20% higher mid-season.")},
        "ZC=F":  {"El Niño": ("bearish", "US Corn Belt tends to see favorable moisture conditions during El Niño, supporting larger yields and lower prices."),
                  "La Niña": ("bullish", "La Niña drought stress on the US Corn Belt historically raises corn prices 15–30% from planting to harvest season.")},
        "ZS=F":  {"El Niño": ("mixed",   "El Niño is generally positive for South American soybean production but can stress US crops. Net effect varies by season."),
                  "La Niña": ("bullish", "La Niña strongly impacts soybean supply from Argentina and Brazil (world's top exporters), historically driving 20–40% price spikes.")},
        "KC=F":  {"El Niño": ("bearish", "El Niño brings excess rainfall to key coffee origins (Colombia, Central America), increasing supply and pressing prices lower."),
                  "La Niña": ("bullish", "La Niña drought in Brazil (world's top coffee producer) is one of the most reliable coffee price catalysts historically.")},
        "CC=F":  {"El Niño": ("bullish", "El Niño triggers severe drought in West Africa (Ivory Coast + Ghana = ~65% of global supply), historically pushing cocoa prices up 20–50%. The 2023–24 El Niño drove cocoa to all-time highs above $10,000/mt."),
                  "La Niña": ("bearish", "La Niña typically brings above-average rainfall to West African cocoa regions, supporting crop yields and easing supply concerns.")},
        "GC=F":  {"El Niño": ("bullish", "El Niño-driven economic disruption in commodity-dependent emerging markets historically increases safe-haven gold demand."),
                  "La Niña": ("neutral", "La Niña's inflationary pressure on food prices can support gold as a hedge, but the effect is secondary to monetary policy.")},
        "SI=F":  {"El Niño": ("neutral", "Silver has limited direct ENSO sensitivity, though industrial demand softness during El Niño-related slowdowns can weigh on prices."),
                  "La Niña": ("neutral", "La Niña has modest direct impact on silver. Agricultural inflation can indirectly support precious metals broadly.")},
        "SB=F":  {"El Niño": ("bullish", "El Niño causes drought in key sugar-producing regions (India, Thailand, Australia), historically spiking sugar 20–40%."),
                  "La Niña": ("bearish", "La Niña brings excess rainfall to Brazil's sugarcane region, boosting output and pressing prices lower.")},
    }
    if ticker in ENSO_EFFECTS and enso_phase in ("El Niño", "La Niña"):
        bias, text = ENSO_EFFECTS[ticker].get(enso_phase, ("neutral", ""))
        if text:
            signals.append({
                "type": "enso",
                "icon": "🌊",
                "title": f"{enso_str.title()} {enso_phase} Active (ONI: {oni:+.2f})",
                "text": text,
                "bias": bias,
            })

    # ── Seasonal signals ───────────────────────────────────────────────────────
    SEASONAL = {
        "NG=F": {
            (11,12,1,2): ("bearish" if enso_phase == "El Niño" else "bullish",
                          "Winter heating season (Nov–Feb) is peak demand for natural gas. "
                          + ("El Niño's warmth reduces this seasonal premium." if enso_phase == "El Niño"
                             else "Cold snaps during this window can cause sharp short-term spikes.")),
            (6,7,8):     ("bullish", "Summer heat drives air-conditioning gas demand for power generation. "
                                     "Hot summers historically lift Henry Hub prices 10–15%."),
            (3,4,5):     ("bearish", "Spring shoulder season — lowest gas demand period. Prices typically make seasonal lows in April–May."),
        },
        "HO=F": {
            (10,11,12,1,2,3): ("bullish", "Heating season (Oct–Mar) is the primary demand driver for heating oil. "
                                          "Prices typically peak in January–February."),
            (5,6,7,8,9):      ("bearish", "Off-season for heating oil. Demand is minimal; refiners often shift capacity to gasoline."),
        },
        "CL=F": {
            (5,6,7,8,9): ("bullish", "US summer driving season (Memorial Day to Labor Day) lifts gasoline and crude demand historically 3–8%."),
            (10,11,12):  ("bearish", "Post-driving season crude inventory build typically weighs on prices Oct–Dec."),
        },
        "GC=F": {
            (1,2):   ("bullish", "January–February historically sees gold buying ahead of Indian wedding season and Chinese New Year jewelry demand."),
            (8,9):   ("bullish", "Late summer is traditionally a strong seasonal window for gold — Indian festival/wedding season buying accelerates."),
            (6,7):   ("bearish", "June–July tends to be gold's weakest seasonal period — 'summer doldrums' with thin trading and low physical demand."),
        },
        "SI=F": {
            (1,2,3):    ("bullish", "Silver benefits from the same Q1 precious metals seasonal bid as gold, amplified by industrial restocking demand."),
            (6,7,8,9):  ("mixed",   "Silver's industrial demand (solar panels, electronics) picks up in H2 but is highly sensitive to global manufacturing PMIs."),
        },
        "ZW=F": {
            (3,4,5):  ("volatile", "US winter wheat crop condition reports in spring create the most volatile price window for wheat annually."),
            (6,7):    ("bearish",  "Northern hemisphere harvest pressure typically pushes wheat prices to seasonal lows in June–July."),
            (10,11):  ("bullish",  "Southern hemisphere harvest uncertainty and northern hemisphere new-crop planting concerns support prices in Oct–Nov."),
        },
        "ZC=F": {
            (6,7):   ("volatile", "Pollination period (June–July) is the most weather-sensitive and volatile time for corn prices. Any heat/drought fears spike prices."),
            (8,9):   ("bearish",  "US harvest pressure builds in Aug–Sep, historically the weakest seasonal window for corn prices."),
            (3,4,5): ("bullish",  "Pre-planting and early growing season uncertainties lift corn prices in spring as funds build long positions."),
        },
        "SB=F": {
            (2,3,4):  ("bearish", "Brazilian Center-South crush season begins in April. Pre-harvest supply expectations typically pressure sugar Feb–Apr."),
            (9,10):   ("bullish", "Northern hemisphere off-season + monsoon uncertainty in Asia historically supports sugar prices in Sept–Oct."),
        },
        "KC=F": {
            (5,6,7):  ("volatile", "Brazilian coffee flowering and early cherry development in May–July creates the highest weather-risk window for prices."),
            (9,10):   ("bullish",  "Pre-harvest rally in coffee as roasters build inventory ahead of Q4 demand peak historically pushes prices higher in Sept–Oct."),
        },
        "CC=F": {
            (10,11,12,1): ("volatile", "Main crop harvest in West Africa runs Oct–Mar. Any mid-season weather stress (dry Harmattan winds, Black Pod disease from excess rain) during this window creates sharp price swings."),
            (4,5,6):      ("bullish",  "The mid-crop (April–June) is smaller and less reliable than the main crop. Disappointing mid-crop estimates historically trigger rallies as traders price in annual supply deficits."),
            (7,8,9):      ("bearish",  "Pre-harvest season — traders anticipate the upcoming main crop. Prices often drift lower in anticipation of new supply arriving Oct–Nov."),
        },
        "BTC-USD": {
            (1,2,3):  ("bullish", "Q1 historically Bitcoin's strongest quarter — post-halving accumulation, tax-season inflows, and institutional rebalancing."),
            (6,7,8):  ("bearish", "Summer crypto bear windows are common — lower retail activity and miner sell pressure after reward seasons."),
            (10,11,12): ("bullish", "Q4 'crypto autumn rally' — historically the most consistent strong seasonal window for Bitcoin."),
        },
        "^GSPC": {
            (1,2,3):   ("bullish", "January effect and Q1 earnings season historically support equities. Institutional rebalancing into stocks is common in Jan."),
            (5,6,7,8,9): ("mixed", "'Sell in May' seasonality — the May–October period historically underperforms. Vol tends to peak in August–September."),
            (10,11,12): ("bullish", "Q4 seasonality is the strongest for US equities — Santa Claus rally, window dressing, and holiday spending optimism."),
        },
    }

    if ticker in SEASONAL:
        for months, (bias, text) in SEASONAL[ticker].items():
            if month in months:
                month_name = datetime.now().strftime("%B")
                signals.append({
                    "type": "seasonal",
                    "icon": "📅",
                    "title": f"Seasonal Pattern — {month_name}",
                    "text": text,
                    "bias": bias,
                })
                break

    # ── Geopolitical / structural signals ─────────────────────────────────────
    GEO = {
        "CL=F":  {"icon": "🛢️", "title": "OPEC+ Production Policy",
                  "text": "OPEC+ controls ~40% of global supply. Voluntary cuts (currently in place) set a price floor. Watch monthly OPEC meetings — surprise cut extensions are the biggest upside risk for crude."},
        "BZ=F":  {"icon": "🛢️", "title": "Brent — Global Benchmark",
                  "text": "Brent prices European, African, and Middle Eastern crude flows. The Brent-WTI spread widens during geopolitical disruptions in the Middle East or shipping route issues (Suez, Strait of Hormuz)."},
        "GC=F":  {"icon": "🏦", "title": "Fed Rate Cycle & Dollar",
                  "text": "Gold moves inversely to real US interest rates and the US Dollar. Rate cut cycles historically trigger gold bull runs. Central bank gold buying (especially China and emerging markets) has been a structural bullish force since 2022."},
        "SI=F":  {"icon": "☀️", "title": "Solar & Green Energy Demand",
                  "text": "Silver is critical for solar panel manufacturing (~10% of global demand). The green energy transition is a long-term structural demand driver. Watch global solar installation targets for multi-year context."},
        "HG=F":  {"icon": "⚡", "title": "Electrification & China Demand",
                  "text": "Copper is the metal of electrification — EVs, grid infrastructure, and data centers all require significant copper. China accounts for ~55% of global demand. Chinese property sector health is the #1 short-term price driver."},
        "EURUSD=X": {"icon": "🇪🇺", "title": "ECB vs. Fed Divergence",
                     "text": "EUR/USD is primarily driven by ECB vs. Federal Reserve rate expectations. When the Fed cuts while ECB holds (or vice versa), the pair moves sharply. Energy price shocks disproportionately hit the Eurozone."},
        "JPY=X":    {"icon": "🇯🇵", "title": "Bank of Japan Policy",
                     "text": "USD/JPY is heavily influenced by BoJ yield curve control (YCC) policy. BoJ normalization (rate hikes) strengthens the yen sharply. Carry trade unwinds — when USD/JPY falls fast — often trigger global market volatility."},
        "BTC-USD":  {"icon": "₿",   "title": "Halving Cycle & ETF Flows",
                     "text": "Bitcoin's 4-year halving cycle (last halving: April 2024) historically precedes 12–18 month bull markets. Spot Bitcoin ETF inflows are the primary new demand driver since Jan 2024. Watch daily ETF flow data as a leading indicator."},
        "^GSPC":    {"icon": "📊", "title": "Fed Policy & Earnings Cycle",
                     "text": "The S&P 500 is primarily driven by Federal Reserve policy (rate cuts = bullish, hikes = bearish) and corporate earnings growth. The current AI capex cycle is a major earnings tailwind for the index."},
    }
    if ticker in GEO:
        g = GEO[ticker]
        signals.append({"type": "geo", "icon": g["icon"], "title": g["title"], "text": g["text"], "bias": "context"})

    return signals


def generate_summary(ticker: str, label: str, forecast_pct: float, pred_bars: int,
                      interval: str, enso: dict, signals: list) -> dict:
    """Generate a plain-English forecast summary."""
    direction = "higher" if forecast_pct > 0 else "lower"
    abs_pct   = abs(forecast_pct)
    interval_label = {"1d": "day", "1h": "hour", "4h": "4-hour bar"}[interval]
    timeframe  = f"{pred_bars} {interval_label}s"

    if abs_pct < 1:
        magnitude = "relatively flat"
    elif abs_pct < 3:
        magnitude = "modestly"
    elif abs_pct < 8:
        magnitude = "noticeably"
    elif abs_pct < 15:
        magnitude = "significantly"
    else:
        magnitude = "sharply"

    # Count bullish vs bearish signals
    bull = sum(1 for s in signals if s.get("bias") in ("bullish",))
    bear = sum(1 for s in signals if s.get("bias") in ("bearish",))

    if bull > bear:
        alignment = f"The broader macro and seasonal context currently leans <strong>supportive</strong> for {label}."
    elif bear > bull:
        alignment = f"Macro and seasonal factors present some <strong>headwinds</strong> for {label} at this time."
    else:
        alignment = f"The macro and seasonal backdrop for {label} is currently <strong>mixed</strong>."

    headline = (
        f"Kronos forecasts <strong>{label}</strong> moving <strong>{magnitude} {direction}</strong> "
        f"over the next {timeframe} "
        f"({forecast_pct:+.1f}%)."
    )

    enso_note = ""
    phase = enso.get("phase", "Unknown")
    if phase in ("El Niño", "La Niña"):
        enso_note = (
            f" A <strong>{enso.get('strength', '')} {phase}</strong> event is currently active "
            f"(ONI index: {enso.get('oni', 0):+.2f}), which has historically influenced {label} prices."
        )

    return {
        "headline": headline,
        "alignment": alignment + enso_note,
        "disclaimer": "This is a probabilistic AI forecast, not financial advice. Markets can move against any model prediction.",
    }


# ── Model loading ──────────────────────────────────────────────────────────────
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
    period_map = {"1d": f"{max(bars * 2, 500)}d", "1h": "730d", "4h": "730d"}
    period = period_map.get(interval, "730d")
    fetch_interval = "1h" if interval == "4h" else interval
    df = yf.download(ticker, period=period, interval=fetch_interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "close"})
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if interval == "4h":
        df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    return df.tail(bars)


def make_future_timestamps(last_ts, n: int, freq_td: timedelta):
    if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is not None:
        last_ts = last_ts.replace(tzinfo=None)
    return pd.Series([pd.Timestamp(last_ts) + freq_td * (i + 1) for i in range(n)])


def infer_freq(ts_index: pd.DatetimeIndex) -> timedelta:
    if len(ts_index) < 2:
        return timedelta(days=1)
    diffs = ts_index[1:] - ts_index[:-1]
    return pd.Series(diffs).median().to_pytimedelta()


def ticker_to_label(ticker: str) -> str:
    for cat in INSTRUMENTS.values():
        for item in cat:
            if item["ticker"] == ticker:
                return item["label"]
    return ticker


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", instruments=INSTRUMENTS)


@app.route("/api/instruments")
def api_instruments():
    return jsonify(INSTRUMENTS)


@app.route("/api/price")
def api_price():
    ticker = request.args.get("ticker", "GC=F").upper()
    try:
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


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    body = request.get_json(force=True)
    ticker       = body.get("ticker", "GC=F").upper()
    interval     = body.get("interval", "1d")
    context_bars = int(body.get("context_bars", 300))
    pred_bars    = int(body.get("pred_bars", 30))
    model_key    = body.get("model", "mini")
    temperature  = float(body.get("temperature", 1.0))
    top_p        = float(body.get("top_p", 0.9))
    sample_count = int(body.get("sample_count", 3))

    if model_key not in MODEL_CONFIGS:
        return jsonify({"error": f"Unknown model '{model_key}'"}), 400

    try:
        df = fetch_ohlcv(ticker, interval, context_bars + 50)
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {e}"}), 500

    if len(df) < 30:
        return jsonify({"error": f"Not enough data for {ticker} ({len(df)} bars)"}), 400

    df = df.tail(context_bars)

    x_ts = pd.Series(pd.to_datetime(df.index).tz_localize(None)
                     if df.index.tz is not None else pd.to_datetime(df.index))
    freq  = infer_freq(pd.DatetimeIndex(x_ts))
    y_ts  = make_future_timestamps(x_ts.iloc[-1], pred_bars, freq)

    try:
        predictor = get_predictor(model_key)
        pred_df = predictor.predict(df, x_ts, y_ts, pred_len=pred_bars,
                                    T=temperature, top_p=top_p,
                                    sample_count=sample_count, verbose=False)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Forecast summary stats
    last_close  = float(df["close"].iloc[-1])
    pred_close  = float(pred_df["close"].iloc[-1])
    forecast_pct = (pred_close - last_close) / last_close * 100

    # Macro context
    enso    = get_enso_status()
    label   = ticker_to_label(ticker)
    signals = get_macro_signals(ticker, enso, forecast_pct)
    summary = generate_summary(ticker, label, forecast_pct, pred_bars, interval, enso, signals)

    def candles(d):
        ts = pd.DatetimeIndex(d.index)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        return {
            "x":     [str(t) for t in ts],
            "open":  d["open"].tolist(),
            "high":  d["high"].tolist(),
            "low":   d["low"].tolist(),
            "close": d["close"].tolist(),
        }

    return jsonify({
        "ticker":      ticker,
        "label":       label,
        "interval":    interval,
        "model":       model_key,
        "historical":  candles(df),
        "forecast":    candles(pred_df),
        "forecast_pct": forecast_pct,
        "last_close":  last_close,
        "signals":     signals,
        "summary":     summary,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
