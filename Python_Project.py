import json
import math
import sqlite3
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

DB = "fmcg_records.db"

MIN_MONTHS, MAX_MONTHS = 2, 36
MIN_TRAIN = 3                 # backtest starts after this many months
INTERVAL_Z = 1.645            # 90% prediction interval
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#f59e0b", "#16a34a", "#dc2626", "#64748b"

PRODUCT_TYPES = ["Food Product", "Beverage Product", "Personal Care Product",
                 "Household Product", "Dairy Product", "Packaged Product"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


# ======================================================================
#  DATABASE (SQLite)
# ======================================================================
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS records_v2(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product TEXT, product_type TEXT,
        months TEXT, sales TEXT,
        current_stock INTEGER,
        predicted INTEGER,
        safety_stock INTEGER,
        required_stock INTEGER,
        order_qty INTEGER,
        accuracy REAL,
        model_info TEXT,
        created TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    con.commit()
    con.close()


def save_record(row):
    con = sqlite3.connect(DB)
    con.execute("""INSERT INTO records_v2
        (product,product_type,months,sales,current_stock,predicted,safety_stock,
         required_stock,order_qty,accuracy,model_info)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)""", row)
    con.commit()
    con.close()


def get_records(limit=10):
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT id,product,product_type,months,predicted,safety_stock,"
                       "required_stock,order_qty,accuracy,model_info,created "
                       "FROM records_v2 ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return rows


def count_records():
    con = sqlite3.connect(DB)
    n = con.execute("SELECT COUNT(*) FROM records_v2").fetchone()[0]
    con.close()
    return n


def delete_record(rec_id):
    con = sqlite3.connect(DB)
    cur = con.execute("DELETE FROM records_v2 WHERE id=?", (rec_id,))
    con.commit()
    deleted = cur.rowcount
    con.close()
    return deleted


def delete_all_records():
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM records_v2")
    con.execute("DROP TABLE IF EXISTS records")      # old version table (if present)
    con.execute("DELETE FROM sqlite_sequence WHERE name='records_v2'")
    con.commit()
    con.close()


# ======================================================================
#  SELF-LEARNING FORECAST ENGINE
#  Several models compete. Each model is tested with walk-forward
#  backtesting (train on past months -> predict the next month -> compare
#  with what really happened). The best models get more weight.
# ======================================================================
def _clip(v, y):
    return float(min(max(v, 0.0), 2.0 * max(y) + 1.0))


def f_naive(y):
    return y[-1]


def f_mean(y):
    return float(np.mean(y))


def f_ma3(y):
    return float(np.mean(y[-3:]))


def _line(y):
    x = np.arange(len(y)).reshape(-1, 1)
    return LinearRegression().fit(x, y)


def f_linear(y):
    m = _line(np.asarray(y, dtype=float))
    return float(m.predict([[len(y)]])[0])


def f_recent_linear(y):
    """Linear regression on the last 6 months only (adapts faster to change)."""
    y = np.asarray(y, dtype=float)[-6:]
    if len(y) < 3:
        return float(y[-1])
    m = _line(y)
    return float(m.predict([[len(y)]])[0])


def f_holt(y):
    """Damped Holt (level + trend). Smoothing values are learned from the data."""
    y = np.asarray(y, dtype=float)
    best_sse, best_pred = None, y[-1]
    for alpha in (0.2, 0.4, 0.6, 0.8):
        for beta in (0.1, 0.3, 0.5):
            for phi in (0.9, 1.0):
                level, trend, sse = y[0], y[1] - y[0], 0.0
                for v in y[1:]:
                    pred = level + phi * trend
                    sse += (v - pred) ** 2
                    new = alpha * v + (1 - alpha) * pred
                    trend = beta * (new - level) + (1 - beta) * phi * trend
                    level = new
                if best_sse is None or sse < best_sse:
                    best_sse, best_pred = sse, level + phi * trend
    return float(best_pred)


def f_seasonal(y, period=12):
    """Trend + yearly seasonality (needs at least 2 full years)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * period:
        raise ValueError("not enough data")
    x = np.arange(n).reshape(-1, 1)
    resid = y - LinearRegression().fit(x, y).predict(x)
    idx = np.array([resid[p::period].mean() for p in range(period)])
    idx -= idx.mean()
    deseason = y - idx[np.arange(n) % period]
    m = LinearRegression().fit(x, deseason)
    return float(m.predict([[n]])[0] + idx[n % period])


MODELS = {
    "Linear Regression": f_linear,
    "Recent Linear (6m)": f_recent_linear,
    "Holt Trend (learned)": f_holt,
    "Seasonal + Trend": f_seasonal,
}


def run_models(train):
    """Return {model name: next-month forecast} for every model that can run."""
    out = {}
    for name, fn in MODELS.items():
        try:
            if name in ("Linear Regression", "Recent Linear (6m)", "Holt Trend (learned)") and len(train) < 3:
                continue
            out[name] = _clip(fn(list(train)), train)
        except Exception:
            continue
    return out


def learn_and_forecast(y):
    """Learn from the data: backtest every model, weight them by accuracy, blend."""
    y = [float(v) for v in y]
    n = len(y)

    # Very little data: no backtest possible
    if n < MIN_TRAIN + 1:
        preds = run_models(y)
        lin = preds.get("Linear Regression", float(np.mean(y)))
        return {"pred": float(lin), "sigma": None, "accuracy": None,
                "table": pd.DataFrame({"Model": ["Linear Regression"], "Avg Error": [np.nan], "Weight %": [100.0]}),
                "backtest": [np.nan] * n, "info": "Linear Regression (scikit-learn)"}

    # Walk-forward backtest
    errors = {k: [] for k in MODELS}
    fold_preds = {k: [] for k in MODELS}
    actuals = []
    for t in range(MIN_TRAIN, n):
        train, actual = y[:t], y[t]
        preds = run_models(train)
        actuals.append(actual)
        for k in MODELS:
            p = preds.get(k, None)
            if p is None:
                fold_preds[k].append(np.nan)
                continue
            fold_preds[k].append(p)
            errors[k].append(abs(p - actual))

    # Models that could run on every fold are ranked by mean abs error
    mae = {k: float(np.mean(e)) for k, e in errors.items() if len(e) == len(actuals)}
    ranked = sorted(mae, key=mae.get)[:3]                    # top 3 models
    w = np.array([1.0 / (mae[k] + 1e-6) for k in ranked])
    w = w / w.sum()
    weights = dict(zip(ranked, w))

    # Ensemble backtest -> honest error estimate
    ens_bt = np.zeros(len(actuals))
    for k, wk in weights.items():
        ens_bt += wk * np.array(fold_preds[k])
    err = ens_bt - np.array(actuals)
    sigma = float(np.sqrt(np.mean(err ** 2)))
    wape = float(np.abs(err).sum() / max(np.sum(actuals), 1.0))
    accuracy = max(0.0, 100.0 * (1.0 - wape))

    # Final forecast using ALL data
    final = run_models(y)
    pred = float(sum(weights[k] * final[k] for k in ranked if k in final) /
                 sum(weights[k] for k in ranked if k in final))

    table = pd.DataFrame({
        "Model": list(mae),
        "Avg Error": [round(mae[k], 1) for k in mae],
        "Weight %": [round(100 * weights.get(k, 0.0), 1) for k in mae],
    }).sort_values("Avg Error").reset_index(drop=True)

    backtest = [np.nan] * MIN_TRAIN + [float(v) for v in ens_bt]
    info = " + ".join(f"{k} ({weights[k] * 100:.0f}%)" for k in ranked)
    return {"pred": pred, "sigma": sigma, "accuracy": accuracy,
            "table": table, "backtest": backtest, "info": info}


# ======================================================================
#  LOGIC
# ======================================================================
def analyze(product, ptype, stock, service, months, sales):
    d = {}
    for i, (m, s) in enumerate(zip(months, sales), start=1):
        m = str(m).strip() if m is not None else ""
        if not m or m == "nan":
            raise ValueError(f"Enter the name of month {i}.")
        if m in d:
            raise ValueError(f"Month '{m}' is entered twice.")
        if s is None or pd.isna(s) or s < 0:
            raise ValueError(f"Sales for '{m}' must be a number, 0 or more.")
        d[m] = int(s)

    df = pd.DataFrame(list(d.items()), columns=["Months", "Sales"])
    n = len(df)
    avg = float(df["Sales"].mean())

    res = learn_and_forecast(df["Sales"].tolist())
    predicted = int(round(res["pred"]))
    df["Backtest"] = np.round(res["backtest"], 0)

    # Safety stock learned from the model's own past errors
    if res["sigma"] is not None:
        z = NormalDist().inv_cdf(service)
        safety = math.ceil(z * res["sigma"])
        low = max(0, int(round(res["pred"] - INTERVAL_Z * res["sigma"])))
        high = int(round(res["pred"] + INTERVAL_Z * res["sigma"]))
    else:
        safety = math.ceil(predicted * 0.20)      # fallback when data is too short
        low, high = predicted, predicted
    required = predicted + safety
    order = max(0, required - stock)

    slope = float(_line(df["Sales"].to_numpy(dtype=float)).coef_[0]) if n >= 2 else 0.0
    best = df.loc[df["Sales"].idxmax()]

    save_record((product, ptype, json.dumps(df["Months"].tolist()),
                 json.dumps(df["Sales"].tolist()), stock, predicted, safety, required, order,
                 None if res["accuracy"] is None else round(res["accuracy"], 1), res["info"]))

    return {
        "product": product, "ptype": ptype, "df": df, "n": n,
        "avg": round(avg, 2), "predicted": predicted, "low": low, "high": high,
        "slope": round(slope, 1), "accuracy": res["accuracy"], "info": res["info"],
        "model_table": res["table"], "service": service,
        "best": f"{best['Months']} ({int(best['Sales']):,})",
        "safety": safety, "required": required, "stock": stock, "order": order,
        "ok": order == 0,
        "status": "Stock Sufficient" if order == 0 else "Stock Order Required",
    }


# ======================================================================
#  CHARTS
# ======================================================================
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=1)
    ax.set_axisbelow(True)


def _month_ticks(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 7 else 0,
                       ha="right" if len(labels) > 7 else "center")


def line_chart(r):
    df, n = r["df"], r["n"]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, df["Sales"], marker="o", markersize=8, linewidth=3, color=BLUE, label="Actual sales")
    ax.fill_between(x, df["Sales"], color=BLUE, alpha=.10)
    if df["Backtest"].notna().any():
        ax.plot(x, df["Backtest"], linestyle="--", linewidth=2, color=GREY, label="Model's past predictions")
    ax.plot([n - 1, n], [df["Sales"].iloc[-1], r["predicted"]], linestyle=":", linewidth=3, color=ORANGE)
    ax.scatter([n], [r["predicted"]], s=130, color=ORANGE, zorder=5, label="Forecast")
    if r["high"] > r["low"]:
        ax.errorbar([n], [r["predicted"]], yerr=[[r["predicted"] - r["low"]], [r["high"] - r["predicted"]]],
                    color=ORANGE, capsize=6, linewidth=2, zorder=4, label="90% range")
    if n <= 12:
        for xi, v in zip(x, df["Sales"]):
            ax.annotate(f"{v:,}", (xi, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, fontweight="bold")
    ax.annotate(f"{r['predicted']:,}", (n, r["predicted"]), textcoords="offset points",
                xytext=(14, -4), ha="left", fontsize=9, fontweight="bold")
    _month_ticks(ax, df["Months"].tolist() + ["Next"])
    ax.set_ylabel("Units sold")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return fig


def bar_chart(r):
    df, n = r["df"], r["n"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(range(n), df["Sales"], color=BLUE, width=.6, label="Monthly sales")
    ax.bar([n], [r["predicted"]], color=ORANGE, width=.6, label="Forecast")
    ax.axhline(r["avg"], color=GREY, linestyle="--", linewidth=2, label=f"Average {r['avg']:,}")
    if n <= 12:
        ax.bar_label(bars, labels=[f"{v:,}" for v in df["Sales"]], padding=3, fontsize=9, fontweight="bold")
        ax.annotate(f"{r['predicted']:,}", (n, r["predicted"]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
    _month_ticks(ax, df["Months"].tolist() + ["Next"])
    ax.set_ylabel("Units")
    ax.set_ylim(0, max(max(df["Sales"]), r["high"], r["avg"], 1) * 1.3)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    _style(ax)
    return fig


def stock_chart(r):
    labels = ["Current stock", "Safety stock", "Predicted sales", "Required stock"]
    values = [r["stock"], r["safety"], r["predicted"], r["required"]]
    colours = [GREEN if r["ok"] else RED, GREY, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(labels, values, color=colours, height=.55)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(values + [1]) * 1.15)
    ax.grid(axis="x", color="#e2e8f0")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return fig


# ======================================================================
#  UI
# ======================================================================
def show_results(r):
    df = r["df"]
    st.header(f"Analysis Dashboard: {r['product']}")

    c = st.columns(4)
    c[0].metric("Average Sales", f"{r['avg']:,}")
    c[1].metric("Predicted Next Month", f"{r['predicted']:,}")
    c[2].metric("Likely Range (90%)", f"{r['low']:,} - {r['high']:,}")
    c[3].metric("Model Accuracy", "N/A" if r["accuracy"] is None else f"{r['accuracy']:.1f}%")
    c = st.columns(4)
    c[0].metric("Safety Stock", f"{r['safety']:,}")
    c[1].metric("Required Stock", f"{r['required']:,}")
    c[2].metric("Current Stock", f"{r['stock']:,}")
    c[3].metric("Stock To Order", f"{r['order']:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Sales Forecast Line Chart")
        st.pyplot(line_chart(r))
    with right:
        st.subheader("Monthly Sales Bar Chart")
        st.pyplot(bar_chart(r))

    left, right = st.columns(2)
    with left:
        st.subheader("Stock Position")
        st.pyplot(stock_chart(r))
    with right:
        st.subheader("Prediction Table")
        table = df.copy()
        table.loc[len(table)] = ["Next Month", r["predicted"], r["predicted"]]
        st.dataframe(table, hide_index=True, width='stretch')

    st.subheader("How the model learned from your data")
    if r["accuracy"] is None:
        st.warning("Data is very short (under 4 months), so the model could not test itself. "
                   "Add more months for a reliable forecast.")
    else:
        st.caption("Each model was tested on your own past months (train on earlier months, "
                   "predict the next one). Lower 'Avg Error' is better; the best models get more weight.")
    st.dataframe(r["model_table"], hide_index=True, width='stretch')
    if r["n"] < 12:
        st.info("Tip: 12+ months of data gives a better forecast. With 24+ months the model also "
                "learns yearly seasonality.")

    st.subheader("Stock Recommendation")
    msg = (f"**{r['product']}** ({r['ptype']}) averaged **{r['avg']:,}** units over {r['n']} months. "
           f"Best month: **{r['best']}**. Sales trend: **{r['slope']:+,}** units per month.\n\n"
           f"Predicted next month sales: **{r['predicted']:,}** units "
           f"(likely range {r['low']:,} to {r['high']:,}). Model used: {r['info']}.\n\n"
           f"Safety stock for {int(r['service'] * 100)}% service level is **{r['safety']:,}**, "
           f"so required stock is **{r['required']:,}** units. "
           f"Current stock is **{r['stock']:,}**, so order approximately **{r['order']:,}** units.\n\n"
           f"Status: **{r['status']}**")
    (st.success if r["ok"] else st.error)(msg)


def show_records():
    st.header("Recent Database Records")
    rows = get_records()
    if not rows:
        st.info("No records yet. Run an analysis to save the first one.")
        return
    data = [{"ID": r[0], "Product": r[1], "Type": r[2],
             "Months": ", ".join(json.loads(r[3])),
             "Predicted": r[4], "Safety": r[5], "Required Stock": r[6], "Order Qty": r[7],
             "Accuracy %": r[8], "Saved On": r[10]} for r in rows]
    st.dataframe(pd.DataFrame(data), hide_index=True, width='stretch')


def delete_section():
    total = count_records()
    with st.expander(f"🗑️ Delete old data ({total} records saved)"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Delete one record**")
            rec_id = st.number_input("Record ID", min_value=1, value=1, step=1, key="del_id")
            if st.button("Delete this record"):
                if delete_record(int(rec_id)):
                    st.success(f"Record {int(rec_id)} deleted.")
                    st.rerun()
                else:
                    st.warning("No record with that ID.")

        with col2:
            st.markdown("**Delete ALL records**")
            sure = st.checkbox("Yes, I want to delete all old data", key="del_confirm")
            if st.button("Delete all data", type="primary", disabled=not sure):
                delete_all_records()
                init_db()
                st.session_state.pop("result", None)
                st.success("All old data deleted.")
                st.rerun()


def main():
    st.set_page_config(page_title="FMCG Stock Prediction", page_icon="📦", layout="wide")
    init_db()

    st.title("📦 FMCG Stock Prediction Dashboard")
    st.caption("Self-learning forecast: the app tests several models on your own data "
               "and blends the best ones.")

    # ---------------- Step 1 ----------------
    st.header("Step 1: Product Details")
    with st.form("setup_form"):
        c1, c2 = st.columns(2)
        product = c1.text_input("Product Name", placeholder="Example: Soap, Shampoo, Biscuit")
        ptype = c2.selectbox("Product Type", PRODUCT_TYPES)
        n = c1.number_input("Number of months", MIN_MONTHS, MAX_MONTHS, 12, step=1)
        stock = c2.number_input("Current Available Stock", min_value=0, value=0, step=1)
        service = c1.selectbox("Service level (chance of NOT running out)",
                               [0.90, 0.95, 0.99], index=1, format_func=lambda v: f"{int(v * 100)}%")
        if st.form_submit_button("Next: Enter Monthly Sales"):
            if not product.strip():
                st.error("Enter a product name.")
            else:
                st.session_state.setup = dict(product=product.strip(), ptype=ptype,
                                              n=int(n), stock=int(stock), service=float(service))
                st.session_state.pop("result", None)

    # ---------------- Step 2 ----------------
    setup = st.session_state.get("setup")
    if setup:
        st.header("Step 2: Monthly Sales")
        st.write(f"**{setup['product']}** ({setup['ptype']}) | Months: **{setup['n']}** "
                 f"| Current stock: **{setup['stock']:,}**")

        defaults = pd.DataFrame({
            "Month": [MONTH_NAMES[i % 12] if i < 12 else f"{MONTH_NAMES[i % 12]} (Y{i // 12 + 1})"
                      for i in range(setup["n"])],
            "Sales": [0] * setup["n"],
        })
        edited = st.data_editor(
            defaults, hide_index=True, width='stretch',
            key=f"editor_{setup['product']}_{setup['n']}",
            column_config={"Sales": st.column_config.NumberColumn(min_value=0, step=1, format="%d")},
        )

        if st.button("Analyze and Save", type="primary"):
            try:
                st.session_state.result = analyze(
                    setup["product"], setup["ptype"], setup["stock"], setup["service"],
                    edited["Month"].tolist(), edited["Sales"].tolist())
            except ValueError as e:
                st.session_state.pop("result", None)
                st.error(str(e))

    if "result" in st.session_state:
        st.divider()
        show_results(st.session_state.result)

    st.divider()
    show_records()
    delete_section()


main()