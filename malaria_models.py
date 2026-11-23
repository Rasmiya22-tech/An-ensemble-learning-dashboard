
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 42
NOTES = []

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    NOTES.append("xgboost not installed: XGBoost skipped.")
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
    NOTES.append("lightgbm not installed: LightGBM skipped.")
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
    NOTES.append("catboost not installed: CatBoost skipped.")
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None
    NOTES.append("statsmodels not installed: SARIMA and the hybrid skipped.")

ALL_MODELS = ["Random Forest", "Bagging (trees)", "XGBoost", "LightGBM", "CatBoost",
              "Gradient Boosting", "Stacking", "RF + ANN", "SARIMA", "SARIMA + RF"]
GROUP = {"Naive (next = this month)": "Baseline", "Seasonal naive (same month last year)": "Baseline",
         "Random Forest": "Bagging", "Bagging (trees)": "Bagging", "XGBoost": "Boosting",
         "LightGBM": "Boosting", "CatBoost": "Boosting", "Gradient Boosting": "Boosting",
         "Stacking": "Stacking", "RF + ANN": "Hybrid", "SARIMA": "Time series",
         "SARIMA + RF": "Hybrid"}


# ----------------------------------------------------------------- data
def load(path="srilanka_malaria_monthly_2000_2024.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["district", "date"]).reset_index(drop=True)


def national(df):
    """National monthly totals (sum of districts) plus mean weather."""
    agg = df.groupby("date").agg(
        indigenous_cases=("indigenous_cases", "sum"), imported_cases=("imported_cases", "sum"),
        total_cases=("total_cases", "sum"), pf_cases=("pf_cases", "sum"), pv_cases=("pv_cases", "sum"),
        deaths=("deaths", "sum"), blood_slides_examined=("blood_slides_examined", "sum"),
        rainfall_mm=("rainfall_mm", "mean"), avg_temp_c=("avg_temp_c", "mean"),
        humidity_pct=("humidity_pct", "mean"), population=("population", "sum"))
    return agg.reset_index()


# ------------------------------------------------------------- features
FEATURES = None


def make_features(df, end="2012-12-01"):
    """District-month features, all known at the end of month t.
    Target = log1p(indigenous cases in month t+1)."""
    d = df[df["date"] <= end].copy()
    g = d.groupby("district")
    d["log_cases"] = np.log1p(d["indigenous_cases"])
    d["target"] = g["log_cases"].shift(-1)
    for k in [1, 2, 3, 12]:
        d[f"lag_{k}"] = g["log_cases"].shift(k)
    d["roll3"] = g["log_cases"].transform(lambda s: s.rolling(3).mean())
    d["roll12"] = g["log_cases"].transform(lambda s: s.rolling(12).mean())
    d["trend_3"] = d["log_cases"] - d["lag_3"]
    for k in [1, 2]:
        d[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)
    d["rain_3m"] = g["rainfall_mm"].transform(lambda s: s.rolling(3).sum())
    d["log_slides"] = np.log1p(d["blood_slides_examined"])
    d["slide_positivity"] = d["indigenous_cases"] / d["blood_slides_examined"].clip(lower=1)
    d["pf_share"] = d["pf_cases"] / d["total_cases"].clip(lower=1)
    d["log_pop"] = np.log(d["population"])
    nxt = (d["month"] % 12) + 1          # month being predicted
    d["m_sin"] = np.sin(2 * np.pi * nxt / 12)
    d["m_cos"] = np.cos(2 * np.pi * nxt / 12)
    for z in ["Dry", "Intermediate", "Wet"]:
        d[f"zone_{z}"] = (d["climate_zone"] == z).astype(int)
    d["target_date"] = d["date"] + pd.offsets.MonthBegin(1)
    d = d.dropna(subset=["target", "lag_12", "roll12"]).reset_index(drop=True)
    return d


BASE_FEATURES = ["log_cases", "lag_1", "lag_2", "lag_3", "lag_12", "roll3", "roll12", "trend_3",
                 "rainfall_mm", "rain_lag1", "rain_lag2", "rain_3m", "avg_temp_c", "humidity_pct",
                 "log_slides", "slide_positivity", "pf_share", "log_pop", "m_sin", "m_cos",
                 "zone_Dry", "zone_Intermediate", "zone_Wet"]


def add_district_mean(X, train_mask):
    """District's average log cases, computed from TRAINING rows only (no leakage)."""
    m = X[train_mask].groupby("district")["log_cases"].mean()
    X = X.copy()
    X["district_mean"] = X["district"].map(m).fillna(m.mean())
    return X


# --------------------------------------------------------------- models
def build_models(rf_p=None, xgb_p=None, use=ALL_MODELS):
    rf_p = rf_p or {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5}
    xgb_p = xgb_p or {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05}
    rf = lambda: RandomForestRegressor(**rf_p, n_jobs=-1, random_state=SEED)
    m = {}
    if "Random Forest" in use:
        m["Random Forest"] = rf()
    if "Bagging (trees)" in use:
        m["Bagging (trees)"] = BaggingRegressor(DecisionTreeRegressor(max_depth=rf_p.get("max_depth")),
                                                n_estimators=100, max_samples=0.8, n_jobs=-1, random_state=SEED)
    if "XGBoost" in use and XGBRegressor:
        m["XGBoost"] = XGBRegressor(**xgb_p, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    if "LightGBM" in use and LGBMRegressor:
        m["LightGBM"] = LGBMRegressor(n_estimators=xgb_p["n_estimators"], learning_rate=xgb_p["learning_rate"],
                                      num_leaves=2 ** xgb_p["max_depth"], subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, random_state=SEED, verbose=-1)
    if "CatBoost" in use and CatBoostRegressor:
        m["CatBoost"] = CatBoostRegressor(iterations=xgb_p["n_estimators"], depth=min(xgb_p["max_depth"] + 2, 10),
                                          learning_rate=xgb_p["learning_rate"], random_seed=SEED, verbose=0)
    if "Gradient Boosting" in use:
        m["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=xgb_p["n_estimators"],
                                                           max_depth=xgb_p["max_depth"],
                                                           learning_rate=xgb_p["learning_rate"],
                                                           subsample=0.8, random_state=SEED)
    if "Stacking" in use:
        base = [("rf", rf())]
        if XGBRegressor:
            base.append(("xgb", XGBRegressor(**xgb_p, subsample=0.8, random_state=SEED, n_jobs=-1)))
        if LGBMRegressor:
            base.append(("lgbm", LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)))
        else:
            base.append(("gbr", GradientBoostingRegressor(random_state=SEED)))
        # cv=5 KFold inside the training period produces the out-of-fold
        # predictions the Ridge meta-model learns from.
        m["Stacking"] = StackingRegressor(base, final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    return m


def _metrics(y_log, p_log):
    y, p = np.expm1(y_log), np.expm1(np.clip(p_log, 0, None))
    return {"RMSE (log)": np.sqrt(mean_squared_error(y_log, p_log)),
            "MAE (log)": mean_absolute_error(y_log, p_log),
            "R2 (log)": r2_score(y_log, p_log),
            "MAE (cases)": mean_absolute_error(y, p)}


def _nat_metrics(y, p):
    return {"National RMSE (cases)": np.sqrt(mean_squared_error(y, p)),
            "National MAE (cases)": mean_absolute_error(y, p),
            "National MAPE %": float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100)}


# ------------------------------------------------------- national SARIMA
def run_sarima(nat, train_end, order=(1, 1, 1), seasonal=(1, 0, 1, 12), end="2012-12-01", hybrid=True, rf_p=None):
    """One-step-ahead SARIMA on national log1p cases, parameters fitted on the
    training period only. Hybrid: RF learns SARIMA's in-sample residuals."""
    s = nat[nat["date"] <= end].set_index("date")
    y = np.log1p(s["indigenous_cases"]).asfreq("MS")
    tr = y[y.index <= train_end]
    fit = SARIMAX(tr, order=order, seasonal_order=seasonal,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    full = SARIMAX(y, order=order, seasonal_order=seasonal,
                   enforce_stationarity=False, enforce_invertibility=False).filter(fit.params)
    pred = full.get_prediction().predicted_mean          # one-step-ahead for every month
    out = pd.DataFrame({"y": y, "SARIMA": pred})
    if hybrid:
        r = out["y"] - out["SARIMA"]
        F = pd.DataFrame(index=y.index)
        for k in [1, 2, 3, 12]:
            F[f"y_lag{k}"] = y.shift(k)
            F[f"res_lag{k}"] = r.shift(k)
        F["rain"] = s["rainfall_mm"].asfreq("MS")
        F["rain_lag1"] = F["rain"].shift(1)
        F["temp"] = s["avg_temp_c"].asfreq("MS")
        F["m_sin"] = np.sin(2 * np.pi * y.index.month / 12)
        F["m_cos"] = np.cos(2 * np.pi * y.index.month / 12)
        F["sarima"] = out["SARIMA"]
        ok = F.notna().all(axis=1) & (F.index > y.index[13])   # skip SARIMA burn-in
        trm = ok & (F.index <= train_end)
        p = dict(rf_p or {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 3})
        p["max_depth"] = min(p.get("max_depth") or 6, 6)          # tiny sample: keep forest shallow
        rf = RandomForestRegressor(**p, random_state=SEED, n_jobs=-1).fit(F[trm], r[trm])
        out["SARIMA + RF"] = out["SARIMA"] + pd.Series(rf.predict(F[ok]), index=F.index[ok])
    return out


# ------------------------------------------------------- full experiment
def run_experiment(df, train_end="2007-12-01", rf_p=None, xgb_p=None, order=(1, 1, 1),
                   use=tuple(ALL_MODELS), end="2012-12-01", smearing=True, predict_change=True):
    notes = list(NOTES)
    X = make_features(df, end)
    train = X["target_date"] <= train_end
    test = ~train
    Xtr, Xte = X.loc[train, BASE_FEATURES], X.loc[test, BASE_FEATURES]
    ytr, yte = X.loc[train, "target"], X.loc[test, "target"]
    # Models learn the month-on-month CHANGE in log cases (like returns in finance),
    # then we add it back to this month's level. Trees cannot extrapolate levels they
    # never saw, but changes look similar in the high-case and low-case eras.
    base_tr, base_te = X.loc[train, "log_cases"].values, X.loc[test, "log_cases"].values
    dtr = ytr - base_tr if predict_change else ytr
    add_te = base_te if predict_change else 0.0
    add_tr = base_tr if predict_change else 0.0

    preds = {"Naive (next = this month)": X.loc[test, "log_cases"].values,
             "Seasonal naive (same month last year)": X.loc[test, "lag_12"].shift(0).values}
    # seasonal naive: value 11 months before t = same calendar month as t+1 last year
    g = X.groupby("district")["log_cases"]
    preds["Seasonal naive (same month last year)"] = g.shift(11).loc[test].fillna(X.loc[test, "log_cases"]).values

    models = build_models(rf_p, xgb_p, use)
    importances, smear = {}, {}
    kf = KFold(3, shuffle=True, random_state=SEED)
    for name, mdl in models.items():
        if smearing:
            oof = cross_val_predict(clone(mdl), Xtr, dtr, cv=kf) + add_tr
            smear[name] = float(np.mean(np.exp(ytr.values - oof)))
        mdl.fit(Xtr, dtr)
        preds[name] = mdl.predict(Xte) + add_te
        if hasattr(mdl, "feature_importances_"):
            fi = np.asarray(mdl.feature_importances_, float)
            importances[name] = fi / fi.sum()
    if "RF + ANN" in use:
        # Hybrid: average of a Random Forest and a neural network (MLP).
        rf = models.get("Random Forest") or RandomForestRegressor(n_estimators=300, max_depth=12,
                                                                  min_samples_leaf=5, n_jobs=-1,
                                                                  random_state=SEED).fit(Xtr, dtr)
        ann = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                           max_iter=800, early_stopping=True,
                                                           random_state=SEED)).fit(Xtr, dtr)
        preds["ANN (MLP)"] = ann.predict(Xte) + add_te
        preds["RF + ANN"] = 0.5 * (rf.predict(Xte) + add_te) + 0.5 * preds["ANN (MLP)"]
        if smearing:
            o_rf = cross_val_predict(clone(rf), Xtr, dtr, cv=kf) + add_tr
            o_ann = cross_val_predict(clone(ann), Xtr, dtr, cv=kf) + add_tr
            smear["ANN (MLP)"] = float(np.mean(np.exp(ytr.values - o_ann)))
            smear["RF + ANN"] = float(np.mean(np.exp(ytr.values - 0.5 * (o_rf + o_ann))))

    # district-level table
    rows = {k: _metrics(yte.values, v) for k, v in preds.items()}
    table = pd.DataFrame(rows).T

    # national comparison: sum district predictions per month
    te = X.loc[test, ["target_date", "district", "target"]].copy()
    for k, v in preds.items():
        # Duan smearing: exp(E[log y]) underestimates E[y]; scale by mean(exp(OOF residual))
        te[k] = np.clip(np.exp(np.clip(v, 0, None)) * smear.get(k, 1.0) - 1, 0, None)
    te["actual"] = np.expm1(te["target"])
    nat_pred = te.groupby("target_date")[["actual"] + list(preds)].sum()

    if SARIMAX is not None and ("SARIMA" in use or "SARIMA + RF" in use):
        try:
            sar = run_sarima(national(df), train_end, order=order, end=end,
                             hybrid="SARIMA + RF" in use, rf_p=rf_p)
            sar = sar.loc[nat_pred.index]
            for c in ["SARIMA", "SARIMA + RF"]:
                if c in sar and c in use:
                    nat_pred[c] = np.expm1(sar[c].clip(lower=0))
        except Exception as e:  # keep the dashboard alive
            notes.append(f"SARIMA failed: {e}")

    nat_rows = {k: _nat_metrics(nat_pred["actual"].values, nat_pred[k].values)
                for k in nat_pred.columns if k != "actual"}
    nat_table = pd.DataFrame(nat_rows).T
    table = nat_table.join(table, how="left")
    table.insert(0, "Technique", [GROUP.get(i, "Hybrid" if "ANN" in i else "") for i in table.index])
    table = table.sort_values("National RMSE (cases)")

    imp = pd.DataFrame(importances, index=BASE_FEATURES) if importances else pd.DataFrame()
    return {"table": table, "national": nat_pred.reset_index().rename(columns={"target_date": "date"}),
            "district_test": te, "importances": imp, "n_train": int(train.sum()),
            "n_test": int(test.sum()), "notes": notes, "features": X, "smearing": smear}


if __name__ == "__main__":
    df = load()
    res = run_experiment(df)
    pd.set_option("display.width", 200)
    print(f"Train rows {res['n_train']}, test rows {res['n_test']}")
    for n in res["notes"]:
        print("NOTE:", n)
    print(res["table"].round(3))
    res["table"].to_csv("results.csv")
    res["importances"].to_csv("feature_importances.csv")
