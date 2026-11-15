#!/usr/bin/env python3
"""
02_models.py  --  Field Shift / Rotation Advisor
-----------------------------------------------------------
Reads farm.db (made by 01_build_database.py) and produces:

  1. AHP weights from pairwise comparisons (+ consistency ratio)
  2. Random Forest that learns, from satellite history, how each crop (and the
     crop before it) behaves in a given cell: vigor (NDVI), water deficit,
     bare-soil months.  Validated against a simple baseline (honest check).
  3. Rotation scenario scoring per cell  (existing + 3 alternatives)
  4. irrigation_advice(): weather-forecast based "irrigate / wait" rule
  5. web/cell_scenarios.json  -> consumed directly by the front end

Usage:
    python 02_models.py --db farm.db --out web
"""
import argparse, json, os
import numpy as np
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
import requests
import joblib


# ---------------------------------------------------------------- 1. AHP ----
RI = {1: 0, 2: 0, 3: .58, 4: .90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
CRITERIA = ["water", "income", "soil", "erosion", "resilience"]


def ahp(matrix):
    """Saaty AHP: principal eigenvector -> weights, plus consistency ratio.
    matrix[i][j] = how many times criterion i is more important than j (1-9)."""
    A = np.array(matrix, float)
    n = len(A)
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real); w /= w.sum()
    ci = (vals[k].real - n) / (n - 1)
    cr = ci / RI[n] if RI[n] else 0.0
    return w, cr          # cr < 0.10  => judgements are acceptably consistent


def matrix_from_sliders(priorities):
    """Farmer-friendly input: 1-9 importance score per criterion.
    Builds the pairwise matrix A[i][j] = p_i / p_j  (always consistent), so a
    farmer never has to fill a 5x5 matrix. Experts can still pass a full matrix."""
    p = np.array([priorities[c] for c in CRITERIA], float)
    return p[:, None] / p[None, :]


PROFILES = {   # example presets (replace with real survey / expert values)
    "water_saver":  dict(water=9, income=4, soil=5, erosion=3, resilience=6),
    "profit_first": dict(water=3, income=9, soil=3, erosion=2, resilience=4),
    "soil_builder": dict(water=5, income=3, soil=9, erosion=7, resilience=5),
}

# ------------------------------------------------------ 2. Random Forest ----
CROPS = {24: "Winter wheat", 4: "Sorghum", 1: "Corn", 5: "Soybeans",
         36: "Alfalfa", 61: "Fallow"}
TARGETS = ["ndvi_gs_max", "et_gs", "bare_months"]   # et_gs = crop water use (mm)
CLIM = ["precip_gs", "t2m_gs", "sm_root_gs"]
STATIC = ["clay_surf", "soil_carbon_surf", "slope_deg"]


def design(df):
    X = df[CLIM + STATIC].copy()
    for code in CROPS:
        X[f"crop_{code}"] = (df.cdl_code == code).astype(int)
        X[f"prev1_{code}"] = (df.crop_prev1 == code).astype(int)
        X[f"prev2_{code}"] = (df.crop_prev2 == code).astype(int)
    return X


def train(feat):
    d = feat[feat.cdl_code.isin(CROPS)].dropna(subset=CLIM + STATIC + TARGETS).copy()
    X, Y = design(d), d[TARGETS]
    # spatial block groups (~10 km) -> no neighbouring-cell leakage in CV
    blocks = (d.lat.round(1).astype(str) + "_" + d.lon.round(1).astype(str))
    rf_cv, base_cv = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for tr, te in GroupKFold(5).split(X, Y, blocks):
        rf = RandomForestRegressor(150, min_samples_leaf=5, n_jobs=-1, random_state=0)
        rf.fit(X.iloc[tr], Y.iloc[tr])
        P = rf.predict(X.iloc[te])
        # baseline = mean per crop (what a rule-based tool would use)
        key = d.cdl_code.iloc[tr]
        means = Y.iloc[tr].groupby(key.values).mean()
        B = means.reindex(d.cdl_code.iloc[te]).values
        for i, t in enumerate(TARGETS):
            rf_cv[t].append(r2_score(Y.iloc[te][t], P[:, i]))
            base_cv[t].append(r2_score(Y.iloc[te][t], B[:, i]))
    report = {t: dict(RF_r2=float(np.mean(rf_cv[t])), baseline_r2=float(np.mean(base_cv[t])))
              for t in TARGETS}
    model = RandomForestRegressor(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X, Y)
    return model, X.columns, d, report


def predict_with_uncertainty(model, X):
    """Mean + std across trees  ->  uncertainty for every prediction."""
    allp = np.stack([t.predict(X.values) for t in model.estimators_])   # trees x n x targets
    return allp.mean(0), allp.std(0)


# -------------------------------------------------- 3. scenario scoring ----
SCENARIOS = {
    "A_current":    ("Current: wheat-sorghum-fallow",        [24, 4, 61]),
    "B_no_fallow":  ("Wheat-sorghum (no bare fallow)",       [24, 4]),
    "C_legume":     ("Wheat-sorghum-soybeans (N-fixing)",    [24, 4, 5]),
    "D_alfalfa":    ("Alfalfa 3 yr then wheat (deep roots)", [36, 36, 36, 24]),
}
# EXPERT PRIORS (0-1) -- literature-based, MUST be validated by local agronomists
SOIL_PRIOR = {24: .5, 4: .6, 1: .6, 5: .6, 36: 1.0, 61: .0}   # OM / N contribution
YIELD_BU_AC = {24: 40, 4: 80, 1: 150, 5: 55, 36: 4.5}          # replaced by QuickStats below
ASSUMED_PRICE = {36: 190.0}                                     # $/ton alfalfa (assumption)


def economics(con):
    """Gross revenue $/acre per crop from real QuickStats yields & official USDA/futures prices."""
    # 1. جلب المردود السنوي الصافي (بدون خلط المسقي مع البعلي)
    y = pd.read_sql("""
        SELECT * FROM yields 
        WHERE [Data Item] IN (
            'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE',
            'WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE',
            'SORGHUM, GRAIN - YIELD, MEASURED IN BU / ACRE',
            'SOYBEANS - YIELD, MEASURED IN BU / ACRE'
        )
    """, con)
    yl = y.sort_values("Year").groupby("Commodity").Value.apply(lambda s: s.tail(5).mean())

    # 2. جلب الأسعار الرسمية من USDA (2020-2024) للسورغو ($/bu) والفصة ($/ton)
    pr_usda = pd.read_sql("""
        SELECT Commodity, Value FROM yields 
        WHERE [Data Item] IN (
            'SORGHUM, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU',
            'HAY, ALFALFA - PRICE RECEIVED, MEASURED IN $ / TON'
        ) AND Year BETWEEN 2020 AND 2024
    """, con)
    pm_usda = pr_usda.groupby("Commodity").Value.mean()

    # 3. جلب أسعار العقود الآجلة للقمح والذرة والصوجا
    p = pd.read_sql("SELECT Crop_Name, Close FROM prices WHERE Year >= 2022", con)
    pm = p.groupby("Crop_Name").Close.mean() / 100.0            # cents/bu -> $/bu

    price = {
        24: pm.get("Wheat_Blé"),
        1:  pm.get("Corn_Mais"),
        5:  pm.get("Soybeans_Soja"),
        4:  pm_usda.get("SORGHUM"),   # سعر السورغو الحقيقي من USDA ($/bu)
        36: pm_usda.get("HAY"),       # سعر الفصة الحقيقي من USDA ($/ton)
    }

    com = {24: "WHEAT", 4: "SORGHUM", 1: "CORN", 5: "SOYBEANS"}
    rev = {c: float(yl.get(com[c], YIELD_BU_AC[c])) * float(price[c]) for c in com if pd.notna(price[c])}
    rev[36] = YIELD_BU_AC[36] * float(price[36])
    rev[61] = 0.0
    return rev

def scenario_metrics(model, cols, feat, rev, cell_ids=None):
    """For each cell and scenario: predicted (cycle-average) criteria + uncertainty."""
    cells = feat.groupby("cell_id").agg(
        precip_gs=("precip_gs", "mean"), t2m_gs=("t2m_gs", "mean"),
        sm_root_gs=("sm_root_gs", "mean"), clay_surf=("clay_surf", "first"),
        soil_carbon_surf=("soil_carbon_surf", "first"),
        slope_deg=("slope_deg", "first"), lat=("lat", "first"),
        lon=("lon", "first")).dropna().reset_index()
    out = {}
    for sid, (_, seq) in SCENARIOS.items():
        rows = []
        for i, crop in enumerate(seq):                       # each year of the cycle
            d = cells.copy()
            d["cdl_code"] = crop
            d["crop_prev1"] = seq[i - 1]
            d["crop_prev2"] = seq[i - 2] if len(seq) > 1 else seq[i - 1]
            m, s = predict_with_uncertainty(model, design(d)[cols])
            rows.append((m, s, crop))
        mean = np.mean([r[0] for r in rows], 0)
        std = np.mean([r[1] for r in rows], 0)
        income = np.mean([rev[r[2]] for r in rows])
        soil = np.mean([SOIL_PRIOR[r[2]] for r in rows])
        bare = mean[:, 2]
        erosion = bare * (1 + cells.slope_deg.values)         # bare months x slope
        out[sid] = pd.DataFrame(dict(
            cell_id=cells.cell_id, water_use=mean[:, 1], income=income,
            soil=soil, erosion=erosion, vigor=mean[:, 0], bare=bare,
            uncertainty=std.mean(1) / (std.mean(1).max() + 1e-9)))
    return out


def score(metrics, w, ref="A_current", full_effect=0.30):
    """Score each scenario RELATIVE TO THE FARMER'S CURRENT ROTATION.
    For each criterion: % change vs current, where a +/-30% change = full score (+1 / -1).
    (Min-max stretching was rejected: it blows tiny differences up to 0-1.)
    Result > 0  => better than what the farmer does now, weighted by AHP priorities."""
    ids = list(metrics)
    sign = {"water_use": -1, "income": 1, "soil": 1, "erosion": -1, "vigor": 1}
    col = dict(zip(CRITERIA, ["water_use", "income", "soil", "erosion", "vigor"]))
    base = metrics[ref]
    res = {}
    for s in ids:
        tot = 0
        for c, wi in zip(CRITERIA, w):
            k = col[c]
            rel = sign[k] * (metrics[s][k].values - base[k].values) / (full_effect * np.abs(base[k].values) + 1e-9)
            tot = tot + wi * np.clip(rel, -1, 1)
        res[s] = tot
    return pd.DataFrame(res, index=base.cell_id)


# ----------------------------------------------- 4. weather -> irrigation ----
def irrigation_advice(lat, lon, soil_moisture_frac=None, rain_skip_mm=6.0, prob_skip=60):
    """Free Open-Meteo forecast (no API key). Returns plain-language advice.
    Swap/add: NASA POWER (history), GPM IMERG 'Early/Late' run (near-real-time)."""
    import requests
    url = "https://api.open-meteo.com/v1/forecast"
    r = requests.get(url, timeout=15, params=dict(
        latitude=lat, longitude=lon, timezone="auto", forecast_days=3,
        daily="precipitation_sum,precipitation_probability_max,"
              "et0_fao_evapotranspiration,temperature_2m_max")).json()["daily"]
    rain48 = sum(r["precipitation_sum"][:2])
    prob = max(r["precipitation_probability_max"][:2])
    et0 = sum(r["et0_fao_evapotranspiration"][:2])
    tmax = max(r["temperature_2m_max"][:2])
    if rain48 >= rain_skip_mm and prob >= prob_skip:
        msg, code = (f"WAIT - about {rain48:.0f} mm of rain is likely in the next 48 h "
                     f"({prob}% chance). Irrigating now would waste water.", "green")
    elif soil_moisture_frac is not None and soil_moisture_frac < 0.10 and tmax > 35:
        msg, code = ("IRRIGATE - very dry root zone, heat >35 C and no useful rain coming.", "red")
    else:
        msg, code = (f"CHECK FIELD - little rain expected ({rain48:.0f} mm); "
                     f"crop water demand ~{et0:.0f} mm over 2 days.", "yellow")
    return dict(advice=msg, level=code, rain48_mm=rain48, rain_prob=prob, et0_48h_mm=et0,
                note="Decision support only - confirm with local knowledge.")


# ----------------------------------------------------------------- main ----
def main(db, out):
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(db):
        raise SystemExit(f"Database '{db}' not found in {os.getcwd()}. "
                         "Run 01_build_database.py first (or pass --db with the full path).")
    con = sqlite3.connect(db)
    try:
        _run(con, db, out)
    finally:
        con.close()          # always release the file (avoids Windows "file in use")


def _run(con, db, out):
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    if "features" not in tables:
        raise SystemExit(f"'{db}' has no 'features' table (found: {sorted(tables) or 'none'}). "
                         "01_build_database.py did not finish: re-run it and wait for 'OK ->'.")
    feat = pd.read_sql("select * from features", con)

    model, cols, d, rep = train(feat)
    print("Spatial-block CV  (R2 RF vs per-crop-mean baseline):")
    for t, v in rep.items():
        print(f"  {t:18s} RF={v['RF_r2']:.2f}  baseline={v['baseline_r2']:.2f}")
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print("Top drivers:\n", imp.head(8).round(3).to_string())

    rev = economics(con)
    print("Gross revenue $/acre:", {CROPS[k]: round(v) for k, v in rev.items()})
    metrics = scenario_metrics(model, cols, feat, rev)

    # Example of a REAL expert pairwise matrix (order = CRITERIA). Slider profiles are
    # always "consistent" by construction, so CR only means something for matrices like this.
    expert = [[1, 3, 2, 4, 2], [1/3, 1, 1/2, 2, 1], [1/2, 2, 1, 3, 1],
              [1/4, 1/2, 1/3, 1, 1/2], [1/2, 1, 1, 2, 1]]
    w_e, cr_e = ahp(expert)
    print("Expert AHP weights:", dict(zip(CRITERIA, w_e.round(3).tolist())), "CR =", round(cr_e, 3))

    result = {"meta": dict(scenarios={k: v[0] for k, v in SCENARIOS.items()},
                           criteria=CRITERIA, cv=rep,
                           disclaimer="Decision support - not a replacement for farmer/agronomist judgement."),
              "profiles": {}, "cells": {}}
    for name, pri in PROFILES.items():
        w, cr = ahp(matrix_from_sliders(pri))
        sc = score(metrics, w)
        result["profiles"][name] = dict(weights=dict(zip(CRITERIA, np.round(w, 3))),
                                        consistency_ratio=round(abs(cr), 3),
                                        best_share={k: float((sc.idxmax(axis=1) == k).mean())
                                                    for k in SCENARIOS})
    # raw criteria per cell -> front end re-weights live with the farmer's sliders
    ids = list(SCENARIOS)
    base = metrics[ids[0]]
    for i, cid in enumerate(base.cell_id):
        result["cells"][cid] = {s: [round(float(metrics[s][c].iloc[i]), 3)
                                    for c in ("water_use", "income", "soil",
                                              "erosion", "vigor", "uncertainty")]
                                for s in ids}
    with open(os.path.join(out, "cell_scenarios.json"), "w") as f:
        json.dump(result, f)
    flat = [dict(cell_id=cid, scenario=s, scenario_name=SCENARIOS[s][0],
                 water_use_mm=v[0], income_usd_ac=v[1], soil_score=v[2],
                 erosion_index=v[3], vigor_ndvi=v[4], uncertainty=v[5])
            for cid, sc in result["cells"].items() for s, v in sc.items()]
    pd.DataFrame(flat).to_csv(os.path.join(out, "cell_scenarios.csv"),
                              index=False, encoding="utf-8-sig")
    print("wrote", os.path.join(out, "cell_scenarios.json"))
    joblib.dump({"model": model, "cols": list(cols)}, "rf_global_model.pkl")

def fetch_global_climate_soil(lat, lon):
    """جلب بيانات المناخ (أمطار، حرارة، رطوبة الجذور) لأي إحداثيات في العالم من Open-Meteo"""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date=2024-04-01&end_date=2024-09-30"
        f"&daily=precipitation_sum,temperature_2m_mean"
        f"&hourly=soil_moisture_28_to_100cm&timezone=auto"
    )
    res = requests.get(url, timeout=15).json()
    daily = res.get("daily", {})
    hourly = res.get("hourly", {})

    precip_gs = float(np.nansum(daily.get("precipitation_sum", [250.0])))
    t2m_gs = float(np.nanmean(daily.get("temperature_2m_mean", [22.0])))
    sm_root = float(np.nanmean(hourly.get("soil_moisture_28_to_100cm", [0.22])))

    return {
        "precip_gs": precip_gs,
        "t2m_gs": t2m_gs if not np.isnan(t2m_gs) else 22.0,
        "sm_root_gs": sm_root if not np.isnan(sm_root) else 0.22,
    }


def predict_anywhere_world(lat, lon, crop_0, crop_1, crop_2, has_well=False, available_crops=None, profile="water_saver"):
    """تشتغل في أي بقعة في العالم باستخدام Open-Meteo + مدخلات الفلاح للمحاصيل السابقة والمردود المتوقع"""
    if not os.path.exists("rf_global_model.pkl"):
        main("farm.db", "web")
    bundle = joblib.load("rf_global_model.pkl")
    rf_model, cols = bundle["model"], bundle["cols"]

    env = fetch_global_climate_soil(lat, lon)

    # متوسطات اقتصادية مرجعية ومردود أساسي ($/acre و qx/ha)
    rev_ref = {24: 260.0, 4: 320.0, 1: 680.0, 5: 560.0, 36: 850.0, 61: 0.0}
    base_yields = {24: 42.0, 4: 68.0, 1: 145.0, 5: 48.0, 36: 4.5, 61: 0.0}
    to_qx_ha = {24: 0.672, 4: 0.627, 1: 0.627, 5: 0.672, 36: 22.417, 61: 0.0}

    # محاكاة السيناريوهات الأربعة عبر Random Forest العالمي مع تمرير تاريخ المحاصيل
    raw_scenarios = {}
    for sid, (label, seq) in SCENARIOS.items():
        sim_rows = []
        for idx, c_next in enumerate(seq):
            p1 = crop_0 if idx == 0 else seq[idx - 1]
            p2 = crop_1 if idx == 0 else (crop_0 if idx == 1 else seq[idx - 2])
            sim_rows.append({
                "precip_gs": env["precip_gs"],
                "t2m_gs": env["t2m_gs"],
                "sm_root_gs": env["sm_root_gs"] + (0.04 if has_well else 0.0),
                "clay_surf": 25.0,
                "soil_carbon_surf": 1.8,
                "slope_deg": 1.5,
                "cdl_code": c_next,
                "crop_prev1": p1,
                "crop_prev2": p2,
            })
        df_sim = pd.DataFrame(sim_rows)
        X_sim = design(df_sim)[cols]
        preds = rf_model.predict(X_sim)  # [ndvi_gs_max, et_gs, bare_months]
        ndvi_m, et_m, bare_m = preds.mean(axis=0)

        soil_sc = round(float(np.mean([SOIL_PRIOR[c] for c in seq])), 3)
        inc_m = round(float(np.mean([rev_ref[c] * (ndvi_m / 0.55) for c in seq])), 2)
        eros = round(float(bare_m * 1.5), 2)
        raw_scenarios[sid] = [round(float(et_m), 1), inc_m, soil_sc, eros, round(float(ndvi_m), 3), 0.15]

    w, _ = ahp(matrix_from_sliders(PROFILES[profile]))
    base = raw_scenarios["A_current"]
    sign = [-1, 1, 1, -1, 1]

    # معامل حساسية كل محصول للجفاف (FAO Ky) والاحتياج المائي (mm)
    ky_factor = {24: 1.0, 4: 0.9, 1: 1.25, 5: 0.85, 36: 1.1, 61: 0.0}
    water_need = {24: 350.0, 4: 320.0, 1: 500.0, 5: 400.0, 36: 650.0, 61: 0.0}

    detailed_results = {}
    for sid, vals in raw_scenarios.items():
        tot_score = sum(w[i] * float(np.clip(sign[i] * (vals[i] - base[i]) / (0.30 * abs(base[i]) + 1e-9), -1, 1)) for i in range(5))
        vigor_factor = vals[4] / 0.55

        harvest_breakdown = []
        adj_incomes = []
        for code in SCENARIOS[sid][1]:
            if code == 61:
                harvest_breakdown.append({"crop": "Fallow (بور)", "expected_yield_metric": "0 (راحة للأرض)"})
                adj_incomes.append(0.0)
            else:
                if has_well:
                    water_stress_mult = 1.0
                else:
                    supply_ratio = min(1.0, (env["precip_gs"] + 80.0) / water_need[code])
                    water_stress_mult = max(0.25, 1.0 - ky_factor[code] * (1.0 - supply_ratio))

                final_mult = vigor_factor * water_stress_mult
                y_qx = round(base_yields[code] * final_mult * to_qx_ha[code], 1)
                harvest_breakdown.append({
                    "crop": CROPS[code],
                    "expected_yield_metric": f"{y_qx} qx/ha"
                })
                adj_incomes.append(rev_ref[code] * final_mult)

        vals_adj = list(vals)
        vals_adj[1] = round(float(np.mean(adj_incomes)), 2)

        detailed_results[sid] = {
            "name": SCENARIOS[sid][0],
            "score": round(tot_score, 3),
            "expected_harvest": harvest_breakdown,
            "criteria": {
                "water_use_mm": vals_adj[0],
                "income_usd_ac": vals_adj[1],
                "soil_score": vals_adj[2],
                "erosion_index": vals_adj[3],
                "vigor_ndvi": vals_adj[4],
                "uncertainty": vals_adj[5],
            }
        }

    match_note = f"🌍 تحليل عالمي مباشر عبر Open-Meteo (أمطار الموسم: {env['precip_gs']:.0f} mm | الحرارة: {env['t2m_gs']:.1f}°C)."
    if available_crops and len(available_crops) > 0:
        avail_set = set(available_crops)
        exact_matches = {sid: res for sid, res in detailed_results.items() if avail_set.issubset(set(SCENARIOS[sid][1]))}
        partial_matches = {sid: res for sid, res in detailed_results.items() if len(avail_set.intersection(set(SCENARIOS[sid][1]))) > 0}
        if exact_matches:
            detailed_results = exact_matches
            match_note += " ✅ تم استغلال جميع بذورك المتوفرة."
        elif partial_matches:
            detailed_results = partial_matches
            match_note += " ⚠️ تم اقتراح أفضل دورة تستغل جزءاً من بذورك."

    ranked = sorted(detailed_results.items(), key=lambda x: x[1]["score"], reverse=True)
    return {
        "status": "ok",
        "seed_match_note": match_note,
        "best_recommendation": ranked[0][1]["name"],
        "scenarios": dict(ranked)
    }
def predict_from_latlon(lat, lon, has_well=False, available_crops=None, db="farm.db", scenarios_json="web/cell_scenarios.json", profile="water_saver"):
    """
    تستقبل إحداثيات GPS + سؤال هل يملك الفلاح بئراً (has_well: True/False)
    وتعيد:
    1. تشخيص المياه الجوفية وأقرب بئر مسجل
    2. المعايير الفيزيائية + المردود المتوقع (bu/ac و qx/ha) لكل سيناريو
    3. أفضل دورة زراعية مقترحة
    """
    con = sqlite3.connect(db)
    cells = pd.read_sql(
        "SELECT cell_id, lat, lon, rotation_class, gw_depth_recent_ft, gw_trend_ft_yr FROM cells", con
    )
    wells = pd.read_sql("SELECT USGS_ID, lat, lon, DEPTH_OF_WELL, USE_OF_WATER_PRIMARY FROM wells", con)
    
    # جلب متوسط المردود الرسمي من USDA (آخر 5 سنوات)
    y_df = pd.read_sql("""
        SELECT Commodity, Value FROM yields 
        WHERE [Data Item] IN (
            'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE',
            'WHEAT, WINTER - YIELD, MEASURED IN BU / ACRE',
            'SORGHUM, GRAIN - YIELD, MEASURED IN BU / ACRE',
            'SOYBEANS - YIELD, MEASURED IN BU / ACRE'
        )
    """, con)
    con.close()

    # 1. البحث عن أقرب قطعة أرض (cell_id)
    kx = 111.32 * np.cos(np.radians(lat))
    ky = 110.57
    dist_km = np.sqrt(((cells["lon"] - lon) * kx) ** 2 + ((cells["lat"] - lat) * ky) ** 2)
    nearest_idx = dist_km.idxmin()
    nearest_cell = cells.loc[nearest_idx]
    min_dist = float(dist_km.loc[nearest_idx])

    if min_dist > 5.0:
        return {"status": "out_of_bounds", "message": f"النقطة بعيدة عن منطقة الدراسة بـ {min_dist:.1f} كم."}

    # 2. تشخيص البئر والمياه الجوفية
    wells_valid = wells.dropna(subset=["lat", "lon"]).copy()
    w_dist = np.sqrt(((wells_valid["lon"] - lon) * kx) ** 2 + ((wells_valid["lat"] - lat) * ky) ** 2)
    w_idx = w_dist.idxmin()
    nearest_well = wells_valid.loc[w_idx]
    well_dist_km = float(w_dist.loc[w_idx])

    gw_depth_ft = float(nearest_cell["gw_depth_recent_ft"]) if pd.notna(nearest_cell["gw_depth_recent_ft"]) else None
    gw_depth_m = round(gw_depth_ft * 0.3048, 1) if gw_depth_ft else None
    gw_trend_ft = float(nearest_cell["gw_trend_ft_yr"]) if pd.notna(nearest_cell["gw_trend_ft_yr"]) else 0.0

    if has_well:
        if gw_trend_ft > 2.0:
            well_advice = (f"لديك بئر، لكن منسوب المياه الجوفية في منطقتك ينخفض بسرعة ({gw_trend_ft:.1f} قدم/سنة) "
                           f"على عمق {gw_depth_m} متر. يُنصح بتجنب المحاصيل المستنزفة للماء (مثل الفصة) واعتماد الري التكميلي الذكي.")
        else:
            well_advice = (f"لديك بئر ومنسوب المياه مستقر نسبياً على عمق {gw_depth_m} متر. "
                           "يمكنك اعتماد دورات زراعية ذات مردودية عالية مع مراقبة رطوبة التربة.")
    else:
        drilling_feasible = (gw_depth_m is not None and gw_depth_m < 60 and gw_trend_ft < 2.5)
        well_advice = (f"لا تملك بئراً حالياً. أقرب بئر مسجل يبعد {well_dist_km:.2f} كم (عمقه {nearest_well['DEPTH_OF_WELL']} قدم). "
                       f"عمق المياه الجوفية تحت أرضك يُقدّر بـ {gw_depth_m} متر — "
                       + ("حفر بئر جديد مُجدٍ تقنياً هنا." if drilling_feasible else "تكلفة الحفر مرتفعة والمياه عميقة؛ نوصي بالزراعات البعلية المقاومة للجفاف."))

    # 3. تحضير المردود المرجعي لكل محصول
    usda_avg = y_df.groupby("Commodity")["Value"].tail(5).groupby(y_df["Commodity"]).mean().to_dict()
    base_yields = {
        24: usda_avg.get("WHEAT", 42.0),     # bu/ac
        4:  usda_avg.get("SORGHUM", 68.0),   # bu/ac
        1:  usda_avg.get("CORN", 145.0),     # bu/ac
        5:  usda_avg.get("SOYBEANS", 48.0),  # bu/ac
        36: 4.5,                             # ton/ac (Alfalfa)
        61: 0.0
    }
    # معامل التحويل من bu/acre إلى قنطار/هكتار (qx/ha) تقريبي
    to_qx_ha = {24: 0.672, 4: 0.627, 1: 0.627, 5: 0.672, 36: 22.417, 61: 0.0}

    # 4. حساب النتائج والمردود المتوقع لكل سيناريو
    with open(scenarios_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cid = nearest_cell["cell_id"]
    raw_scenarios = data["cells"][cid]
    w, _ = ahp(matrix_from_sliders(PROFILES[profile]))
    base = raw_scenarios["A_current"]
    sign = [-1, 1, 1, -1, 1]

    detailed_results = {}
    for sid, vals in raw_scenarios.items():
        tot_score = sum(
            w[i] * float(np.clip(sign[i] * (vals[i] - base[i]) / (0.30 * abs(base[i]) + 1e-9), -1, 1))
            for i in range(5)
        )

        # حساب المردود المتوقع لكل محصول في الدورة بناءً على قوة النمو (NDVI) المتوقعة للخلية
        vigor_factor = vals[4] / 0.55  # تعديل المردود حسب NDVI المتوقع لهذه الخلية
        seq_codes = SCENARIOS[sid][1]
        harvest_breakdown = []
        for code in seq_codes:
            if code == 61:
                harvest_breakdown.append({"crop": "Fallow (بور)", "expected_yield": "0 (راحة للأرض)"})
            else:
                unit = "ton/ac" if code == 36 else "bu/ac"
                y_ac = round(base_yields[code] * vigor_factor, 1)
                y_ha = round(y_ac * to_qx_ha[code], 1)
                harvest_breakdown.append({
                    "crop": CROPS[code],
                    "expected_yield_imperial": f"{y_ac} {unit}",
                    "expected_yield_metric": f"{y_ha} qx/ha"
                })

        detailed_results[sid] = {
            "name": SCENARIOS[sid][0],
            "score": round(tot_score, 3),
            "expected_harvest": harvest_breakdown,
            "criteria": {
                "water_use_mm": vals[0],
                "income_usd_ac": vals[1],
                "soil_score": vals[2],
                "erosion_index": vals[3],
                "vigor_ndvi": vals[4],
                "uncertainty": vals[5],
            }
        }
    # فلترة الدورات بناءً على البذور المتوفرة لدى الفلاح (إن وجدت)
    match_note = "تم تقييم جميع الدورات الزراعية المتاحة."
    if available_crops and len(available_crops) > 0:
        avail_set = set(available_crops)
        # البحث عن السيناريوهات التي تحتوي على كل أو بعض البذور المتوفرة عند الفلاح
        exact_matches = {
            sid: res for sid, res in detailed_results.items()
            if avail_set.issubset(set(SCENARIOS[sid][1]))
        }
        partial_matches = {
            sid: res for sid, res in detailed_results.items()
            if len(avail_set.intersection(set(SCENARIOS[sid][1]))) > 0
        }

        if exact_matches:
            detailed_results = exact_matches
            match_note = "✅ تم العثور على دورات زراعية تستغل جميع البذور المتوفرة لديك بالكامل."
        elif partial_matches:
            detailed_results = partial_matches
            match_note = "⚠️ لم نجد دورة تضم كل بذورك معاً، فاقترحنا أفضل الدورات التي تستغل جزءاً منها وتكملها بمحاصيل مناسبة لأرضك."
        else:
            match_note = "❌ البذور المختارة لا تتوافق مع الدورات القياسية لهذه المنطقة؛ إليك أفضل البدائل المقترحة لأرضك."

    ranked = sorted(detailed_results.items(), key=lambda x: x[1]["score"], reverse=True)

    return {
        "status": "ok",
        "cell_id": cid,
        "distance_km": round(min_dist, 2),
        "groundwater_and_well": {
            "farmer_has_well": has_well,
            "estimated_water_depth_m": gw_depth_m,
            "water_level_trend_ft_yr": round(gw_trend_ft, 2),
            "nearest_registered_well_id": nearest_well["USGS_ID"],
            "nearest_well_distance_km": round(well_dist_km, 2),
            "nearest_well_depth_ft": nearest_well["DEPTH_OF_WELL"],
            "advice": well_advice
        },
        "seed_match_note": match_note,
        "best_recommendation": ranked[0][1]["name"],
        "scenarios": dict(ranked)
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="farm.db")
    ap.add_argument("--out", default="web")
    a, _ = ap.parse_known_args()   # ignores Jupyter's "-f kernel.json"
    main(a.db, a.out)
