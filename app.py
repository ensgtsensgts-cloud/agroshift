import os
import json
import sqlite3
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from model import (
    ahp, matrix_from_sliders, CRITERIA, SCENARIOS, CROPS,
    predict_from_latlon, predict_anywhere_world, main as build_model_outputs
)

app = Flask(__name__)
CORS(app)

DB_PATH = "farm.db"
SCENARIOS_JSON = "web/cell_scenarios.json"

# التأكد من توليد ملفات الموديل أول مرة إذا لم تكن موجودة
if not os.path.exists(SCENARIOS_JSON) or not os.path.exists("rf_global_model.pkl"):
    build_model_outputs(DB_PATH, "web")


def polygon_area_ha(poly):
    pts = np.asarray(poly, float)
    lat0, lon0 = pts[:, 0].mean(), pts[:, 1].mean()
    kx = 111.32 * np.cos(np.radians(lat0))
    ky = 110.57
    vx = (pts[:, 1] - lon0) * kx
    vy = (pts[:, 0] - lat0) * ky
    area_km2 = 0.5 * abs(np.dot(vx, np.roll(vy, -1)) - np.dot(vy, np.roll(vx, -1)))
    return round(float(area_km2 * 100.0), 2)


@app.route("/")
@app.route("/shift")
def index():
    return send_file("shift.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        data = request.get_json(force=True)
        polygon = data.get("polygon", [])
        raw_weights = data.get("weights", {})
        has_well = bool(data.get("has_well", False))
        prev_crops = data.get("prev_crops", [])  # [{"crop": "24", "yield": "35"}, ...]

        if len(polygon) < 3:
            return jsonify({"error": "Polygon requires at least 3 points."}), 400

        pts = np.asarray(polygon, float)
        lat = float(pts[:, 0].mean())
        lon = float(pts[:, 1].mean())
        area_ha = polygon_area_ha(polygon)

        # 1. حساب أوزان AHP من سلايدرات الفلاح (1-9)
        sliders = {c: float(raw_weights.get(c, 5)) for c in CRITERIA}
        w_vec, _ = ahp(matrix_from_sliders(sliders))
        weights_dict = {c: round(float(w_vec[i]), 4) for i, c in enumerate(CRITERIA)}

        # قراءة دقة النموذج (CV) من الملف المحفوظ
        with open(SCENARIOS_JSON, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        cv_report = meta_json.get("meta", {}).get("cv", {})

        # 2. محاولة التحليل المحلي أولاً (Finney County)
        local_res = predict_from_latlon(
            lat=lat, lon=lon, has_well=has_well, db=DB_PATH, scenarios_json=SCENARIOS_JSON
        )

        # حساب معامل تعديل المردود إذا أدخل الفلاح محاصيله السابقة خارج منطقة الدراسة
        ref_bu = {24: 40.0, 4: 80.0, 1: 150.0, 5: 55.0, 36: 4.5}
        factors = []
        c_hist = [24, 4, 61]
        for idx, item in enumerate(prev_crops):
            c_code = int(item.get("crop", 24))
            if idx < 3:
                c_hist[idx] = c_code
            y_val = float(item.get("yield", 0) or 0)
            if y_val > 0 and c_code in ref_bu:
                factors.append(y_val / ref_bu[c_code])
        yield_factor = round(float(np.clip(np.mean(factors), 0.4, 1.8)), 2) if factors else 1.0

        if local_res.get("status") == "ok":
            covered = True
            cid = local_res["cell_id"]
            con = sqlite3.connect(DB_PATH)
            cell_row = pd.read_sql(f"SELECT * FROM cells WHERE cell_id = '{cid}'", con).iloc[0]
            feat_row = pd.read_sql(f"SELECT * FROM features WHERE cell_id = '{cid}' ORDER BY year DESC LIMIT 1", con).iloc[0]
            cdl_map = dict(con.execute("SELECT DISTINCT CAST(cdl_code AS INT), crop FROM cdl WHERE cdl_code IS NOT NULL").fetchall())
            con.close()

            raw_seq = str(cell_row.get("seq_2018_2024", "–"))
            named_seq = " → ".join(
                cdl_map.get(int(code), code) for code in raw_seq.split("-") if code.strip().isdigit()
            ) or raw_seq

            report = {
                "seq_2018_2024": named_seq,
                "rotation_class": str(cell_row.get("rotation_class", "–")),
                "rotation_class": str(cell_row.get("rotation_class", "–")),
                "ndvi": round(float(feat_row.get("ndvi_gs_max", 0)), 2),
                "bare": int(feat_row.get("bare_months", 0)),
                "deficit": round(float(feat_row.get("water_deficit_mm", 0)), 0),
                "precip": round(float(feat_row.get("precip_gs", 0)), 0),
                "slope_deg": round(float(cell_row.get("slope_deg", 0)), 2),
                "clay_surf": round(float(cell_row.get("clay_surf", 0)), 1),
            }
            gw_info = local_res["groundwater_and_well"]
            scenarios = local_res["scenarios"]
        else:
            # 3. التحليل العالمي المباشر عبر Open-Meteo لأي إحداثيات في العالم
            covered = False
            world_res = predict_anywhere_world(
                lat=lat, lon=lon, crop_0=c_hist[0], crop_1=c_hist[1], crop_2=c_hist[2], has_well=has_well
            )
            scenarios = world_res["scenarios"]
            report = {}
            gw_info = {"estimated_water_depth_m": None, "water_level_trend_ft_yr": 0, "nearest_well_distance_km": "N/A"}

        # إعادة ترتيب السيناريوهات حسب أوزان سلايدرات الفلاح الدقيقة + تطبيق yield_factor
        base = scenarios["A_current"]["criteria"]
        sign = [-1, 1, 1, -1, 1]
        crit_keys = ["water_use_mm", "income_usd_ac", "soil_score", "erosion_index", "vigor_ndvi"]

        for sid, sc in scenarios.items():
            c = sc["criteria"]
            if not covered and yield_factor != 1.0:
                c["income_usd_ac"] = round(c["income_usd_ac"] * yield_factor, 2)
            tot = 0.0
            for i, k in enumerate(crit_keys):
                rel = sign[i] * (c[k] - base[k]) / (0.30 * abs(base[k]) + 1e-9)
                tot += w_vec[i] * float(np.clip(rel, -1, 1))
            sc["score"] = round(tot, 3)
            sc["warning"] = bool(covered and gw_info.get("water_level_trend_ft_yr", 0) > 2.0 and sid == "D_alfalfa")

        ranking = sorted(scenarios.keys(), key=lambda k: scenarios[k]["score"], reverse=True)
        seq_map = {sid: SCENARIOS[sid][1] for sid in SCENARIOS}

        return jsonify({
            "covered": covered,
            "area_ha": area_ha,
            "cell_id": local_res.get("cell_id") if covered else "Global (Open-Meteo)",
            "distance_km": local_res.get("distance_km", 0),
            "weights": weights_dict,
            "report": report,
            "groundwater_and_well": gw_info,
            "cv": cv_report,
            "scenarios": scenarios,
            "ranking": ranking,
            "seq": seq_map,
            "yield_factor": yield_factor,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/series")
def api_series():
    lat = float(request.args.get("lat", 0))
    lon = float(request.args.get("lon", 0))
    con = sqlite3.connect(DB_PATH)
    cells = pd.read_sql("SELECT cell_id, lat, lon FROM cells", con)
    kx = 111.32 * np.cos(np.radians(lat))
    dist = np.sqrt(((cells["lon"] - lon) * kx) ** 2 + ((cells["lat"] - lat) * 110.57) ** 2)
    idx = dist.idxmin()
    if float(dist.loc[idx]) > 5.0:
        con.close()
        return jsonify({"covered": False})

    cid = cells.loc[idx, "cell_id"]
    m = pd.read_sql(f"SELECT year, month, NDVI FROM monthly WHERE cell_id = '{cid}'", con)
    con.close()
    ndvi_2024 = m[m["year"] == 2024].sort_values("month")["NDVI"].fillna(0).tolist()
    ndvi_mean = m.groupby("month")["NDVI"].mean().sort_index().fillna(0).tolist()
    return jsonify({"covered": True, "ndvi": ndvi_2024, "ndvi_mean": ndvi_mean})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)