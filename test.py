#!/usr/bin/env python3
"""
01_build_database.py  (FIXED v5)  --  Field Shift / Rotation Advisor
-----------------------------------------------------------------------
Merges every dataset of the study area (Finney County, Kansas) into ONE
SQLite database used by the models (02, 03) and the web app.

Usage:
    python 01_build_database.py
    (Optional CLI overrides: --data ./donee --out ./farm.db --et openet)
"""
import argparse, glob, os, sqlite3, sys
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION (Modifie tes chemins et paramètres ici)
# ==============================================================================
DATA_DIR   = "./donee"        # Dossier contenant tes données d'entrée
OUT_DB     = "./farm.db"      # Chemin de sortie de la base SQLite
ET_SOURCE  = "openet"         # Source ET : "openet" ou "model"
CSV_EXPORT = "csv_export"     # Dossier d'export CSV (mettre "none" pour ignorer)
ANNUAL_CSV = None             # Chemin explicite vers annual_environmental_indicators...csv (ou None)
# ==============================================================================

CDL_NAMES = {
    1: "Corn", 2: "Cotton", 4: "Sorghum", 5: "Soybeans", 6: "Sunflower",
    12: "Sweet corn", 21: "Barley", 22: "Durum wheat", 23: "Spring wheat",
    24: "Winter wheat", 25: "Other small grains", 26: "Dbl WinWheat/Soy",
    27: "Rye", 28: "Oats", 29: "Millet", 31: "Canola", 36: "Alfalfa",
    37: "Other hay", 42: "Dry beans", 43: "Potatoes", 59: "Sod/Grass seed",
    61: "Fallow", 111: "Water", 121: "Developed", 122: "Developed",
    123: "Developed", 124: "Developed", 131: "Barren", 141: "Forest",
    152: "Shrubland", 176: "Grass/Pasture", 190: "Wetland", 195: "Wetland",
    205: "Triticale", 225: "Dbl WinWheat/Corn", 226: "Dbl Oats/Corn",
    228: "Dbl Triticale/Corn", 236: "Dbl WinWheat/Sorghum",
    237: "Dbl Barley/Corn", 241: "Dbl Corn/Soy"}
CROPLAND = [1, 2, 4, 5, 6, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 36, 37,
            42, 43, 59, 61, 205, 225, 226, 228, 236, 237, 241]
DEVELOPED = [121, 122, 123, 124]
GROWING = [4, 5, 6, 7, 8, 9, 10]                 # April-October
WSF_NEXT = {61: 24, 24: 4, 4: 61}                # fallow -> wheat -> sorghum -> fallow
PRICE_CODE = {"Wheat_Blé": 24, "Corn_Mais": 1, "Soybeans_Soja": 5, "Oats_Avoine": 28}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def norm_id(s):
    """'-367,1680' and '-3,671,680' both become '-3671680' (comma = formatting only)."""
    return s.astype(str).str.replace(",", "", regex=False).str.strip()


def read_many(pattern, usecols=None):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f, usecols=usecols) for f in files], ignore_index=True)


# --------------------------------------------------------------- rotation ----
def classify_rotation(seq):
    s = [int(x) for x in seq if pd.notna(x)]
    if not s:
        return "unknown"
    if sum(1 for c in s if c == 176) >= 5:
        return "grass_pasture"
    if len(set(s)) == 1:
        return "mono_" + CDL_NAMES.get(s[0], str(s[0])).replace(" ", "_").replace("/", "_")
    if set(s) <= {24, 4, 61}:
        ok = sum(1 for a, b in zip(s[:-1], s[1:]) if WSF_NEXT.get(a) == b)
        if ok >= len(s) - 2:                      # at most one deviation
            return "wheat_sorghum_fallow"
    if any(c == 176 for c in s):
        return "mixed_with_grass"
    return "other_rotation"


# --------------------------------------------------------------- LANDFIRE ----
def albers_5070(lon, lat):
    """NAD83 / CONUS Albers (EPSG:5070), pure numpy (no pyproj needed)."""
    a = 6378137.0; f = 1 / 298.257222101; e2 = 2 * f - f * f; e = np.sqrt(e2); r = np.pi / 180

    def m(p):
        s = np.sin(p); return np.cos(p) / np.sqrt(1 - e2 * s * s)

    def q(p):
        s = np.sin(p)
        return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)))
    p1, p2, p0, l0 = 29.5 * r, 45.5 * r, 23.0 * r, -96.0 * r
    n = (m(p1) ** 2 - m(p2) ** 2) / (q(p2) - q(p1))
    C = m(p1) ** 2 + n * q(p1)
    rho0 = a * np.sqrt(C - n * q(p0)) / n
    phi = np.asarray(lat, float) * r; lam = np.asarray(lon, float) * r
    rho = a * np.sqrt(C - n * q(phi)) / n
    th = n * (lam - l0)
    return rho * np.sin(th), rho0 - rho * np.cos(th)


def read_tif(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    arr = np.array(im)
    tg = im.tag_v2
    sx, sy = tg[33550][0], tg[33550][1]
    tp = tg[33922]
    return arr, tp[3], tp[4], sx, sy


def landfire_features(cells, data, half=16):
    evts = sorted(glob.glob(os.path.join(data, "**", "landfire_EVT_finney.tif"), recursive=True),
                  key=os.path.getmtime, reverse=True)
    for evt_path in evts:
        evc_path = evt_path.replace("EVT", "EVC")
        try:
            evt, x0, y0, sx, sy = read_tif(evt_path)
        except Exception as ex:
            say(f"LANDFIRE: cannot read {evt_path} ({ex})"); continue
        if evt.max() <= 0:
            say(f"LANDFIRE: {evt_path} is EMPTY (all zeros) -> skipped"); continue
        evc = read_tif(evc_path)[0] if os.path.exists(evc_path) else None
        x, y = albers_5070(cells.lon.values, cells.lat.values)
        col = np.floor((x - x0) / sx).astype(int); row = np.floor((y0 - y) / sy).astype(int)
        H, W = evt.shape
        out = {k: np.full(len(cells), np.nan) for k in
               ["lf_evt_mode", "lf_evt_mode_frac", "lf_evt_n_classes", "lf_evc_mode", "lf_evc_mode_frac"]}
        for i in range(len(cells)):
            if not (0 <= row[i] < H and 0 <= col[i] < W):
                continue
            sl = (slice(max(row[i] - half, 0), min(row[i] + half + 1, H)),
                  slice(max(col[i] - half, 0), min(col[i] + half + 1, W)))
            for name, arr in (("evt", evt), ("evc", evc)):
                if arr is None:
                    continue
                w = arr[sl].ravel(); w = w[w > 0]
                if w.size == 0:
                    continue
                u, c = np.unique(w, return_counts=True); j = c.argmax()
                out[f"lf_{name}_mode"][i] = u[j]; out[f"lf_{name}_mode_frac"][i] = c[j] / w.size
                if name == "evt":
                    out["lf_evt_n_classes"][i] = len(u)
        say(f"LANDFIRE: used {evt_path}  (cells covered: {int(np.isfinite(out['lf_evt_mode']).sum())}/{len(cells)})")
        return pd.DataFrame(out, index=cells.index)
    say("LANDFIRE: no usable raster found -> lf_* columns not created")
    return None


# ------------------------------------------------------------ groundwater ----
def idw_to_cells(cells, pts, col, k=5, power=2.0):
    v = pts.dropna(subset=[col, "lat", "lon"])
    if len(v) < k:
        return np.full(len(cells), np.nan), np.full(len(cells), np.nan)
    kx, ky = 111.32 * np.cos(np.radians(38.0)), 110.57
    cx, cy = cells.lon.values * kx, cells.lat.values * ky
    wx, wy = v.lon.values * kx, v.lat.values * ky
    d = np.sqrt((cx[:, None] - wx[None, :]) ** 2 + (cy[:, None] - wy[None, :]) ** 2)
    idx = np.argsort(d, axis=1)[:, :k]
    dd = np.take_along_axis(d, idx, 1)
    vals = v[col].values[idx]
    w = 1.0 / np.maximum(dd, 0.5) ** power
    return (w * vals).sum(1) / w.sum(1), dd[:, 0]


def groundwater_tables(data, cells):
    gw = pd.read_csv(f"{data}/USGS/USGS/kgs_wlevels.csv.csv")
    gw["date"] = pd.to_datetime(gw["MEASUREMENT_DATE_AND_TIME"], errors="coerce")
    gw = gw.dropna(subset=["date", "DEPTH_TO_WATER"])
    gw["USGS_ID"] = gw["USGS_ID"].astype(str)
    gw = gw[["USGS_ID", "date", "DEPTH_TO_WATER"]].rename(columns={"DEPTH_TO_WATER": "depth_ft"})
    gw["year"] = gw.date.dt.year
    gw["yf"] = gw.year + (gw.date.dt.dayofyear - 1) / 365.25
    wells = pd.read_csv(f"{data}/USGS/USGS/kgs_sites.csv.csv", dtype={"USGS_ID": str})
    wells = wells.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"})

    def slope(g, y0, min_n, min_span):
        h = g[g.yf >= y0]
        if len(h) >= min_n and (h.yf.max() - h.yf.min()) >= min_span:
            return float(np.polyfit(h.yf, h.depth_ft, 1)[0])
        return np.nan
    rows = []
    for wid, g in gw.groupby("USGS_ID"):
        rec = g[g.year >= 2019]
        rows.append(dict(USGS_ID=wid, n_obs=len(g), first_year=int(g.year.min()), last_year=int(g.year.max()),
                         slope_all_ft_yr=slope(g, 0, 8, 10), slope_2005_ft_yr=slope(g, 2005, 6, 8),
                         depth_recent_ft=float(rec.depth_ft.median()) if len(rec) else np.nan,
                         n_recent=len(rec)))
    wt = pd.DataFrame(rows).merge(wells[["USGS_ID", "lat", "lon", "DEPTH_OF_WELL"]], on="USGS_ID", how="left")
    cg = pd.DataFrame({"cell_id": cells.cell_id.values})
    cg["gw_trend_ft_yr"], cg["gw_nearest_trend_km"] = idw_to_cells(cells, wt, "slope_2005_ft_yr")
    cg["gw_depth_recent_ft"], cg["gw_nearest_depth_km"] = idw_to_cells(cells, wt, "depth_recent_ft")
    gw_out = gw[["USGS_ID", "date", "depth_ft", "year"]].copy()
    gw_out["date"] = gw_out["date"].dt.strftime("%Y-%m-%d")
    return gw_out, wells[["USGS_ID", "lat", "lon", "DEPTH_OF_WELL", "USE_OF_WATER_PRIMARY"]], wt, cg


# ---------------------------------------------------------------- yields ----
def tidy_quickstats(q):
    t = q[["Year", "Commodity", "Data Item", "Value"]].copy()
    item = t["Data Item"].str.upper()
    t["irrigation"] = np.where(item.str.contains("NON-IRRIGATED"), "non_irrigated",
                        np.where(item.str.contains("IRRIGATED"), "irrigated", "all"))
    t["measure"] = np.select([item.str.contains("ACRES PLANTED"), item.str.contains("ACRES HARVESTED"),
                              item.str.contains("PRODUCTION"), item.str.contains("YIELD")],
                             ["acres_planted", "acres_harvested", "production_bu", "yield_bu_ac"], "other")
    t["crop_code"] = t.Commodity.map({"WHEAT": 24, "CORN": 1, "SORGHUM": 4, "SOYBEANS": 5})
    w = t.pivot_table(index=["crop_code", "Commodity", "Year", "irrigation"], columns="measure",
                      values="Value", aggfunc="first").reset_index()
    w.columns.name = None
    return w.rename(columns={"Year": "year"})


# ------------------------------------------------------------------ main ----
def main(data=DATA_DIR, out=OUT_DB, et=ET_SOURCE, csv_dir=CSV_EXPORT, annual=ANNUAL_CSV):
    sp = os.path.join(data, "SpaceApps_RotationAdvisor")
    if not os.path.isdir(sp):
        raise SystemExit(f"Folder not found: {sp}  (check DATA_DIR at the top of the script)")
    if os.path.exists(out):
        try:
            os.remove(out)
        except PermissionError:
            raise SystemExit(f"'{out}' is locked (old Jupyter connection?). Restart the kernel or change OUT_DB")
    T = {}                                                        # tables to write

    # ---- static cells ------------------------------------------------------
    cells = pd.read_csv(f"{sp}/static_layers.csv").drop(columns=[".geo", "system:index"])
    cells["cell_key"] = norm_id(cells.cell_id)
    assert cells.cell_key.is_unique, "cell_id collision after removing commas"
    say("cells:", len(cells))

    # ---- CDL (ROUND, never truncate) --------------------------------------
    raw = read_many(f"{sp}/cdl_*.csv", ["cell_id", "cdl_code", "year"])
    n_trunc = int((np.floor(raw.cdl_code) != raw.cdl_code.round()).sum())
    dev = (raw.cdl_code - raw.cdl_code.round()).abs()
    raw["cdl_ambiguous"] = (dev > 0.05).astype(int)               # mean of several classes (mostly urban)
    raw["cdl_code"] = raw.cdl_code.round().astype("Int64")
    bad = (raw.cdl_ambiguous == 1) & (~raw.cdl_code.isin(DEVELOPED))
    raw.loc[bad, "cdl_code"] = pd.NA
    say(f"CDL: {n_trunc} of {len(raw)} codes ({100 * n_trunc / len(raw):.1f}%) would have been corrupted "
        f"by astype(int); fixed by rounding. Ambiguous mixed cells set to NaN: {int(bad.sum())}")
    cdl = raw.copy()
    cdl["crop"] = cdl.cdl_code.map(CDL_NAMES).fillna("Other")
    T["cdl"] = cdl.astype({"cdl_code": "float"})

    cdl_w = cdl.pivot_table(index="cell_id", columns="year", values="cdl_code", aggfunc="first")
    cdl_w = cdl_w.astype(float)
    yrs = [y for y in range(2018, 2025) if y in cdl_w.columns]
    seq = cdl_w[yrs].apply(lambda r: "-".join("" if pd.isna(v) else str(int(v)) for v in r), axis=1)
    rot = pd.DataFrame({"cell_id": cdl_w.index, "seq_2018_2024": seq.values,
                        "rotation_class": [classify_rotation(r) for r in cdl_w[yrs].values],
                        "grass_years": (cdl_w == 176).sum(axis=1).values,
                        "cropland_years": cdl_w.isin(CROPLAND).sum(axis=1).values})
    cells = cells.merge(rot, on="cell_id", how="left")
    say("rotation_class (cells):", cells.rotation_class.value_counts().to_dict())

    # ---- groundwater -------------------------------------------------------
    gw, wells, wtr, cg = groundwater_tables(data, cells)
    T.update(groundwater=gw, wells=wells, well_trends=wtr, cell_groundwater=cg)
    cells = cells.merge(cg, on="cell_id", how="left")
    ok = wtr.slope_2005_ft_yr.notna()
    say(f"groundwater: {len(gw)} measurements, {wtr.shape[0]} wells, {int(ok.sum())} wells with a 2005+ trend; "
        f"median trend {wtr.slope_2005_ft_yr.median():+.2f} ft/yr (+ = deeper); "
        f"cells farther than 15 km from a trend well: {int((cg.gw_nearest_trend_km > 15).sum())}")

    # ---- LANDFIRE ----------------------------------------------------------
    lf = landfire_features(cells, data)
    if lf is not None:
        cells = pd.concat([cells, lf], axis=1)

    # ---- monthly NASA features + ECOSTRESS + OpenET -----------------------
    m = read_many(f"{sp}/monthly_*.csv").drop(columns=[".geo", "system:index"])
    eco = read_many(f"{sp}/ecostress_*.csv").drop(columns=[".geo", "system:index"])
    oet = read_many(f"{sp}/openet_*.csv").drop(columns=[".geo", "system:index"])
    n0 = len(m)
    keys = ["cell_id", "year", "month"]
    m = (m.merge(eco, on=keys, how="left").merge(oet, on=keys, how="left")
          .drop(columns=["lat", "lon"], errors="ignore"))
    assert len(m) == n0, "duplicate keys while merging monthly/ecostress/openet"
    for c in ("LST_max_C", "LST_mean_C"):                        # scene artefacts (saw 134 C)
        m.loc[(m[c] > 70) | (m[c] < -40), c] = np.nan
    m["month"] = m["month"].astype(int); m["year"] = m["year"].astype(int)
    T["monthly"] = m
    say("monthly rows:", len(m))

    # ---- county OpenET -----------------------------------------------------
    f = glob.glob(os.path.join(data, "OpenET_Finney*.csv"))[0]
    o = pd.read_csv(f).drop(columns=[".geo", "system:index"])
    o["et_spread_mm"] = (o.et_ensemble_mad - o.et_ensemble_sam).abs()
    T["openet_county"] = o

    # ---- annual indicators (optional file) ---------------------------------
    if annual is None:
        cand = glob.glob(os.path.join(data, "annual_environmental_indicators*.csv")) + \
               glob.glob("annual_environmental_indicators*.csv")
        annual = cand[0] if cand else None
    an = None
    if annual and os.path.exists(annual):
        an = pd.read_csv(annual)
        an["cell_key"] = norm_id(an.cell_id)
        key2id = dict(zip(cells.cell_key, cells.cell_id))
        an["cell_id"] = an.cell_key.map(key2id)
        unmatched = int(an.cell_id.isna().sum())
        chk = an.drop_duplicates("cell_key").merge(cells[["cell_key", "lat", "lon"]], on="cell_key",
                                                   suffixes=("", "_c"))
        dpos = max((chk.lat - chk.lat_c).abs().max(), (chk.lon - chk.lon_c).abs().max())
        say(f"annual file: re-keyed ids, unmatched rows={unmatched}, max lat/lon difference={dpos:.6f} deg")
        if unmatched or dpos > 1e-3:
            raise SystemExit("annual file does not match the grid -> stop (check cell ids)")
        drop = ["cell_key", "lat", "lon", "clay_0cm", "organic_carbon_0cm", "sand_0cm", "n_months"]
        an = an.drop(columns=drop)                              # soil columns are county constants (useless)
        an = an.rename(columns={c: f"an_{c}" for c in an.columns if c not in ("cell_id", "year")})
        T["annual"] = an

   # ---- yields (QuickStats) + prices -------------------------------------
    q = pd.read_csv(f"{data}/quickstats.csv")
    q["Value"] = pd.to_numeric(q["Value"].astype(str).str.replace(",", ""), errors="coerce")

    # إضافة أسعار الفصة ($/TON)
    df_hay = pd.read_csv("CC4ABE41-55F0-3587-AC69-C5EA4F07955A.csv")
    alfalfa = df_hay[df_hay["Data Item"] == "HAY, ALFALFA - PRICE RECEIVED, MEASURED IN $ / TON"][
        ["Year", "Commodity", "Data Item", "Value"]
    ].copy()

    # إضافة أسعار السورغو وتحويلها من ($/CWT) إلى ($/BU) بالضرب في 0.56
    df_sorg = pd.read_csv("DC3675C4-24D7-3D5E-93EE-15F1CB43C3C2.csv")
    sorghum = df_sorg[["Year", "Commodity", "Data Item", "Value"]].copy()
    sorghum["Value"] = pd.to_numeric(sorghum["Value"], errors="coerce") * 0.56
    sorghum["Data Item"] = "SORGHUM, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU"

    # دمج الأسعار الجديدة مع جدول yields
    q_full = pd.concat([q[["Year", "Commodity", "Data Item", "Value"]], alfalfa, sorghum], ignore_index=True)

    T["yields"] = q_full
    T["yields_clean"] = tidy_quickstats(q)

    pr_file = sorted(glob.glob(os.path.join(data, "Kansas_All_Crops*")))[0]
    pr = pd.read_csv(pr_file)                                   # '.xls' is really a CSV
    nd = int(pr.duplicated(["Date", "Ticker"]).sum())
    pr = pr.drop_duplicates(["Date", "Ticker"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    T["prices"] = pr
    pw = pr.pivot(index="Date", columns="Crop_Name", values="Close").sort_index()
    T["prices_wide"] = pw.reset_index()
    pm = pr.assign(ym=pr.Date.str[:7]).groupby(["ym", "Crop_Name"]).Close.mean().unstack().reset_index()
    T["prices_monthly"] = pm
    say(f"prices: {pr.Date.min()} -> {pr.Date.max()}, duplicates removed={nd}, "
        f"missing closes={pw.isna().sum().to_dict()}  (units: cents/bushel)")

    # ---- FEATURE TABLE (cell x year) --------------------------------------
    g = m[m.month.isin(GROWING)]
    gs = g.groupby(["cell_id", "year"]).agg(
        ndvi_gs_mean=("NDVI", "mean"), ndvi_gs_max=("NDVI", "max"),
        ndmi_gs_mean=("NDMI", "mean"), et_model_gs=("ET_mm", "sum"),
        openet_gs=("openet_mm", "sum"), precip_gs=("precip_mm", "sum"),
        sm_root_gs=("sm_rootzone", "mean"), sm_surf_gs=("sm_surface", "mean"),
        lst_max_gs=("LST_max_C", "max"), lst_mean_gs=("LST_mean_C", "mean"),
        t2m_gs=("T2m_C", "mean")).reset_index()
    low = m.assign(bare=(m.NDVI < 0.25).astype(int))
    bare = low.groupby(["cell_id", "year"]).bare.sum().rename("bare_months").reset_index()
    bare_gs = (low[low.month.isin(GROWING)].groupby(["cell_id", "year"]).bare.sum()
               .rename("bare_months_gs").reset_index())
    hot = (g.assign(hot=(g.LST_max_C > 40).astype(int)).groupby(["cell_id", "year"]).hot.sum()
           .rename("hot_months").reset_index())
    feat = gs.merge(bare).merge(bare_gs).merge(hot)
    feat["et_gs"] = feat["openet_gs"] if et == "openet" else feat["et_model_gs"]
    feat["water_deficit_mm"] = (feat["et_gs"] - feat["precip_gs"]).clip(lower=0)

    long = cdl_w.reset_index().melt(id_vars="cell_id", var_name="year", value_name="cdl_code")
    long["year"] = long["year"].astype(int)          # works on every pandas version
    for k in (1, 2, 3):
        sh = long.copy(); sh["year"] += k
        feat = feat.merge(sh.rename(columns={"cdl_code": f"crop_prev{k}"}), on=["cell_id", "year"], how="left")
    feat = feat.merge(long, on=["cell_id", "year"], how="left")
    feat = feat.merge(cells.drop(columns=["cell_key"]), on="cell_id", how="left")
    feat["crop"] = feat.cdl_code.map(CDL_NAMES).fillna("Other")
    if an is not None:
        feat = feat.merge(an, on=["cell_id", "year"], how="left")
    assert len(feat) == len(gs), "feature table changed size during merges"
    T["features"] = feat
    T["cells"] = cells.drop(columns=["cell_key"])
    say("features:", feat.shape)

    # ---- QA REPORT ----------------------------------------------------------
    say("\n=== QA REPORT ===")
    say("crop cell-years 2018-2024:", feat.crop.value_counts().head(8).to_dict())
    n_soy = int((feat.cdl_code == 5).sum())
    if n_soy < 300:
        say(f"WARNING: soybeans have only {n_soy} cell-years -> too few to learn soybean behaviour")
    ann = m.groupby(["cell_id", "year"]).agg(et_model=("ET_mm", "sum"), et_openet=("openet_mm", "sum"),
                                             p=("precip_mm", "sum")).dropna()
    say(f"ET sources (annual): ET_mm mean={ann.et_model.mean():.0f} mm vs OpenET mean={ann.et_openet.mean():.0f} mm, "
        f"corr={ann.et_model.corr(ann.et_openet):.2f}  -> they disagree; keep --et fixed and say which one you use")
    say(f"precip annual mean={ann.p.mean():.0f} mm  (check the source against a weather station)")
    na = feat[["ndvi_gs_max", "et_gs", "precip_gs", "sm_root_gs", "lst_max_gs", "cdl_code", "crop_prev1",
               "crop_prev2"]].isna().mean().round(3).to_dict()
    say("share of NaN in key features:", na)
    if lf is not None and "lf_evt_mode" in cells:
        c24 = cdl_w[2024].rename("cdl24").reset_index().merge(cells[["cell_id", "lf_evt_mode"]], on="cell_id")
        top = c24.lf_evt_mode.value_counts().head(6).index
        tab = {int(t): c24[c24.lf_evt_mode == t].cdl24.round().map(CDL_NAMES).value_counts().head(2).to_dict()
               for t in top}
        say("LANDFIRE EVT mode vs CDL-2024 (empirical meaning of each code):", tab)

    # ---- write DB -----------------------------------------------------------
    con = sqlite3.connect(out)
    for name, df in T.items():
        df.to_sql(name, con, index=False)
    for t, c in [("cdl", "cell_id"), ("monthly", "cell_id"), ("features", "cell_id"),
                 ("features", "year"), ("cells", "cell_id"), ("groundwater", "USGS_ID")]:
        con.execute(f"CREATE INDEX idx_{t}_{c} ON {t}({c})")
    con.commit()
    if csv_dir and csv_dir.lower() != "none":
        os.makedirs(csv_dir, exist_ok=True)
        for name, df in T.items():
            df.to_csv(os.path.join(csv_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")
    con.close()
    with open("qa_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(LOG))
    say("ET source used:", et, "| QA report: qa_report.txt")
    say("OK ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--out", default=OUT_DB)
    ap.add_argument("--et", default=ET_SOURCE, choices=["openet", "model"])
    ap.add_argument("--csv", default=CSV_EXPORT)
    ap.add_argument("--annual", default=ANNUAL_CSV)
    a, _ = ap.parse_known_args()
    main(a.data, a.out, a.et, a.csv, a.annual)

