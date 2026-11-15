"""Combined Groundwater Advisor App with Worldwide USGS Live API, Local Fallback, & Multilingual UI (EN / FR / AR).

Install : pip install streamlit folium streamlit-folium pandas numpy requests
Run UI  : streamlit run combined_groundwater_app.py
"""
import os
import requests
import numpy as np
import pandas as pd
import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium
from model import predict_from_latlon

# =====================================================================
# 0. TRANSLATIONS (English, Français, العربية)
# =====================================================================
TRANSLATIONS = {
    "en": {
        "lang_label": "Language / Langue / اللغة",
        "title": "🌍 Worldwide Groundwater Advisor & USGS Live Explorer",
        "subtitle": (
            "This tool evaluates groundwater anywhere in the world using **live USGS water APIs** "
            "(with local dataset fallback). Draw your field polygon on the map below to fetch "
            "regional well telemetry and receive tailored advisories."
        ),
        "has_well_q": "Do you already have a well on your land?",
        "no": "No",
        "yes": "Yes",
        "mode_yes": (
            "**Mode: Future Water Depth Forecast (Existing Well)**\n\n"
            "We will analyze regional well telemetry and project how deep your water level will go in the future "
            "based on 10-year historical trends."
        ),
        "mode_no": (
            "**Mode: Groundwater Detection & Drilling Advisory (No Well)**\n\n"
            "We will detect whether viable groundwater exists beneath your field, estimate depth, and advise whether to build a well."
        ),
        "horizon_slider": "Projection horizon (years):",
        "well_depth_input": "Your current well total depth (ft) [0 = skip]:",
        "draw_header": "Draw your field boundary anywhere on the map",
        "draw_caption": "Use the polygon or rectangle tool on the left side of the map to outline your land, then click Analyze.",
        "btn_yes": "Analyze Future Well Depth",
        "btn_no": "Detect Groundwater & Get Advice",
        "spinner": "Fetching live USGS telemetry & analyzing hydrogeology...",
        "results_header": "Analysis Results",
        "metric_evidence": "Evidence Level",
        "metric_conf": "Confidence",
        "metric_area": "Field Area",
        "level_strong": "STRONG",
        "level_moderate": "MODERATE",
        "level_weak": "WEAK",
        "level_none": "NONE",
        "well_inside_notice": "**Notice:** Detected **{n} monitored well(s)** inside your drawn field boundary!",
        "well_inside_item": "- **Well ID:** `{wid}` | Current Depth: **{depth} ft**",
        "insufficient_forecast": "Insufficient monitoring data in this bounding box to forecast future water depth.",
        "est_depth_header": "### Estimated Regional Water Depth: **{mean} ft** (range {low}–{high} ft)",
        "trend_unavail": "Future Trend Unavailable: Insufficient historical measurements for annual trend calculation.",
        "proj_depth_metric": "Projected Depth",
        "declining_warn": "**Declining Water Table:** Local groundwater drops by ~**{trend:.2f} ft/year**. Plan for deeper pumping.",
        "stable_ok": "**Stable/Recovering Water Table:** Annual trend is stable ({trend:+.2f} ft/year).",
        "dry_alert": "**Critical Well Depth Alert:** Your well's total depth ({wdepth} ft) is shallower than the projected water depth ({pdepth} ft) in {yrs} years. High risk of running dry.",
        "low_col_warn": "**Low Water Column Warning:** Only ~**{rem} ft** of water column would remain after {yrs} years—pumping efficiency may drop.",
        "depth_ok_info": "**Well Depth Check:** With a total depth of {wdepth} ft, approximately **{rem} ft** of water column would remain in {yrs} years.",
        "detected_strong_mod": (
            "**Groundwater Detected ({lvl} Evidence):** Live regional telemetry indicates viable groundwater. "
            "**Advice:** Proceed with planning and drilling a well, starting with an on-site hydrogeological survey."
        ),
        "detected_weak": (
            "**Weak Groundwater Evidence:** Only distant monitored wells were found. "
            "**Advice:** Do not start drilling yet—consult a local hydrogeologist first."
        ),
        "detected_none": (
            "**No Usable Groundwater Evidence Found:** Region lacks active monitored wells in USGS telemetry or local datasets. "
            "**Advice:** Do not rely on this tool to build a well here without local geophysical testing."
        ),
        "est_drill_depth": "Estimated Drilling Depth to Water Table: **{mean} ft** (range: {low}–{high} ft)",
        "notices_header": "Important Notices & Cautions",
        "caution_base": "Indicative only: this is not a drilling guarantee. Confirm with a licensed hydrogeologist / test drilling before investing.",
        "caution_none": "Field is outside active well coverage (nearest well > 50 km).",
        "caution_weak": "Nearest active well is {d:.1f} km away; local conditions may differ.",
        "caution_trend": "Water table is falling about {t:.1f} units/yr in nearby wells; plan for deeper pumping depth.",
        "caution_nodata": "No wells found in live USGS query or local fallback.",
        "expander_wells": "View Nearest Monitored Wells Data",
    },
    "fr": {
        "lang_label": "Language / Langue / اللغة",
        "title": "🌍 Conseiller Mondial des Eaux Souterraines & Explorateur USGS",
        "subtitle": (
            "Cet outil évalue les eaux souterraines partout dans le monde via les **API en direct de l'USGS** "
            "(avec repli sur les données locales). Dessinez le polygone de votre parcelle sur la carte pour "
            "récupérer les mesures des puits voisins et obtenir des recommandations personnalisées."
        ),
        "has_well_q": "Avez-vous déjà un puits sur votre terrain ?",
        "no": "Non",
        "yes": "Oui",
        "mode_yes": (
            "**Mode : Prévision de la profondeur future de l'eau (Puits existant)**\n\n"
            "Nous analyserons la télémétrie des puits régionaux pour projeter l'évolution future du niveau d'eau "
            "sur la base des tendances historiques sur 10 ans."
        ),
        "mode_no": (
            "**Mode : Détection des eaux souterraines & Conseil de forage (Sans puits)**\n\n"
            "Nous détecterons la présence d'eaux souterraines viables sous votre parcelle, estimerons la profondeur "
            "et vous conseillerons sur l'opportunité de forer un puits."
        ),
        "horizon_slider": "Horizon de projection (années) :",
        "well_depth_input": "Profondeur totale actuelle de votre puits (pieds / ft) [0 = ignorer] :",
        "draw_header": "Dessinez les limites de votre terrain n'importe où sur la carte",
        "draw_caption": "Utilisez l'outil polygone ou rectangle à gauche de la carte pour délimiter votre terrain, puis cliquez sur Analyser.",
        "btn_yes": "Analyser la profondeur future du puits",
        "btn_no": "Détecter l'eau souterraine & Obtenir un conseil",
        "spinner": "Récupération des données USGS en direct et analyse hydrogéologique...",
        "results_header": "Résultats de l'analyse",
        "metric_evidence": "Niveau de preuve",
        "metric_conf": "Confiance",
        "metric_area": "Superficie",
        "level_strong": "FORT",
        "level_moderate": "MODÉRÉ",
        "level_weak": "FAIBLE",
        "level_none": "AUCUN",
        "well_inside_notice": "**Avis :** **{n} puits surveillé(s)** détecté(s) directement à l'intérieur de votre parcelle tracée !",
        "well_inside_item": "- **ID du puits :** `{wid}` | Profondeur actuelle : **{depth} ft**",
        "insufficient_forecast": "Données de surveillance insuffisantes dans cette zone pour prévoir la profondeur future.",
        "est_depth_header": "### Profondeur régionale estimée de l'eau : **{mean} ft** (plage {low}–{high} ft)",
        "trend_unavail": "Tendance future indisponible : mesures historiques insuffisantes pour calculer la tendance annuelle.",
        "proj_depth_metric": "Profondeur projetée",
        "declining_warn": "**Baisse de la nappe phréatique :** Le niveau baisse d'environ **{trend:.2f} ft/an**. Prévoyez un pompage plus profond.",
        "stable_ok": "**Nappe stable / en hausse :** La tendance annuelle est stable ({trend:+.2f} ft/an).",
        "dry_alert": "**Alerte critique de profondeur :** La profondeur totale de votre puits ({wdepth} ft) est inférieure à la profondeur projetée ({pdepth} ft) dans {yrs} ans. Risque élevé d'assèchement.",
        "low_col_warn": "**Avertissement de faible colonne d'eau :** Il ne resterait qu'environ **{rem} ft** d'eau après {yrs} ans — le rendement de pompage risque de chuter.",
        "depth_ok_info": "**Vérification du puits :** Avec une profondeur totale de {wdepth} ft, il resterait environ **{rem} ft** de colonne d'eau dans {yrs} ans.",
        "detected_strong_mod": (
            "**Eau souterraine détectée (Preuve {lvl}) :** Les données régionales indiquent une ressource viable. "
            "**Conseil :** Vous pouvez planifier le forage d'un puits, en commençant par une étude hydrogéologique sur site."
        ),
        "detected_weak": (
            "**Preuve faible d'eau souterraine :** Seuls des puits éloignés ont été trouvés. "
            "**Conseil :** Ne commencez pas le forage — consultez d'abord un hydrogéologue local."
        ),
        "detected_none": (
            "**Aucune preuve exploitable d'eau souterraine :** Cette zone manque de puits actifs dans l'USGS ou les données locales. "
            "**Conseil :** Ne vous fiez pas uniquement à cet outil pour forer ici sans étude géophysique locale."
        ),
        "est_drill_depth": "Profondeur de forage estimée jusqu'à la nappe : **{mean} ft** (plage : {low}–{high} ft)",
        "notices_header": "Avis importants & Précautions",
        "caution_base": "À titre indicatif uniquement : ceci n'est pas une garantie de forage. Confirmez avec un hydrogéologue agréé avant d'investir.",
        "caution_none": "La parcelle est hors de la zone de couverture active (puits le plus proche > 50 km).",
        "caution_weak": "Le puits actif le plus proche est à {d:.1f} km ; les conditions locales peuvent différer.",
        "caution_trend": "La nappe baisse d'environ {t:.1f} unités/an dans les puits voisins ; prévoyez un pompage plus profond.",
        "caution_nodata": "Aucun puits trouvé dans la requête USGS en direct ni dans les données locales.",
        "expander_wells": "Voir les données des puits surveillés les plus proches",
    },
    "ar": {
        "lang_label": "Language / Langue / اللغة",
        "title": "🌍 المستشار العالمي للمياه الجوفية ومستكشف بيانات USGS الحية",
        "subtitle": (
            "تُقيّم هذه الأداة المياه الجوفية في أي مكان في العالم باستخدام **واجهات برمجة تطبيقات USGS الحية** "
            "(مع الرجوع التلقائي للبيانات المحلية). ارسم حدود حقلك على الخريطة أدناه لجلب بيانات الآبار القريبة والحصول على توصيات مخصصة."
        ),
        "has_well_q": "هل لديك بئر بالفعل في أرضك؟",
        "no": "لا",
        "yes": "نعم",
        "mode_yes": (
            "**الوضع: التنبؤ بعمق المياه المستقبلي (بئر موجود)**\n\n"
            "سنقوم بتحليل بيانات الآبار الإقليمية وتوقع العمق الذي سيصل إليه مستوى المياه في المستقبل "
            "بناءً على الاتجاهات التاريخية لآخر 10 سنوات."
        ),
        "mode_no": (
            "**الوضع: الكشف عن المياه الجوفية وتقديم المشورة للحفر (لا يوجد بئر)**\n\n"
            "سنقوم بالكشف عن وجود مياه جوفية صالحة أسفل حقلك، وتقدير العمق، وتقديم المشورة حول جدوى حفر بئر جديد."
        ),
        "horizon_slider": "أفق التنبؤ المستقبلي (بالسنوات):",
        "well_depth_input": "العمق الكلي الحالي لبئرك (بالقدم ft) [0 = تخطي]:",
        "draw_header": "ارسم حدود حقلك في أي مكان على الخريطة",
        "draw_caption": "استخدم أداة المضلع (Polygon) أو المستطيل على يسار الخريطة لتحديد أرضك، ثم اضغط على زر التحليل.",
        "btn_yes": "تحليل عمق البئر المستقبلي",
        "btn_no": "الكشف عن المياه الجوفية والحصول على المشورة",
        "spinner": "جاري جلب بيانات USGS الحية وتحليل الهيدروجيولوجيا...",
        "results_header": "نتائج التحليل",
        "metric_evidence": "مستوى الأدلة",
        "metric_conf": "نسبة الثقة",
        "metric_area": "مساحة الحقل",
        "level_strong": "قوي",
        "level_moderate": "متوسط",
        "level_weak": "ضعيف",
        "level_none": "معدوم",
        "well_inside_notice": "**تنبيه:** تم رصد **{n} بئر (آبار) مراقبة** تقع مباشرة داخل حدود حقلك المرسوم!",
        "well_inside_item": "- **معرّف البئر:** `{wid}` | عمق المياه الحالي: **{depth} قدم**",
        "insufficient_forecast": "بيانات المراقبة في هذا النطاق غير كافية للتنبؤ بعمق المياه المستقبلي.",
        "est_depth_header": "### العمق الإقليمي المقدر للمياه: **{mean} قدم** (النطاق {low}–{high} قدم)",
        "trend_unavail": "الاتجاه المستقبلي غير متاح: القياسات التاريخية غير كافية لحساب معدل التغير السنوي.",
        "proj_depth_metric": "العمق المتوقع مستقبلاً",
        "declining_warn": "**انخفاض منسوب المياه الجوفية:** ينخفض المنسوب بمقدار ~**{trend:.2f} قدم/سنة**. خطط لتعميق المضخة مستقبلاً.",
        "stable_ok": "**منسوب مياه مستقر / متعافٍ:** الاتجاه السنوي مستقر ({trend:+.2f} قدم/سنة).",
        "dry_alert": "**تنبيه حرج لعمق البئر:** العمق الكلي لبئرك ({wdepth} قدم) أقل من عمق المياه المتوقع ({pdepth} قدم) بعد {yrs} سنوات. خطر مرتفع لجفاف البئر.",
        "low_col_warn": "**تحذير انخفاض عمود الماء:** سيتبقى حوالي **{rem} قدم** فقط من عمود الماء بعد {yrs} سنوات — قد تنخفض كفاءة الضخ.",
        "depth_ok_info": "**فحص عمق البئر:** بعمق كلي يبلغ {wdepth} قدم، سيتبقى حوالي **{rem} قدم** من عمود الماء بعد {yrs} سنوات.",
        "detected_strong_mod": (
            "**تم رصد مياه جوفية (دليل {lvl}):** تشير البيانات الإقليمية الحية إلى وجود مياه جوفية واعدة. "
            "**النصيحة:** يُنصح بالمضي قدماً في التخطيط لحفر بئر، بدءاً بإجراء مسح هيدروجيولوجي ميداني."
        ),
        "detected_weak": (
            "**أدلة ضعيفة على المياه الجوفية:** تم العثور على آبار مراقبة بعيدة فقط. "
            "**النصيحة:** لا تبدأ الحفر بعد — استشر مختصاً في الهيدروجيولوجيا المحلية أولاً."
        ),
        "detected_none": (
            "**لا توجد أدلة كافية على المياه الجوفية:** تفتقر المنطقة لآبار مراقبة نشطة في بيانات USGS أو البيانات المحلية. "
            "**النصيحة:** لا تعتمد على هذه الأداة لحفر بئر هنا دون إجراء فحوصات جيوفيزيائية محلية."
        ),
        "est_drill_depth": "عمق الحفر المقدر للوصول إلى المياه: **{mean} قدم** (النطاق: {low}–{high} قدم)",
        "notices_header": "ملاحظات وتحذيرات هامة",
        "caution_base": "للاسترشاد فقط: هذه النتائج ليست ضماناً للحفر. يرجى التأكد مع مهندس هيدروجيولوجي مرخص / حفر تجريبي قبل الاستثمار.",
        "caution_none": "الحقل يقع خارج نطاق تغطية الآبار النشطة (أقرب بئر > 50 كم).",
        "caution_weak": "أقرب بئر نشط يبعد {d:.1f} كم؛ قد تختلف الظروف الجيولوجية محلياً.",
        "caution_trend": "ينخفض منسوب المياه بحوالي {t:.1f} وحدة/سنة في الآبار المجاورة؛ خطط لعمق ضخ أكبر.",
        "caution_nodata": "لم يتم العثور على آبار في استعلام USGS الحي أو البيانات المحلية.",
        "expander_wells": "عرض بيانات أقرب الآبار المراقبة",
    },
}

# =====================================================================
# 1. USGS API & DATA LOADING LOGIC (Worldwide + Local Fallback)
# =====================================================================
ACTIVE_WINDOW_DAYS = 730
TREND_YEARS = 10
KM_PER_DEG_LAT = 111.32


def _to_xy(lat, lon, lat0, lon0):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    x = (lon - lon0) * KM_PER_DEG_LAT * np.cos(np.radians(lat0))
    y = (lat - lat0) * KM_PER_DEG_LAT
    return x, y


def _point_in_poly(px, py, vx, vy):
    inside = np.zeros(len(px), bool)
    n = len(vx)
    j = n - 1
    for i in range(n):
        cond = ((vy[i] > py) != (vy[j] > py)) & (
            px < (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-12) + vx[i]
        )
        inside ^= cond
        j = i
    return inside


def _dist_to_segments(px, py, vx, vy):
    d = np.full(len(px), np.inf)
    n = len(vx)
    for i in range(n):
        ax, ay, bx, by = vx[i], vy[i], vx[(i + 1) % n], vy[(i + 1) % n]
        abx, aby = bx - ax, by - ay
        t = ((px - ax) * abx + (py - ay) * aby) / (abx**2 + aby**2 + 1e-12)
        t = np.clip(t, 0, 1)
        d = np.minimum(d, np.hypot(px - (ax + t * abx), py - (ay + t * aby)))
    return d


def polygon_area_km2(vx, vy):
    return 0.5 * abs(np.dot(vx, np.roll(vy, -1)) - np.dot(vy, np.roll(vx, -1)))


import sqlite3

@st.cache_data
def load_local_data_from_db(db_path="farm.db"):
    """قراءة قياسات الآبار مباشرة من قاعدة بيانات farm.db"""
    try:
        con = sqlite3.connect(db_path)
        m = pd.read_sql("SELECT USGS_ID, date, depth_ft AS DEPTH_TO_WATER FROM groundwater", con)
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        
        s = pd.read_sql("""
            SELECT w.USGS_ID, w.lat AS LATITUDE, w.lon AS LONGITUDE, 
                   w.DEPTH_OF_WELL, w.USE_OF_WATER_PRIMARY, 
                   COALESCE(t.n_obs, 10) AS measurement_count
            FROM wells w
            LEFT JOIN well_trends t ON w.USGS_ID = t.USGS_ID
        """, con)
        con.close()
        return m, s
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def fetch_usgs_data_for_bbox(min_lat, min_lon, max_lat, max_lon):
    """Fetch live USGS groundwater site summaries and measurements for a bounding box."""
    try:
        site_url = (
            f"https://waterservices.usgs.gov/nwis/site/?format=rdb"
            f"&bBox={min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}&siteStatus=all"
        )
        res = requests.get(site_url, timeout=10)
        if res.status_code != 200:
            return None, None

        lines = [line for line in res.text.splitlines() if not line.startswith("#")]
        if len(lines) < 3:
            return None, None

        header = lines[0].split("\t")
        data_rows = [l.split("\t") for l in lines[2:]]
        sites_df = pd.DataFrame(data_rows, columns=header)

        if "site_no" not in sites_df.columns or "dec_lat_va" not in sites_df.columns:
            return None, None

        summary = pd.DataFrame({
            "USGS_ID": sites_df["site_no"],
            "LATITUDE": pd.to_numeric(sites_df["dec_lat_va"], errors="coerce"),
            "LONGITUDE": pd.to_numeric(sites_df["dec_long_va"], errors="coerce"),
            "DEPTH_OF_WELL": 300.0,
            "USE_OF_WATER_PRIMARY": "Unknown",
            "measurement_count": 10,
        }).dropna(subset=["LATITUDE", "LONGITUDE"])

        if len(summary) == 0:
            return None, None

        site_ids = ",".join(summary["USGS_ID"].head(50).tolist())
        gw_url = f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_ids}&parameterCd=72019"
        gw_res = requests.get(gw_url, timeout=10)

        measurements = []
        if gw_res.status_code == 200:
            gw_json = gw_res.json()
            time_series = gw_json.get("value", {}).get("timeSeries", [])
            for ts in time_series:
                try:
                    s_code = ts["sourceInfo"]["siteCode"][0]["value"]
                    values = ts["values"][0]["value"]
                    for v in values:
                        val = float(v["value"])
                        dt = pd.to_datetime(v["dateTime"])
                        measurements.append({
                            "USGS_ID": s_code,
                            "MEASUREMENT_DATE_AND_TIME": dt,
                            "DEPTH_TO_WATER": val,
                        })
                except Exception:
                    continue

        meas_df = pd.DataFrame(measurements)
        if len(meas_df) > 0:
            meas_df["date"] = pd.to_datetime(meas_df["MEASUREMENT_DATE_AND_TIME"])
        return meas_df, summary
    except Exception:
        return None, None


def _well_stats(m, s, ref_date):
    rows = []
    if len(m) == 0 or len(s) == 0:
        return pd.DataFrame(
            columns=[
                "USGS_ID", "LATITUDE", "LONGITUDE", "DEPTH_OF_WELL",
                "latest_depth", "days_since", "trend_ft_yr", "inside", "dist_km", "active"
            ]
        )

    for wid, g in m.groupby("USGS_ID"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        r = {
            "USGS_ID": wid,
            "latest_depth": last["DEPTH_TO_WATER"],
            "latest_date": last["date"],
            "days_since": (ref_date - last["date"]).days if pd.notnull(last["date"]) else 0,
            "trend_ft_yr": np.nan,
        }
        h = g[g["date"] >= ref_date - pd.Timedelta(days=365.25 * TREND_YEARS)] if pd.notnull(ref_date) else g
        if len(h) >= 3 and (h["date"].max() - h["date"].min()).days > 300:
            t = (h["date"] - h["date"].min()).dt.days / 365.25
            r["trend_ft_yr"] = np.polyfit(t, h["DEPTH_TO_WATER"], 1)[0]
        rows.append(r)

    st_df = pd.DataFrame(rows).merge(
        s[["USGS_ID", "LATITUDE", "LONGITUDE", "DEPTH_OF_WELL",
           "USE_OF_WATER_PRIMARY", "measurement_count"]],
        on="USGS_ID",
        how="inner",
    )
    if "DEPTH_OF_WELL" in st_df.columns and "latest_depth" in st_df.columns:
        st_df["water_column"] = st_df["DEPTH_OF_WELL"] - st_df["latest_depth"]
    else:
        st_df["water_column"] = 100.0
    return st_df


def analyze_polygon(polygon, m, s, tr, k=8, ref_date=None):
    poly = np.asarray(polygon, float)
    if len(poly) < 3:
        raise ValueError("Polygon needs at least 3 vertices (lat, lon).")

    ref_date = ref_date or (m["date"].max() if len(m) > 0 and "date" in m.columns else pd.Timestamp.now())
    lat0, lon0 = poly[:, 0].mean(), poly[:, 1].mean()
    vx, vy = _to_xy(poly[:, 0], poly[:, 1], lat0, lon0)

    st_df = _well_stats(m, s, ref_date)
    if len(st_df) == 0:
        return {
            "evidence_level": "none",
            "confidence": 0,
            "cautions": [tr["caution_base"], tr["caution_nodata"]],
            "depth_estimate": None,
            "trend_ft_yr": np.nan,
            "field_area_km2": round(polygon_area_km2(vx, vy), 3),
            "wells_inside_list": [],
            "active_within_5km": 0,
            "active_within_10km": 0,
            "nearest_active_km": None,
            "nearest_wells": pd.DataFrame(),
        }

    wx, wy = _to_xy(st_df["LATITUDE"], st_df["LONGITUDE"], lat0, lon0)
    st_df["inside"] = _point_in_poly(wx, wy, vx, vy)
    st_df["dist_km"] = np.where(st_df["inside"], 0.0, _dist_to_segments(wx, wy, vx, vy))
    st_df["active"] = st_df["days_since"] <= ACTIVE_WINDOW_DAYS

    active = (
        st_df[st_df["active"]].sort_values("dist_km")
        if len(st_df[st_df["active"]]) > 0
        else st_df.sort_values("dist_km")
    )
    nearest = active.head(k).copy()

    n5 = int((active["dist_km"] <= 5).sum()) if len(active) > 0 else 0
    n10 = int((active["dist_km"] <= 10).sum()) if len(active) > 0 else 0
    d_min = float(active["dist_km"].min()) if len(active) else np.inf

    est = None
    if len(nearest):
        w = 1 / (nearest["dist_km"] + 1.0) ** 2
        mu = float(np.average(nearest["latest_depth"], weights=w))
        sd = float(nearest["latest_depth"].std(ddof=0)) if len(nearest) > 1 else 0.0
        est = {
            "depth_mean": round(mu, 1),
            "depth_low": round(max(0.0, mu - sd), 1),
            "depth_high": round(mu + sd, 1),
        }
    trend = (
        float(nearest["trend_ft_yr"].median())
        if len(nearest) > 0 and nearest["trend_ft_yr"].notna().any()
        else np.nan
    )

    if d_min > 50 or len(active) == 0:
        level = "none"
    elif d_min <= 5 and n10 >= 2:
        level = "strong"
    elif d_min <= 15 and n10 >= 1:
        level = "moderate"
    else:
        level = "weak"

    if level == "none":
        est, trend = None, np.nan
        nearest = nearest.iloc[0:0]

    conf = 0 if level == "none" else int(np.clip(50 * np.exp(-d_min / 15) + 5 * min(n10, 6), 0, 100))

    cautions = [tr["caution_base"]]
    if level == "none":
        cautions.append(tr["caution_none"])
    elif level == "weak":
        cautions.append(tr["caution_weak"].format(d=d_min))
    if not np.isnan(trend) and trend > 1.0:
        cautions.append(tr["caution_trend"].format(t=trend))

    wells_inside_list = (
        st_df[st_df["inside"]][
            ["USGS_ID", "LATITUDE", "LONGITUDE", "latest_depth", "DEPTH_OF_WELL", "trend_ft_yr"]
        ].to_dict(orient="records")
        if "inside" in st_df.columns
        else []
    )

    return {
        "evidence_level": level,
        "confidence": conf,
        "cautions": cautions,
        "depth_estimate": est,
        "trend_ft_yr": trend,
        "field_area_km2": round(polygon_area_km2(vx, vy), 3),
        "wells_inside_list": wells_inside_list,
        "active_within_5km": n5,
        "active_within_10km": n10,
        "nearest_active_km": None if np.isinf(d_min) else round(d_min, 2),
        "nearest_wells": nearest[[
            "USGS_ID", "LATITUDE", "LONGITUDE", "dist_km",
            "latest_depth", "latest_date", "trend_ft_yr",
            "DEPTH_OF_WELL", "water_column"
        ]] if len(nearest) > 0 else pd.DataFrame(),
    }


# =====================================================================
# 2. STREAMLIT APP UI (Multilingual EN / FR / AR + RTL Support)
# =====================================================================
def run_streamlit_app():
    st.set_page_config(page_title="Worldwide Groundwater Advisor", layout="wide")

    # Language Selector at the top
    lang_map = {"English": "en", "Français": "fr", "العربية": "ar"}
    selected_label = st.selectbox(
        "🌐 Language / Langue / اللغة",
        options=list(lang_map.keys()),
        index=0,
    )
    lang = lang_map[selected_label]
    tr = TRANSLATIONS[lang]

    # Inject RTL styling automatically when Arabic is selected
    if lang == "ar":
        st.markdown(
            """
            <style>
            .stApp {
                direction: rtl;
                text-align: right;
            }
            .stRadio > div {
                flex-direction: row-reverse;
                justify-content: flex-end;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.title(tr["title"])
    st.markdown(tr["subtitle"])

    has_well_choice = st.radio(
        tr["has_well_q"],
        options=[tr["no"], tr["yes"]],
        index=0,
        horizontal=True,
    )
    has_well = has_well_choice == tr["yes"]

    profile_map = {
        "💧 توفير المياه (Water Saver)": "water_saver",
        "💰 تعظيم الربح (Profit First)": "profit_first",
        "🌱 حماية التربة (Soil Builder)": "soil_builder",
    }
    selected_profile_label = st.selectbox("🎯 ما هي أولويتك الزراعية الأساسية؟", list(profile_map.keys()))
    selected_profile = profile_map[selected_profile_label]
    # اختيار البذور المتوفرة لدى الفلاح
    seed_options = {
        "🌾 قمح شتوي (Winter Wheat)": 24,
        "🌿 سورغو / ذرة رفيعة (Sorghum)": 4,
        "🫘 صوجا / فول الصويا (Soybeans)": 5,
        "🌱 فصة / برسيم حجازي (Alfalfa)": 36,
        "🌽 ذرة (Corn)": 1,
    }
    selected_seeds_labels = st.multiselect(
        "🌰 (اختياري) ما هي البذور المتوفرة لديك حالياً في المخزن؟",
        options=list(seed_options.keys()),
        placeholder="اختر محصولاً أو أكثر، أو اتركه فارغاً للاقتراح الحر..."
    )
    available_crop_codes = [seed_options[lbl] for lbl in selected_seeds_labels]
    crop_history_map = {
        "🌾 قمح (Wheat)": 24,
        "🌿 سورغو (Sorghum)": 4,
        "🌽 ذرة (Corn)": 1,
        "🫘 صوجا (Soybeans)": 5,
        "🌱 فصة (Alfalfa)": 36,
        "🏜️ أرض بور / راحة (Fallow)": 61,
    }
    c_h1, c_h2, c_h3 = st.columns(3)
    with c_h1:
        c0 = crop_history_map[st.selectbox("🌱 المحصول الحالي (هذا الموسم):", list(crop_history_map.keys()), index=0)]
    with c_h2:
        c1 = crop_history_map[st.selectbox("📅 محصول العام الماضي:", list(crop_history_map.keys()), index=1)]
    with c_h3:
        c2 = crop_history_map[st.selectbox("📆 محصول قبل عامين:", list(crop_history_map.keys()), index=5)]

    forecast_years = 5
    user_well_depth = 0.0

    if has_well:
        st.info(tr["mode_yes"])
        col_y, col_d = st.columns(2)
        with col_y:
            forecast_years = st.slider(tr["horizon_slider"], 1, 20, 5)
        with col_d:
            user_well_depth = st.number_input(tr["well_depth_input"], 0.0, 5000.0, 0.0, 10.0)
    else:
        st.info(tr["mode_no"])

    st.divider()
    st.subheader(tr["draw_header"])
    st.caption(tr["draw_caption"])

    fmap = folium.Map(location=[38.0, -100.5], zoom_start=6, tiles="OpenStreetMap")
    Draw(
        draw_options={
            "polygon": True,
            "rectangle": True,
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    out = st_folium(fmap, height=500, width=None, returned_objects=["all_drawings"])
    drawings = out.get("all_drawings") or []

    button_label = tr["btn_yes"] if has_well else tr["btn_no"]

    if st.button(button_label, type="primary", disabled=not drawings):
        coords = drawings[-1]["geometry"]["coordinates"][0]
        polygon = [(lat, lon) for lon, lat in coords[:-1]]

        poly_arr = np.array(polygon)
        min_lat, max_lat = poly_arr[:, 0].min() - 1.0, poly_arr[:, 0].max() + 1.0
        min_lon, max_lon = poly_arr[:, 1].min() - 1.0, poly_arr[:, 1].max() + 1.0

        with st.spinner(tr["spinner"]):
            m, s = fetch_usgs_data_for_bbox(min_lat, min_lon, max_lat, max_lon)
            if m is None or len(m) == 0 or s is None or len(s) == 0:
                local_meas = "data/groundwater_measurements_clean.csv"
                local_sum = "data/groundwater_well_summary.csv"
                if os.path.exists(local_meas) and os.path.exists(local_sum):
                    m, s = load_local_data(local_meas, local_sum)
                else:
                    m, s = pd.DataFrame(), pd.DataFrame()

            res = analyze_polygon(polygon, m, s, tr)

        st.divider()
        st.subheader(tr["results_header"])

        level_key = f"level_{res['evidence_level']}"
        c1, c2, c3 = st.columns(3)
        c1.metric(tr["metric_evidence"], tr[level_key])
        c2.metric(tr["metric_conf"], f"{res['confidence']}%")
        c3.metric(tr["metric_area"], f"{res['field_area_km2']} km²")

        has_well_inside = len(res["wells_inside_list"]) > 0

        if has_well or has_well_inside:
            if has_well_inside:
                st.success(tr["well_inside_notice"].format(n=len(res["wells_inside_list"])))
                for w_in in res["wells_inside_list"]:
                    st.write(tr["well_inside_item"].format(wid=w_in["USGS_ID"], depth=w_in["latest_depth"]))

            if res["evidence_level"] == "none" or not res["depth_estimate"]:
                st.error(tr["insufficient_forecast"])
            else:
                e = res["depth_estimate"]
                current_depth = e["depth_mean"]
                trend = res["trend_ft_yr"]

                st.markdown(tr["est_depth_header"].format(mean=current_depth, low=e["depth_low"], high=e["depth_high"]))

                if np.isnan(trend):
                    st.warning(tr["trend_unavail"])
                else:
                    hrs = forecast_years if has_well else 5
                    projected_depth = round(current_depth + trend * hrs, 1)
                    st.metric(
                        tr["proj_depth_metric"],
                        f"{projected_depth} ft",
                        delta=f"{trend * hrs:+.1f} ft",
                        delta_color="inverse",
                    )

                    if trend > 0.5:
                        st.warning(tr["declining_warn"].format(trend=trend))
                    else:
                        st.success(tr["stable_ok"].format(trend=trend))

                    if has_well and user_well_depth > 0:
                        rem = round(user_well_depth - projected_depth, 1)
                        if rem <= 0:
                            st.error(tr["dry_alert"].format(wdepth=user_well_depth, pdepth=projected_depth, yrs=hrs))
                        elif rem < 20:
                            st.warning(tr["low_col_warn"].format(rem=rem, yrs=hrs))
                        else:
                            st.info(tr["depth_ok_info"].format(wdepth=user_well_depth, rem=rem, yrs=hrs))
        else:
            if res["evidence_level"] in ("strong", "moderate"):
                st.success(tr["detected_strong_mod"].format(lvl=tr[level_key]))
            elif res["evidence_level"] == "weak":
                st.warning(tr["detected_weak"])
            else:
                st.error(tr["detected_none"])

            if res["depth_estimate"]:
                e = res["depth_estimate"]
                st.write(tr["est_drill_depth"].format(mean=e["depth_mean"], low=e["depth_low"], high=e["depth_high"]))

        st.subheader(tr["notices_header"])
        for c in res["cautions"]:
            st.warning(c)

        if len(res["nearest_wells"]) > 0:
            with st.expander(tr["expander_wells"]):
                st.dataframe(res["nearest_wells"], use_container_width=True)

        # --- زيد الكود هنا (محاذي تماماً مع if len(res...) الفوقانية) ---
        center_lat = float(poly_arr[:, 0].mean())
        center_lon = float(poly_arr[:, 1].mean())

        from model import predict_from_latlon, predict_anywhere_world

        crop_res = predict_from_latlon(
            lat=center_lat, lon=center_lon, has_well=has_well,
            available_crops=available_crop_codes, db="farm.db", profile=selected_profile
        )
        # إذا كان الحقل خارج قاعدة البيانات المحلية، انتقل تلقائياً للربط العالمي Open-Meteo
        if crop_res.get("status") == "out_of_bounds":
            crop_res = predict_anywhere_world(
                lat=center_lat, lon=center_lon, crop_0=c0, crop_1=c1, crop_2=c2,
                has_well=has_well, available_crops=available_crop_codes, profile=selected_profile
            )

        if crop_res["status"] == "ok":
            st.divider()
            st.subheader("🌾 التوصية بالدورة الزراعية والمردود المتوقع")
            st.info(crop_res["seed_match_note"])
            st.success(f"**أفضل دورة زراعية مقترحة لأرضك:** {crop_res['best_recommendation']}")

            rows = []
            for sid, sc in crop_res["scenarios"].items():
                harvest_str = " | ".join(
                    [f"{h['crop']}: {h.get('expected_yield_metric', '0')}" for h in sc["expected_harvest"]]
                )
                rows.append({
                    "السيناريو": sc["name"],
                    "التقييم (Score)": sc["score"],
                    "استهلاك الماء (mm)": sc["criteria"]["water_use_mm"],
                    "المدخول ($/acre)": f"${sc['criteria']['income_usd_ac']:.0f}",
                    "المردود المتوقع (قنطار/هكتار)": harvest_str
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        


if __name__ == "__main__":
    run_streamlit_app()
