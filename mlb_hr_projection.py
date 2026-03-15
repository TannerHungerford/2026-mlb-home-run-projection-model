import warnings
warnings.filterwarnings("ignore")

import re
import time
import numpy as np
import pandas as pd

from pybaseball import (
    batting_stats,
    statcast_batter,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_batter_percentile_ranks,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge


# ============================================================
# CONFIG
# ============================================================

OUTPUT_FILE = "projected_2026_hr_dataset.csv"
BACKTEST_FILE = "hr_backtest_results.csv"

HISTORICAL_SEASONS = list(range(2019, 2026))   # 2019-2025
MIN_PA = 120

LEAGUE_HR_PER_PA_DEFAULT = 0.0315
MAX_REASONABLE_HR_PER_PA = 0.095
PARK_FACTOR_STRENGTH = 0.30
REGRESSION_PA = 1200.0

RECENT_WEIGHTS = {
    1: 5.0,
    2: 4.0,
    3: 3.0,
}


# ============================================================
# HELPERS
# ============================================================

def pick_first_existing(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def normalize_team(team):
    if pd.isna(team):
        return np.nan
    team = str(team).strip().upper()
    mapping = {
        "CHW": "CWS",
        "KC": "KCR",
        "WSH": "WSN",
        "SD": "SDP",
        "SF": "SFG",
        "TB": "TBR",
        "AZ": "ARI",
    }
    return mapping.get(team, team)


def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def coerce_pct(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].dropna().median()
            if pd.notna(med) and med > 1:
                df[c] = df[c] / 100.0
    return df


def safe_series(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def safe_fill(df, preferred, fallback, default=np.nan):
    a = safe_series(df, preferred, default)
    b = safe_series(df, fallback, default)
    return a.fillna(b)


def clip_series(s, lo, hi):
    return np.minimum(np.maximum(s, lo), hi)


def weighted_mean(vals, wts):
    vals = np.array(vals, dtype=float)
    wts = np.array(wts, dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return np.nan
    vals = vals[mask]
    wts = wts[mask]
    wts = wts / wts.sum()
    return float(np.sum(vals * wts))


def safe_nan_to_num(x, default=0.0):
    if pd.isna(x):
        return default
    return float(x)


def aging_multiplier_hr(age):
    if pd.isna(age):
        return 1.00
    if age <= 23:
        return 1.040
    if age <= 25:
        return 1.022
    if age <= 27:
        return 1.012
    if age <= 29:
        return 1.000
    if age <= 31:
        return 0.988
    if age <= 33:
        return 0.960
    if age <= 35:
        return 0.920
    return 0.875


def aging_multiplier_pa(age):
    if pd.isna(age):
        return 1.00
    if age <= 24:
        return 1.03
    if age <= 27:
        return 1.01
    if age <= 30:
        return 1.00
    if age <= 32:
        return 0.98
    if age <= 34:
        return 0.94
    if age <= 36:
        return 0.89
    return 0.83


def soft_cap_hr_rate(rate):
    if pd.isna(rate):
        return LEAGUE_HR_PER_PA_DEFAULT
    if rate <= 0.072:
        return rate
    if rate <= 0.082:
        return 0.072 + 0.82 * (rate - 0.072)
    if rate <= 0.095:
        return 0.0802 + 0.60 * (rate - 0.082)
    return 0.088 + 0.35 * (rate - 0.095)


# ============================================================
# LOAD FANRAPHS HISTORY
# ============================================================

def load_fangraphs_history():
    dfs = []

    for year in HISTORICAL_SEASONS:
        loaded = False

        for attempt in range(3):
            try:
                y = batting_stats(year, year, qual=0)
                y["Season"] = year
                dfs.append(y)
                print(f"Loaded FanGraphs batting stats for {year}")
                loaded = True
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {year}: {e}")
                time.sleep(3)

        if not loaded:
            print(f"Skipping {year} after repeated failures.")

    if not dfs:
        raise RuntimeError("Could not load any FanGraphs batting stats.")

    fg = pd.concat(dfs, ignore_index=True)

    colmap = {
        "Name": pick_first_existing(fg, ["Name", "PLAYER", "Player"]),
        "Season": pick_first_existing(fg, ["Season", "season"]),
        "Age": pick_first_existing(fg, ["Age", "AGE", "age"]),
        "Team": pick_first_existing(fg, ["Team", "Tm", "TEAM", "team_name"]),
        "PA": pick_first_existing(fg, ["PA", "pa"]),
        "HR": pick_first_existing(fg, ["HR", "hr"]),
        "Pull%": pick_first_existing(fg, ["Pull%", "Pull %", "pull_pct"]),
        "EV": pick_first_existing(fg, ["EV", "AvgEV", "avgEV", "ev"]),
        "HardHit%": pick_first_existing(fg, ["HardHit%", "HardHit %", "hard_hit_pct"]),
        "HR/FB": pick_first_existing(fg, ["HR/FB", "HR/FB%", "hr_fb"]),
        "FB%": pick_first_existing(fg, ["FB%", "FB %", "fb_pct"]),
        "Barrel%": pick_first_existing(fg, ["Barrel%", "Barrel %", "barrel_pct"]),
        "Barrels": pick_first_existing(fg, ["Barrels", "barrels"]),
        "IDfg": pick_first_existing(fg, ["IDfg", "playerid", "PlayerId"]),
    }

    rename_dict = {v: k for k, v in colmap.items() if v is not None}
    fg = fg.rename(columns=rename_dict).copy()

    required = ["Name", "Season", "Age", "Team", "PA", "HR"]
    missing = [c for c in required if c not in fg.columns]
    if missing:
        raise ValueError(f"Missing required batting_stats columns: {missing}")

    if "IDfg" not in fg.columns:
        fg["IDfg"] = fg["Name"]

    fg = coerce_numeric(fg, ["Season", "Age", "PA", "HR", "EV", "Barrels"])
    fg = coerce_pct(fg, ["Pull%", "HardHit%", "HR/FB", "FB%", "Barrel%"])
    fg["Team"] = fg["Team"].apply(normalize_team)

    fg = fg[fg["PA"].fillna(0) >= MIN_PA].copy()
    fg["HR_per_PA"] = fg["HR"] / fg["PA"]

    if "Barrels" in fg.columns:
        fg["Barrels_per_PA"] = fg["Barrels"] / fg["PA"]
    elif "Barrel%" in fg.columns:
        fg["Barrels_per_PA"] = fg["Barrel%"]
    else:
        fg["Barrels_per_PA"] = np.nan

    return fg


# ============================================================
# LOAD STATCAST TABLES
# ============================================================

def load_exitvelo_barrels():
    dfs = []
    for year in HISTORICAL_SEASONS:
        try:
            d = statcast_batter_exitvelo_barrels(year, minBBE=1)
            d["Season"] = year
            dfs.append(d)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=["Name", "Season"])

    sc = pd.concat(dfs, ignore_index=True)

    rename_map = {
        "last_name, first_name": "Name",
        "player_name": "Name",
        "player_id": "MLBAMID",
        "barrels": "sc_Barrels",
        "barrel_batted_rate": "sc_Barrel%",
        "launch_angle_avg": "LaunchAngle",
        "exit_velocity_avg": "sc_EV",
        "avg_hit_speed": "sc_EV",
        "max_ev": "MaxEV",
        "hard_hit_percent": "sc_HardHit%",
        "hard_hit_rate": "sc_HardHit%",
        "sweet_spot_percent": "SweetSpot%",
    }

    for old, new in rename_map.items():
        if old in sc.columns:
            sc = sc.rename(columns={old: new})

    keep = [c for c in [
        "Name", "Season", "MLBAMID", "sc_Barrels", "sc_Barrel%",
        "LaunchAngle", "sc_EV", "MaxEV", "sc_HardHit%", "SweetSpot%"
    ] if c in sc.columns]

    sc = sc[keep].copy()
    sc = coerce_numeric(sc, ["sc_Barrels", "LaunchAngle", "sc_EV", "MaxEV", "MLBAMID"])
    sc = coerce_pct(sc, ["sc_Barrel%", "sc_HardHit%", "SweetSpot%"])
    return sc


def load_expected_stats():
    dfs = []
    for year in HISTORICAL_SEASONS:
        try:
            d = statcast_batter_expected_stats(year, minPA=1)
            d["Season"] = year
            dfs.append(d)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=["Name", "Season"])

    sc = pd.concat(dfs, ignore_index=True)

    rename_map = {
        "last_name, first_name": "Name",
        "player_name": "Name",
        "xba": "xBA",
        "xslg": "xSLG",
        "xwoba": "xwOBA",
        "woba": "wOBA",
    }

    for old, new in rename_map.items():
        if old in sc.columns:
            sc = sc.rename(columns={old: new})

    keep = [c for c in ["Name", "Season", "xBA", "xSLG", "xwOBA", "wOBA"] if c in sc.columns]
    sc = sc[keep].copy()
    sc = coerce_numeric(sc, ["xBA", "xSLG", "xwOBA", "wOBA"])
    return sc


def load_percentile_ranks():
    dfs = []
    for year in HISTORICAL_SEASONS:
        try:
            d = statcast_batter_percentile_ranks(year)
            d["Season"] = year
            dfs.append(d)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame(columns=["Name", "Season"])

    sc = pd.concat(dfs, ignore_index=True)

    rename_map = {
        "player_name": "Name",
        "last_name, first_name": "Name",
        "bat_speed": "BatSpeed",
        "blast_rate": "BlastRate",
        "squared_up_percent": "SquaredUp%",
        "swing_length": "SwingLength",
    }

    for old, new in rename_map.items():
        if old in sc.columns:
            sc = sc.rename(columns={old: new})

    keep = [c for c in [
        "Name", "Season", "BatSpeed", "BlastRate", "SquaredUp%", "SwingLength"
    ] if c in sc.columns]

    sc = sc[keep].copy()
    sc = coerce_numeric(sc, ["BatSpeed", "SwingLength"])
    sc = coerce_pct(sc, ["BlastRate", "SquaredUp%"])
    return sc


# ============================================================
# PARK FACTORS
# ============================================================

def _extract_hr_column(table):
    cols = {str(c).strip().lower(): c for c in table.columns}
    for key in cols:
        if re.fullmatch(r"hr", key) or "home run" in key:
            return cols[key]
    for key in cols:
        if "hr" in key:
            return cols[key]
    return None


def _extract_team_column(table):
    cols = {str(c).strip().lower(): c for c in table.columns}
    for key in cols:
        if key in {"team", "club", "tm"}:
            return cols[key]
    for key in cols:
        if "team" in key or "club" in key:
            return cols[key]
    return None


def load_park_factors():
    candidate_urls = [
        "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors",
        "https://baseballsavant.mlb.com/leaderboard/statcast-venue?rolling=1",
    ]

    parsed = None

    for url in candidate_urls:
        try:
            tables = pd.read_html(url)
            for t in tables:
                team_col = _extract_team_column(t)
                hr_col = _extract_hr_column(t)
                if team_col is not None and hr_col is not None:
                    tmp = t[[team_col, hr_col]].copy()
                    tmp.columns = ["Team", "HR_Park_Factor"]
                    tmp["Team"] = tmp["Team"].astype(str).str.extract(r"([A-Z]{2,3})")[0]
                    tmp["Team"] = tmp["Team"].apply(normalize_team)
                    tmp["HR_Park_Factor"] = pd.to_numeric(tmp["HR_Park_Factor"], errors="coerce")
                    tmp = tmp.dropna(subset=["Team", "HR_Park_Factor"])
                    tmp = tmp[(tmp["HR_Park_Factor"] > 50) & (tmp["HR_Park_Factor"] < 150)]
                    if len(tmp) >= 20:
                        parsed = tmp.drop_duplicates("Team").copy()
                        break
            if parsed is not None:
                break
        except Exception:
            pass

    if parsed is None:
        teams = [
            "ARI","ATL","BAL","BOS","CHC","CIN","CLE","COL","CWS","DET",
            "HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK",
            "PHI","PIT","SDP","SEA","SFG","STL","TBR","TEX","TOR","WSN"
        ]
        parsed = pd.DataFrame({
            "Team": teams,
            "HR_Park_Factor": [100.0] * len(teams)
        })

    parsed["HR_Park_Index"] = parsed["HR_Park_Factor"] / 100.0
    return parsed


# ============================================================
# RECENT FORM FEATURES
# ============================================================

def infer_mlbam_mapping(master_df):
    mapping = {}

    if "IDfg" not in master_df.columns or "MLBAMID" not in master_df.columns:
        return mapping

    tmp = master_df[["IDfg", "MLBAMID"]].dropna().copy()
    if tmp.empty:
        return mapping

    tmp["MLBAMID"] = pd.to_numeric(tmp["MLBAMID"], errors="coerce")
    tmp = tmp.dropna()

    for pid, grp in tmp.groupby("IDfg"):
        vals = grp["MLBAMID"].mode()
        if len(vals):
            mapping[pid] = int(vals.iloc[0])

    return mapping


def compute_recent_statcast_form(player_id, year):
    default = {
        "Recent_HardHit_Trend": np.nan,
        "Recent_HR_per_BBE_Trend": np.nan,
        "Recent_AvgEV_Trend": np.nan,
        "Recent_HR_Count": np.nan,
        "Recent_BBE": np.nan,
    }

    if pd.isna(player_id):
        return default

    try:
        late = statcast_batter(f"{year}-08-01", f"{year}-11-30", int(player_id))
        early = statcast_batter(f"{year}-03-01", f"{year}-07-31", int(player_id))
    except Exception:
        return default

    def summarize(df):
        if df is None or df.empty:
            return {
                "hardhit": np.nan,
                "hr_per_bbe": np.nan,
                "avg_ev": np.nan,
                "hr_count": np.nan,
                "bbe": np.nan,
            }

        out = df.copy()

        if "launch_speed" in out.columns:
            out["launch_speed"] = pd.to_numeric(out["launch_speed"], errors="coerce")
        else:
            out["launch_speed"] = np.nan

        if "events" not in out.columns:
            out["events"] = np.nan

        bbe = out[out["launch_speed"].notna()].copy()
        if bbe.empty:
            return {
                "hardhit": np.nan,
                "hr_per_bbe": np.nan,
                "avg_ev": np.nan,
                "hr_count": np.nan,
                "bbe": 0.0,
            }

        bbe["is_hardhit"] = (bbe["launch_speed"] >= 95).astype(float)
        bbe["is_hr"] = (bbe["events"] == "home_run").astype(float)

        return {
            "hardhit": float(bbe["is_hardhit"].mean()),
            "hr_per_bbe": float(bbe["is_hr"].mean()),
            "avg_ev": float(bbe["launch_speed"].mean()),
            "hr_count": float(bbe["is_hr"].sum()),
            "bbe": float(len(bbe)),
        }

    s_late = summarize(late)
    s_early = summarize(early)

    return {
        "Recent_HardHit_Trend": s_late["hardhit"] - s_early["hardhit"]
            if pd.notna(s_late["hardhit"]) and pd.notna(s_early["hardhit"]) else np.nan,
        "Recent_HR_per_BBE_Trend": s_late["hr_per_bbe"] - s_early["hr_per_bbe"]
            if pd.notna(s_late["hr_per_bbe"]) and pd.notna(s_early["hr_per_bbe"]) else np.nan,
        "Recent_AvgEV_Trend": s_late["avg_ev"] - s_early["avg_ev"]
            if pd.notna(s_late["avg_ev"]) and pd.notna(s_early["avg_ev"]) else np.nan,
        "Recent_HR_Count": s_late["hr_count"],
        "Recent_BBE": s_late["bbe"],
    }


def add_recent_form_features(proj_df, idfg_to_mlbam, source_year=2025):
    feats = []
    for _, row in proj_df.iterrows():
        pid = row["IDfg"]
        mlbam = idfg_to_mlbam.get(pid, np.nan)
        feats.append(compute_recent_statcast_form(mlbam, source_year))

    recent_df = pd.DataFrame(feats, index=proj_df.index)
    proj_df = pd.concat([proj_df, recent_df], axis=1)

    proj_df["Recent_Power_Trend"] = (
        0.35 * proj_df["Barrel_blend_trend"].fillna(0.0) +
        0.20 * proj_df["HardHit_blend_trend"].fillna(0.0) +
        0.15 * proj_df["xSLG_trend"].fillna(0.0) +
        0.10 * proj_df["HR_per_PA_trend"].fillna(0.0) +
        0.08 * proj_df["Recent_HardHit_Trend"].fillna(0.0) +
        0.07 * proj_df["Recent_HR_per_BBE_Trend"].fillna(0.0) +
        0.05 * (proj_df["Recent_AvgEV_Trend"].fillna(0.0) / 10.0)
    )

    return proj_df


# ============================================================
# MASTER TABLE
# ============================================================

def build_master_table():
    fg = load_fangraphs_history()
    evb = load_exitvelo_barrels()
    exp = load_expected_stats()
    pct = load_percentile_ranks()
    pf = load_park_factors()

    df = fg.merge(evb, on=["Name", "Season"], how="left")
    df = df.merge(exp, on=["Name", "Season"], how="left")
    df = df.merge(pct, on=["Name", "Season"], how="left")
    df = df.merge(pf, on="Team", how="left")

    df["HR_Park_Factor"] = safe_series(df, "HR_Park_Factor", 100.0).fillna(100.0)
    df["HR_Park_Index"] = safe_series(df, "HR_Park_Index", 1.0).fillna(1.0)

    df["EV_blend"] = safe_fill(df, "sc_EV", "EV", 88.0).fillna(88.0)
    df["HardHit_blend"] = safe_fill(df, "sc_HardHit%", "HardHit%", 0.35).fillna(0.35)
    df["Barrel_blend"] = safe_fill(df, "sc_Barrel%", "Barrel%", 0.08).fillna(0.08)
    df["MaxEV"] = safe_fill(df, "MaxEV", "EV_blend", 88.0).fillna(88.0)
    df["LaunchAngle"] = safe_series(df, "LaunchAngle", 12.0).fillna(12.0)
    df["SweetSpot%"] = safe_series(df, "SweetSpot%", 0.33).fillna(0.33)
    df["xSLG"] = safe_series(df, "xSLG", 0.400).fillna(0.400)
    df["xwOBA"] = safe_series(df, "xwOBA", 0.320).fillna(0.320)
    df["BatSpeed"] = safe_series(df, "BatSpeed", 72.0).fillna(72.0)
    df["BlastRate"] = safe_series(df, "BlastRate", 0.08).fillna(0.08)
    df["SquaredUp%"] = safe_series(df, "SquaredUp%", 0.25).fillna(0.25)
    df["SwingLength"] = safe_series(df, "SwingLength", 7.0).fillna(7.0)
    df["FB%"] = safe_series(df, "FB%", 0.35).fillna(0.35)
    df["HR/FB"] = safe_series(df, "HR/FB", 0.12).fillna(0.12)

    df["xHR_proxy"] = (
        0.29 * df["Barrel_blend"] +
        0.17 * df["HardHit_blend"] +
        0.12 * df["FB%"] +
        0.12 * df["HR/FB"] +
        0.14 * df["xSLG"] +
        0.08 * ((df["MaxEV"] - 85.0) / 20.0) +
        0.05 * ((df["LaunchAngle"] - 10.0) / 10.0) +
        0.03 * df["SweetSpot%"]
    )

    return df


# ============================================================
# FEATURES / BASELINES
# ============================================================

FEATURE_BASE = [
    "HR_per_PA", "Barrels_per_PA", "Pull%", "EV_blend", "HardHit_blend",
    "HR/FB", "FB%", "Barrel_blend", "LaunchAngle", "xSLG", "xwOBA",
    "MaxEV", "SweetSpot%", "BatSpeed", "BlastRate", "SquaredUp%",
    "SwingLength", "xHR_proxy", "HR_Park_Index", "PA"
]


def compute_league_hr_rate(df):
    if df.empty:
        return LEAGUE_HR_PER_PA_DEFAULT
    total_hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0).sum()
    total_pa = pd.to_numeric(df["PA"], errors="coerce").fillna(0).sum()
    if total_pa <= 0:
        return LEAGUE_HR_PER_PA_DEFAULT
    return float(total_hr / total_pa)


def marcel_hr_rate(player_hist, league_hr_rate):
    hist = player_hist.sort_values("Season").copy()
    if hist.empty:
        return league_hr_rate

    seasons = sorted(hist["Season"].unique(), reverse=True)
    vals = []
    pas = []
    weights = []

    rank = 1
    for yr in seasons[:3]:
        row = hist[hist["Season"] == yr].iloc[-1]
        vals.append(row["HR_per_PA"])
        pas.append(row["PA"])
        weights.append(RECENT_WEIGHTS.get(rank, 3.0))
        rank += 1

    vals = np.array(vals, dtype=float)
    pas = np.array(pas, dtype=float)
    weights = np.array(weights, dtype=float)

    weighted_pa = np.sum(pas * weights)
    weighted_hr = np.sum(vals * pas * weights)

    regressed_rate = (weighted_hr + league_hr_rate * REGRESSION_PA) / (weighted_pa + REGRESSION_PA)
    return float(regressed_rate)


def marcel_pa(player_hist):
    hist = player_hist.sort_values("Season").copy()
    if hist.empty:
        return 500.0

    vals = []
    wts = []
    seasons = sorted(hist["Season"].unique(), reverse=True)

    rank = 1
    for yr in seasons[:3]:
        row = hist[hist["Season"] == yr].iloc[-1]
        vals.append(row["PA"])
        wts.append(RECENT_WEIGHTS.get(rank, 3.0))
        rank += 1

    pa = weighted_mean(vals, wts)
    if pd.isna(pa):
        pa = hist["PA"].mean()

    next_age = hist["Age"].iloc[-1] + 1 if hist["Age"].notna().any() else np.nan
    pa *= aging_multiplier_pa(next_age)

    return float(np.clip(pa, 150, 710))


def injury_risk_multiplier(player_hist):
    pa = player_hist.sort_values("Season")["PA"].tail(3).astype(float).values

    if len(pa) == 0:
        return 0.92

    avg_pa = np.nanmean(pa)
    sd_pa = np.nanstd(pa) if len(pa) >= 2 else 0.0

    if avg_pa >= 625 and sd_pa <= 60:
        return 1.00
    elif avg_pa >= 560:
        return 0.98
    elif avg_pa >= 500:
        return 0.95
    elif avg_pa >= 400:
        return 0.91
    else:
        return 0.85


def add_summary_features(player_hist, row, feature_list):
    hist = player_hist.sort_values("Season").copy()

    for f in feature_list:
        if f in hist.columns:
            vals = pd.to_numeric(hist[f], errors="coerce")
            row[f + "_last"] = vals.iloc[-1] if len(vals) else np.nan
            row[f + "_mean"] = vals.mean() if vals.notna().any() else np.nan

            recent = vals.tail(3).dropna()
            if len(recent) >= 2:
                row[f + "_trend"] = recent.iloc[-1] - recent.iloc[0]
            else:
                row[f + "_trend"] = 0.0
        else:
            row[f + "_last"] = np.nan
            row[f + "_mean"] = np.nan
            row[f + "_trend"] = np.nan

    return row


# ============================================================
# TRAIN / PROJECTION ROWS
# ============================================================

def build_training_rows(df, target_year):
    rows = []

    hist_all = df[df["Season"] < target_year].copy()
    tgt = df[df["Season"] == target_year].copy()

    if hist_all.empty or tgt.empty:
        return pd.DataFrame()

    league_hr_rate = compute_league_hr_rate(hist_all)

    for pid in tgt["IDfg"].dropna().unique():
        player_hist = hist_all[hist_all["IDfg"] == pid].sort_values("Season").tail(3).copy()
        player_tgt = tgt[tgt["IDfg"] == pid].copy()

        if player_hist.empty or player_tgt.empty:
            continue

        if (len(player_hist) < 2) and (player_hist["PA"].sum() < 300):
            continue

        tgt_row = player_tgt.iloc[0]

        row = {
            "Name": tgt_row["Name"],
            "IDfg": pid,
            "target_season": target_year,
            "Age": tgt_row["Age"],
            "HR_target": tgt_row["HR"],
            "PA_target": tgt_row["PA"],
            "HR_rate_target": tgt_row["HR_per_PA"],
            "marcel_hr_rate": marcel_hr_rate(player_hist, league_hr_rate),
            "marcel_pa": marcel_pa(player_hist),
            "Injury_Risk_Mult": injury_risk_multiplier(player_hist),
            "PA_last": player_hist["PA"].iloc[-1],
            "PA_3yr_avg": player_hist["PA"].mean(),
        }

        row = add_summary_features(player_hist, row, FEATURE_BASE)

        row["Recent_Power_Trend"] = (
            0.45 * safe_nan_to_num(row.get("Barrel_blend_trend", 0.0)) +
            0.25 * safe_nan_to_num(row.get("HardHit_blend_trend", 0.0)) +
            0.20 * safe_nan_to_num(row.get("xSLG_trend", 0.0)) +
            0.10 * safe_nan_to_num(row.get("HR_per_PA_trend", 0.0))
        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_projection_rows(df, target_year=2026):
    hist_years = [target_year - 3, target_year - 2, target_year - 1]
    hist = df[df["Season"].isin(hist_years)].copy()
    latest = hist.sort_values(["IDfg", "Season"]).groupby("IDfg").tail(1)

    league_hr_rate = compute_league_hr_rate(hist)

    rows = []
    for _, p in latest.iterrows():
        pid = p["IDfg"]
        player_hist = hist[hist["IDfg"] == pid].sort_values("Season").tail(3).copy()
        if player_hist.empty:
            continue

        if (len(player_hist) < 2) and (player_hist["PA"].sum() < 300):
            continue

        age_target = p["Age"] + 1 if pd.notna(p["Age"]) else np.nan

        row = {
            "Name": p["Name"],
            "IDfg": pid,
            "Team": normalize_team(p["Team"]),
            "Age": age_target,
            "Projected_PA_Baseline": marcel_pa(player_hist),
            "marcel_pa": marcel_pa(player_hist),
            "marcel_hr_rate": marcel_hr_rate(player_hist, league_hr_rate),
            "Injury_Risk_Mult": injury_risk_multiplier(player_hist),
            "HR_Park_Factor": p.get("HR_Park_Factor", 100.0),
            "HR_Park_Index": p.get("HR_Park_Index", 1.0),
            "PA_last": player_hist["PA"].iloc[-1],
            "PA_3yr_avg": player_hist["PA"].mean(),
        }

        row = add_summary_features(player_hist, row, FEATURE_BASE)

        row["Recent_Power_Trend"] = (
            0.45 * safe_nan_to_num(row.get("Barrel_blend_trend", 0.0)) +
            0.25 * safe_nan_to_num(row.get("HardHit_blend_trend", 0.0)) +
            0.20 * safe_nan_to_num(row.get("xSLG_trend", 0.0)) +
            0.10 * safe_nan_to_num(row.get("HR_per_PA_trend", 0.0))
        )

        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns(train_df, target="rate"):
    exclude = {
        "Name", "IDfg", "target_season", "HR_target", "PA_target", "HR_rate_target"
    }
    cols = [c for c in train_df.columns if c not in exclude]

    if target == "rate":
        return cols
    return [c for c in cols if c != "marcel_hr_rate"]


# ============================================================
# MODELS
# ============================================================

def fit_models(train_df):
    if train_df.empty:
        raise RuntimeError("Training data is empty.")

    rate_feature_cols = get_feature_columns(train_df, target="rate")
    pa_feature_cols = get_feature_columns(train_df, target="pa")

    pre_rate = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), rate_feature_cols)
    ])

    pre_pa = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), pa_feature_cols)
    ])

    X_rate = train_df[rate_feature_cols].copy()
    y_rate = train_df["HR_rate_target"].copy()

    X_pa = train_df[pa_feature_cols].copy()
    y_pa = train_df["PA_target"].copy()

    rate_models = {
        "rf": Pipeline([
            ("prep", pre_rate),
            ("model", RandomForestRegressor(
                n_estimators=900,
                max_depth=9,
                min_samples_leaf=3,
                random_state=42
            ))
        ]),
        "et": Pipeline([
            ("prep", pre_rate),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                max_depth=9,
                min_samples_leaf=2,
                random_state=42
            ))
        ]),
        "gb": Pipeline([
            ("prep", pre_rate),
            ("model", GradientBoostingRegressor(
                n_estimators=450,
                learning_rate=0.035,
                max_depth=3,
                random_state=42
            ))
        ]),
        "ridge": Pipeline([
            ("prep", pre_rate),
            ("model", Ridge(alpha=1.8))
        ]),
    }

    pa_models = {
        "rf": Pipeline([
            ("prep", pre_pa),
            ("model", RandomForestRegressor(
                n_estimators=550,
                max_depth=7,
                min_samples_leaf=4,
                random_state=11
            ))
        ]),
        "ridge": Pipeline([
            ("prep", pre_pa),
            ("model", Ridge(alpha=4.0))
        ]),
    }

    for model in rate_models.values():
        model.fit(X_rate, y_rate)

    for model in pa_models.values():
        model.fit(X_pa, y_pa)

    return {
        "rate_models": rate_models,
        "pa_models": pa_models,
        "rate_feature_cols": rate_feature_cols,
        "pa_feature_cols": pa_feature_cols,
    }


# ============================================================
# PROJECTION
# ============================================================

def project_year(models, proj_df, target_year=2026):
    X_rate = proj_df[models["rate_feature_cols"]].copy()
    X_pa = proj_df[models["pa_feature_cols"]].copy()

    rate_preds = {
        name: model.predict(X_rate)
        for name, model in models["rate_models"].items()
    }

    pa_preds = {
        name: model.predict(X_pa)
        for name, model in models["pa_models"].items()
    }

    ml_rate = (
        0.32 * pd.Series(rate_preds["rf"], index=proj_df.index) +
        0.24 * pd.Series(rate_preds["et"], index=proj_df.index) +
        0.20 * pd.Series(rate_preds["gb"], index=proj_df.index) +
        0.24 * pd.Series(rate_preds["ridge"], index=proj_df.index)
    )

    marcel_rate = proj_df["marcel_hr_rate"].fillna(LEAGUE_HR_PER_PA_DEFAULT)

    # less conservative than before
    blended_rate = 0.48 * marcel_rate + 0.52 * ml_rate

    barrel_boost = (
        proj_df["Barrel_blend_last"].fillna(proj_df["Barrel_blend_mean"]).fillna(0.08) - 0.08
    ) * 0.30

    ev_boost = (
        proj_df["EV_blend_last"].fillna(proj_df["EV_blend_mean"]).fillna(88.0) - 88.0
    ) * 0.0025

    xslg_boost = (
        proj_df["xSLG_last"].fillna(proj_df["xSLG_mean"]).fillna(0.400) - 0.400
    ) * 0.14

    trend_boost = 0.18 * proj_df["Recent_Power_Trend"].fillna(0.0)

    recent_form_boost = (
        0.07 * proj_df.get("Recent_HardHit_Trend", pd.Series(0.0, index=proj_df.index)).fillna(0.0) +
        0.12 * proj_df.get("Recent_HR_per_BBE_Trend", pd.Series(0.0, index=proj_df.index)).fillna(0.0) +
        0.0025 * proj_df.get("Recent_AvgEV_Trend", pd.Series(0.0, index=proj_df.index)).fillna(0.0)
    )

    blended_rate = blended_rate + barrel_boost + ev_boost + xslg_boost + trend_boost + recent_form_boost

    proj_df["age_multiplier"] = proj_df["Age"].apply(aging_multiplier_hr)
    blended_rate = blended_rate * proj_df["age_multiplier"]

    park_adj = 1.0 + (proj_df["HR_Park_Index"].fillna(1.0) - 1.0) * PARK_FACTOR_STRENGTH
    blended_rate = blended_rate * park_adj

    blended_rate = blended_rate * (0.990 + 0.010 * proj_df["Injury_Risk_Mult"].fillna(0.92))

    blended_rate = blended_rate.apply(soft_cap_hr_rate)
    blended_rate = clip_series(blended_rate, 0.008, MAX_REASONABLE_HR_PER_PA)

    ml_pa = (
        0.55 * pd.Series(pa_preds["rf"], index=proj_df.index) +
        0.45 * pd.Series(pa_preds["ridge"], index=proj_df.index)
    )

    baseline_pa = proj_df["Projected_PA_Baseline"].fillna(500.0)
    projected_pa = 0.63 * baseline_pa + 0.37 * ml_pa

    # much lighter PA durability penalty so elite guys can still reach 650-700 PA
    projected_pa = projected_pa * (0.985 + 0.015 * proj_df["Injury_Risk_Mult"].fillna(0.92))
    projected_pa = clip_series(projected_pa, 170, 710)

    proj_df["HR_rate_final"] = blended_rate
    proj_df["Projected_PA"] = projected_pa
    proj_df["Projected_HR"] = proj_df["HR_rate_final"] * proj_df["Projected_PA"]
    proj_df["Downside_HR"] = proj_df["Projected_HR"] * 0.90
    proj_df["Upside_HR"] = proj_df["Projected_HR"] * 1.10

    output = proj_df[[
        "Name",
        "Team",
        "Age",
        "Projected_PA",
        "HR_Park_Factor",
        "marcel_hr_rate",
        "Injury_Risk_Mult",
        "Recent_Power_Trend",
        "age_multiplier",
        "HR_rate_final",
        "Projected_HR",
        "Downside_HR",
        "Upside_HR"
    ]].copy()

    output = output.sort_values("Projected_HR", ascending=False).reset_index(drop=True)

    output["Projected_PA"] = output["Projected_PA"].round(1)
    output["HR_Park_Factor"] = output["HR_Park_Factor"].round(1)
    output["marcel_hr_rate"] = output["marcel_hr_rate"].round(4)
    output["Injury_Risk_Mult"] = output["Injury_Risk_Mult"].round(3)
    output["Recent_Power_Trend"] = output["Recent_Power_Trend"].round(4)
    output["age_multiplier"] = output["age_multiplier"].round(3)
    output["HR_rate_final"] = output["HR_rate_final"].round(4)
    output["Projected_HR"] = output["Projected_HR"].round(1)
    output["Downside_HR"] = output["Downside_HR"].round(1)
    output["Upside_HR"] = output["Upside_HR"].round(1)

    return output


# ============================================================
# BACKTEST
# ============================================================

def evaluate_leaderboard_accuracy(proj_output, actual_df, actual_year, top_n=20):
    actual = actual_df[actual_df["Season"] == actual_year][["Name", "HR"]].copy()
    actual = actual.rename(columns={"HR": "Actual_HR"})

    merged = proj_output.merge(actual, on="Name", how="inner").copy()
    if merged.empty:
        return {
            "year": actual_year,
            "MAE": np.nan,
            "RMSE": np.nan,
            f"Top{top_n}_Overlap": np.nan,
        }

    merged["Abs_Error"] = (merged["Projected_HR"] - merged["Actual_HR"]).abs()
    merged["Sq_Error"] = (merged["Projected_HR"] - merged["Actual_HR"]) ** 2

    leaderboard_proj = merged.sort_values("Projected_HR", ascending=False).head(top_n)
    leaderboard_actual = merged.sort_values("Actual_HR", ascending=False).head(top_n)

    mae = merged["Abs_Error"].mean()
    rmse = np.sqrt(merged["Sq_Error"].mean())

    proj_names = set(leaderboard_proj["Name"])
    actual_names = set(leaderboard_actual["Name"])
    overlap = len(proj_names & actual_names)

    return {
        "year": actual_year,
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        f"Top{top_n}_Overlap": int(overlap),
    }


def run_backtest(df):
    results = []

    idfg_to_mlbam = infer_mlbam_mapping(df)

    for yr in [2024, 2025]:
        train_parts = []
        for target in range(2021, yr):
            part = build_training_rows(df, target)
            if not part.empty:
                train_parts.append(part)

        if not train_parts:
            continue

        train_df = pd.concat(train_parts, ignore_index=True)
        models = fit_models(train_df)

        proj_df = build_projection_rows(df, target_year=yr)
        if proj_df.empty:
            continue

        proj_df = add_recent_form_features(proj_df, idfg_to_mlbam, source_year=yr - 1)
        proj_output = project_year(models, proj_df, target_year=yr)

        metrics = evaluate_leaderboard_accuracy(proj_output, df, actual_year=yr, top_n=20)
        results.append(metrics)

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame(columns=["year", "MAE", "RMSE", "Top20_Overlap"])


# ============================================================
# MAIN
# ============================================================

def main():
    df = build_master_table()
    if df.empty:
        raise RuntimeError("Master table is empty.")

    train_parts = []
    for target in range(2021, 2026):
        part = build_training_rows(df, target)
        if not part.empty:
            train_parts.append(part)

    if not train_parts:
        raise RuntimeError("Training data came back empty.")

    train_df = pd.concat(train_parts, ignore_index=True)
    models = fit_models(train_df)

    proj_df = build_projection_rows(df, target_year=2026)
    if proj_df.empty:
        raise RuntimeError("Projection data came back empty.")

    idfg_to_mlbam = infer_mlbam_mapping(df)
    proj_df = add_recent_form_features(proj_df, idfg_to_mlbam, source_year=2025)

    output = project_year(models, proj_df, target_year=2026)
    output.to_csv(OUTPUT_FILE, index=False)

    backtest = run_backtest(df)
    if not backtest.empty:
        backtest.to_csv(BACKTEST_FILE, index=False)
        print(backtest)

    print(f"Saved projections to {OUTPUT_FILE}")
    if not backtest.empty:
        print(f"Saved backtest results to {BACKTEST_FILE}")


if __name__ == "__main__":
    main()
