import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd

from pybaseball import (
    batting_stats,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_batter_percentile_ranks,
)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge


OUTPUT_FILE = "projected_2026_hr_dataset.csv"
SEASONS = [2022, 2023, 2024, 2025]
MIN_PA = 150


# ============================================================
# BASIC HELPERS
# ============================================================

def pick_first_existing(df, candidates, default=None):
    for col in candidates:
        if col in df.columns:
            return col
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


# ============================================================
# FANRAPHS / PYBASEBALL HISTORY
# ============================================================

def load_fangraphs_history():
    fg = batting_stats(2022, 2025, qual=0)

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
# STATCAST ENRICHMENT
# ============================================================

def load_exitvelo_barrels():
    dfs = []
    for year in SEASONS:
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
    sc = coerce_numeric(sc, ["sc_Barrels", "LaunchAngle", "sc_EV", "MaxEV"])
    sc = coerce_pct(sc, ["sc_Barrel%", "sc_HardHit%", "SweetSpot%"])
    return sc


def load_expected_stats():
    dfs = []
    for year in SEASONS:
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
    for year in SEASONS:
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
# AUTOMATIC PARK FACTORS
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
        "https://baseballsavant.mlb.com/leaderboard/statcast-venue?rolling=1"
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
# MERGE ALL DATA
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
        0.30 * df["Barrel_blend"] +
        0.18 * df["HardHit_blend"] +
        0.12 * df["FB%"] +
        0.12 * df["HR/FB"] +
        0.12 * df["xSLG"] +
        0.08 * ((df["MaxEV"] - 85.0) / 20.0) +
        0.04 * ((df["LaunchAngle"] - 10.0) / 10.0) +
        0.04 * df["SweetSpot%"]
    )

    return df


# ============================================================
# MODEL HELPERS
# ============================================================

FEATURE_BASE = [
    "HR_per_PA", "Barrels_per_PA", "Pull%", "EV_blend", "HardHit_blend",
    "HR/FB", "FB%", "Barrel_blend", "LaunchAngle", "xSLG", "xwOBA",
    "MaxEV", "SweetSpot%", "BatSpeed", "BlastRate", "SquaredUp%",
    "SwingLength", "xHR_proxy", "HR_Park_Index"
]


def aging_multiplier(age):
    if pd.isna(age):
        return 1.00
    if age <= 23:
        return 1.07
    if age <= 25:
        return 1.05
    if age <= 27:
        return 1.02
    if age <= 29:
        return 1.00
    if age <= 31:
        return 0.97
    if age <= 33:
        return 0.93
    if age <= 35:
        return 0.88
    return 0.82


def projected_pa_from_history(player_hist):
    pa_2025 = player_hist.loc[player_hist["Season"] == 2025, "PA"]
    pa_2024 = player_hist.loc[player_hist["Season"] == 2024, "PA"]
    pa_2023 = player_hist.loc[player_hist["Season"] == 2023, "PA"]

    pa_2025 = pa_2025.iloc[0] if len(pa_2025) else np.nan
    pa_2024 = pa_2024.iloc[0] if len(pa_2024) else np.nan
    pa_2023 = pa_2023.iloc[0] if len(pa_2023) else np.nan

    vals = np.array([pa_2025, pa_2024, pa_2023], dtype=float)
    weights = np.array([0.60, 0.30, 0.10], dtype=float)

    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return 500.0

    vals = vals[mask]
    weights = weights[mask]
    weights = weights / weights.sum()

    pa = float(np.sum(vals * weights))
    pa_avg = player_hist["PA"].mean()

    if pd.notna(pa_avg):
        pa = 0.80 * pa + 0.20 * pa_avg

    next_age = player_hist["Age"].max() + 1 if player_hist["Age"].notna().any() else np.nan
    if pd.notna(next_age):
        if next_age >= 34:
            pa *= 0.94
        elif next_age <= 25:
            pa *= 1.03

    return float(np.clip(pa, 150, 725))


def add_summary_features(player_hist, row, feature_list):
    for f in feature_list:
        if f in player_hist.columns:
            vals = pd.to_numeric(player_hist[f], errors="coerce")
            vals = vals.dropna()
            if len(vals) > 0:
                row[f + "_mean"] = vals.mean()
                row[f + "_last"] = vals.iloc[-1]
                row[f + "_trend"] = vals.iloc[-1] - vals.iloc[0] if len(vals) >= 2 else 0.0
            else:
                row[f + "_mean"] = np.nan
                row[f + "_last"] = np.nan
                row[f + "_trend"] = np.nan
        else:
            row[f + "_mean"] = np.nan
            row[f + "_last"] = np.nan
            row[f + "_trend"] = np.nan
    return row


def build_training_rows(df, target_year):
    hist = df[df["Season"] < target_year].copy()
    tgt = df[df["Season"] == target_year].copy()
    rows = []

    for pid in tgt["IDfg"].dropna().unique():
        player_hist = hist[hist["IDfg"] == pid].sort_values("Season").tail(3).copy()
        player_tgt = tgt[tgt["IDfg"] == pid].copy()

        if player_hist.empty or player_tgt.empty:
            continue

        row = {
            "Name": player_tgt["Name"].iloc[0],
            "IDfg": pid,
            "Age": player_tgt["Age"].iloc[0],
            "PA_target": player_tgt["PA"].iloc[0],
            "HR_target": player_tgt["HR"].iloc[0],
            "HR_rate_target": player_tgt["HR_per_PA"].iloc[0],
            "target_season": target_year,
            "PA_last": player_hist["PA"].iloc[-1],
            "PA_3yr_avg": player_hist["PA"].mean(),
        }

        row = add_summary_features(player_hist, row, FEATURE_BASE)
        rows.append(row)

    return pd.DataFrame(rows)


def build_projection_rows(df):
    hist = df[df["Season"].isin([2023, 2024, 2025])].copy()
    latest = hist.sort_values(["IDfg", "Season"]).groupby("IDfg").tail(1)
    rows = []

    for _, p in latest.iterrows():
        pid = p["IDfg"]
        player_hist = hist[hist["IDfg"] == pid].sort_values("Season").tail(3).copy()
        if player_hist.empty:
            continue

        team_2026 = normalize_team(p["Team"])
        age_2026 = p["Age"] + 1 if pd.notna(p["Age"]) else np.nan

        row = {
            "Name": p["Name"],
            "IDfg": pid,
            "Team": team_2026,
            "Age": age_2026,
            "Projected_PA": projected_pa_from_history(player_hist),
            "HR_Park_Factor": p.get("HR_Park_Factor", 100.0),
            "HR_Park_Index": p.get("HR_Park_Index", 1.0),
            "PA_last": player_hist["PA"].iloc[-1],
            "PA_3yr_avg": player_hist["PA"].mean(),
        }

        row = add_summary_features(player_hist, row, FEATURE_BASE)
        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns(train_df):
    exclude = {
        "Name", "IDfg", "PA_target", "HR_target", "HR_rate_target", "target_season"
    }
    return [c for c in train_df.columns if c not in exclude]


# ============================================================
# FIT MODELS
# ============================================================

def fit_models(train_df):
    feature_cols = get_feature_columns(train_df)

    train_mask = train_df["target_season"].isin([2023, 2024])
    X_train = train_df.loc[train_mask, feature_cols].copy()
    y_train = train_df.loc[train_mask, "HR_rate_target"].copy()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), feature_cols)
    ])

    main_models = {
        "rf": RandomForestRegressor(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42
        ),
        "gb": GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            random_state=42
        ),
        "et": ExtraTreesRegressor(
            n_estimators=700,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42
        ),
        "ridge": Ridge(alpha=1.25)
    }

    fitted = {}
    for name, model in main_models.items():
        pipe = Pipeline([
            ("prep", pre),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

    xhr_feature_cols = [c for c in feature_cols if any(tag in c for tag in [
        "Barrel", "EV", "HardHit", "LaunchAngle", "xSLG", "xwOBA",
        "MaxEV", "SweetSpot", "BatSpeed", "BlastRate", "SquaredUp",
        "SwingLength", "xHR_proxy", "FB%", "HR/FB"
    ])]

    pre_xhr = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), xhr_feature_cols)
    ])

    xhr_model = Pipeline([
        ("prep", pre_xhr),
        ("model", GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.035,
            max_depth=3,
            random_state=7
        ))
    ])
    xhr_model.fit(train_df.loc[train_mask, xhr_feature_cols], y_train)

    fitted["xhr_model"] = xhr_model
    fitted["xhr_feature_cols"] = xhr_feature_cols
    return fitted, feature_cols


# ============================================================
# PROJECT 2026
# ============================================================

def project_2026(models, feature_cols, proj_df):
    X = proj_df[feature_cols].copy()

    pred_rf = models["rf"].predict(X)
    pred_gb = models["gb"].predict(X)
    pred_et = models["et"].predict(X)
    pred_ridge = models["ridge"].predict(X)

    main_rate = (
        0.32 * pred_rf +
        0.28 * pred_gb +
        0.25 * pred_et +
        0.15 * pred_ridge
    )

    xhr_cols = models["xhr_feature_cols"]
    xhr_rate = models["xhr_model"].predict(proj_df[xhr_cols].copy())

    main_rate_series = pd.Series(main_rate, index=proj_df.index)

    proxy_rate = proj_df["xHR_proxy_last"].fillna(
        proj_df["xHR_proxy_mean"]
    ).fillna(main_rate_series)

    raw_rate = (
        0.62 * main_rate +
        0.23 * xhr_rate +
        0.15 * proxy_rate
    )

    raw_rate = np.clip(raw_rate, 0.005, 0.12)

    proj_df["HR_rate_raw"] = raw_rate
    proj_df["age_multiplier"] = proj_df["Age"].apply(aging_multiplier)
    proj_df["HR_rate_age_adj"] = proj_df["HR_rate_raw"] * proj_df["age_multiplier"]
    proj_df["HR_rate_final"] = proj_df["HR_rate_age_adj"] * proj_df["HR_Park_Index"]

    proj_df["Projected_HR"] = proj_df["HR_rate_final"] * proj_df["Projected_PA"]
    proj_df["Projected_HR"] = np.clip(proj_df["Projected_HR"], 0, None)

    output = proj_df[[
        "Name",
        "Team",
        "Age",
        "Projected_PA",
        "HR_Park_Factor",
        "HR_rate_raw",
        "age_multiplier",
        "HR_rate_final",
        "Projected_HR"
    ]].copy()

    output = output.sort_values("Projected_HR", ascending=False).reset_index(drop=True)

    output["Projected_PA"] = output["Projected_PA"].round(1)
    output["HR_Park_Factor"] = output["HR_Park_Factor"].round(1)
    output["HR_rate_raw"] = output["HR_rate_raw"].round(4)
    output["age_multiplier"] = output["age_multiplier"].round(3)
    output["HR_rate_final"] = output["HR_rate_final"].round(4)
    output["Projected_HR"] = output["Projected_HR"].round(1)

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    df = build_master_table()

    train23 = build_training_rows(df, 2023)
    train24 = build_training_rows(df, 2024)
    train25 = build_training_rows(df, 2025)
    train_df = pd.concat([train23, train24, train25], ignore_index=True)

    if train_df.empty:
        raise RuntimeError("Training data came back empty.")

    models, feature_cols = fit_models(train_df)
    proj_df = build_projection_rows(df)

    if proj_df.empty:
        raise RuntimeError("Projection data came back empty.")

    output = project_2026(models, feature_cols, proj_df)
    output.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()