import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

SEED = 42
SAMPLE_TRAIN_FRAC = 0.5 
NEG_KEEP_FRAC = 0.5     
N_ESTIMATORS = 150
LEARNING_RATE = 0.06
NUM_LEAVES = 63
MAX_DEPTH = 6
FEATURE_FRACTION = 0.9
BAGGING_FRACTION = 0.9
BAGGING_FREQ = 1
L2_REG = 0.0
N_JOBS = -1

FEAT_COLS = [
    "K","has_title","title_len","seed_len",
    "pop_log1p","pop_count",
    "title_overlap_count","title_overlap_ratio",
    "artist_overlap","album_overlap",
]

def out_dir(base: Path) -> Path:
    d = base / "ranking"
    d.mkdir(parents=True, exist_ok=True)
    return d

def parse_listcol(s):
    if isinstance(s, str) and s.strip():
        return s.split()
    if isinstance(s, (list, tuple)):
        return [x for x in s if isinstance(x, str)]
    return []

def downsample_negs(df: pd.DataFrame, keep_frac: float, seed: int) -> pd.DataFrame:
    if df.empty or keep_frac >= 0.999:
        return df
    out = []
    for pid, g in df.groupby("pid", sort=False, group_keys=False):
        pos = g[g["y"] == 1.0]
        neg = g[g["y"] == 0.0]
        if len(neg):
            neg = neg.sample(frac=keep_frac, random_state=seed)
        out.append(pd.concat([pos, neg], ignore_index=True))
    return pd.concat(out, ignore_index=True) if out else df

def rank_topk(df_scores: pd.DataFrame, topk=500) -> pd.DataFrame:
    out = []
    for pid, g in df_scores.groupby("pid"):
        ranked = g.sort_values("score", ascending=False)["cand_uri"].tolist()[:topk]
        row = {"pid": pid}
        for i, u in enumerate(ranked, start=1):
            row[f"trackuri_{i}"] = u
        out.append(row)
    return pd.DataFrame(out).sort_values("pid")

def main():
    ap = argparse.ArgumentParser(description="Train LightGBM ranker on precomputed features.")
    ap.add_argument("--in_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--k_eval", type=int, default=500)
    args = ap.parse_args()

    base = Path(args.in_dir)
    out = out_dir(Path(args.out_dir))

    # load feature tables
    ftrain = base / "ranking" / "features_train.parquet"
    fval   = base / "ranking" / "features_val.parquet"
    ftest  = base / "ranking" / "features_test.parquet"
    for p in [ftrain, fval, ftest]:
        if not p.exists(): raise SystemExit(f"Missing features file: {p}")
    train_tbl = pd.read_parquet(ftrain)
    val_tbl   = pd.read_parquet(fval)
    test_tbl  = pd.read_parquet(ftest)

    if train_tbl.empty or val_tbl.empty or test_tbl.empty:
        raise SystemExit("One of the feature splits is empty.")

    keep_pids = (train_tbl["pid"].drop_duplicates()
                 .sample(frac=SAMPLE_TRAIN_FRAC, random_state=SEED))
    train_tbl = train_tbl[train_tbl["pid"].isin(keep_pids)].reset_index(drop=True)
    train_tbl = downsample_negs(train_tbl, NEG_KEEP_FRAC, SEED)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_tbl[FEAT_COLS].values.astype(np.float32))
    Xva = scaler.transform(val_tbl[FEAT_COLS].values.astype(np.float32))
    Xte = scaler.transform(test_tbl[FEAT_COLS].values.astype(np.float32))
    ytr = train_tbl["y"].astype(np.float32).values

    clf = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        subsample=BAGGING_FRACTION,
        subsample_freq=BAGGING_FREQ,
        colsample_bytree=FEATURE_FRACTION,
        reg_lambda=L2_REG,
        objective="binary",
        random_state=SEED,
        n_jobs=N_JOBS,
        metric="auc",
        deterministic=True,
        force_row_wise=True,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, val_tbl["y"].astype(np.float32).values)], eval_metric="auc", verbose=False)

    def _blend_scores(X, base_tbl):
        pop_idx = FEAT_COLS.index("pop_log1p")
        pop_v = X[:, pop_idx]
        mn, mx = float(pop_v.min()), float(pop_v.max())
        pop_norm = (pop_v - mn) / (mx - mn + 1e-8)
        base_tbl["ranker_score"] = clf.predict_proba(X)[:,1].astype(np.float32)
        base_tbl["score"] = 0.9*base_tbl["ranker_score"].values + 0.1*pop_norm
        return base_tbl

    val_tbl  = _blend_scores(Xva, val_tbl)
    test_tbl = _blend_scores(Xte, test_tbl)

    joblib.dump(clf, out / "model.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(clf, out / "model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")

    sub = rank_topk(test_tbl[["pid","cand_uri","score"]], topk=500)
    sub_path = out / "submission_preview.csv"
    sub.to_csv(sub_path, index=False)

    print(f"[saved] model → {out/'model.joblib'}")
    print(f"[saved] scaler → {out/'scaler.joblib'}")
    print(f"[saved] submission preview → {sub_path}")

if __name__ == "__main__":
    main()
