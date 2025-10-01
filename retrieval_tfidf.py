import argparse, glob, random, json
import numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy import sparse
import joblib

def output_directories(base_out: Path):
    splits_dir = base_out / "splits"
    retr_dir = base_out / "retrieval"
    models_dir = base_out / "models"
    for d in (splits_dir, retr_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return splits_dir, retr_dir, models_dir

def read_parts(in_dir: Path, which: str) -> List[Path]:
    paths = sorted(glob.glob(str(in_dir / "parts" / which / "part-*.csv")))
    if not paths:
        raise FileNotFoundError(f"No parts found: {in_dir}/parts/{which}/part-*.csv")
    return [Path(p) for p in paths]

def make_leave_last_m(parts_pl: List[Path], parts_pt: List[Path], mask_m: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    pid_to_name: Dict[int, str] = {}
    for p in parts_pl:
        df = pd.read_csv(p, usecols=["pid", "name"])
        pid_to_name.update(df.set_index("pid")["name"].to_dict())

    pid_to_tracks: Dict[int, List[Tuple[int, str]]] = {}
    for p in parts_pt:
        df = pd.read_csv(p, usecols=["pid", "pos", "track_uri"])
        for pid, g in df.groupby("pid"):
            lst = [(int(r.pos), r.track_uri) for r in g.itertuples(index=False)]
            if pid not in pid_to_tracks:
                pid_to_tracks[pid] = lst
            else:
                pid_to_tracks[pid].extend(lst)

    rows = []
    for pid, lst in pid_to_tracks.items():
        if not lst:
            continue
        lst.sort(key=lambda x: x[0])
        uris = [u for _, u in lst]
        if len(uris) <= mask_m:
            continue
        rows.append(
            {"pid": pid, "name": pid_to_name.get(pid, "") or "",
             "seeds_all": uris[:-mask_m], "targets": uris[-mask_m:], "n": len(uris)}
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No eligible playlists (<= mask_m tracks).")
    return df

def sklearn_splits(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, seed=42) -> pd.DataFrame:
    pids = df["pid"].tolist()
    p_train, p_temp = train_test_split(pids, test_size=val_ratio + test_ratio, random_state=seed, shuffle=True)
    rel_test = test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0
    p_val, p_test = train_test_split(p_temp, test_size=rel_test, random_state=seed, shuffle=True)
    pid2split = {pid: "train" for pid in p_train}
    pid2split.update({pid: "val" for pid in p_val})
    pid2split.update({pid: "test" for pid in p_test})
    out = df.copy()
    out["split"] = out["pid"].map(pid2split)
    return out

def expand_challenge_scenarios(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for rec in df.itertuples(index=False):
        seeds_all = list(rec.seeds_all)
        title = rec.name if isinstance(rec.name, str) else ""
        rows.append({"pid": rec.pid, "split": rec.split, "scenario": "title_only",
                     "K": 0, "has_title": 1, "seeds": [], "title": title, "targets": rec.targets})
        if len(seeds_all) >= 5:
            rows.append({"pid": rec.pid, "split": rec.split, "scenario": "title_first_5",
                         "K": 5, "has_title": 1, "seeds": seeds_all[:5], "title": title, "targets": rec.targets})
            rows.append({"pid": rec.pid, "split": rec.split, "scenario": "first_5_only",
                         "K": 5, "has_title": 0, "seeds": seeds_all[:5], "title": "", "targets": rec.targets})
        if len(seeds_all) >= 25:
            rows.append({"pid": rec.pid, "split": rec.split, "scenario": "title_first_25",
                         "K": 25, "has_title": 1, "seeds": seeds_all[:25], "title": title, "targets": rec.targets})
            rows.append({"pid": rec.pid, "split": rec.split, "scenario": "title_rand_25",
                         "K": 25, "has_title": 1, "seeds": rng.sample(seeds_all, 25), "title": title, "targets": rec.targets})
    return pd.DataFrame(rows)

def write_scenarios(scen_df: pd.DataFrame, out_dir: Path) -> Path:
    df = scen_df.copy()
    df["seeds"] = df["seeds"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    df["targets"] = df["targets"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    out = out_dir / "scenarios.csv"
    df.to_csv(out, index=False)
    return out

def load_track_text(in_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(
        in_dir / "features" / "track_meta.csv",
        usecols=["track_uri", "track_name", "artist_name", "album_name"],
    )
    def mk_text(r):
        t = str(r.get("track_name", "") or "")
        a = str(r.get("artist_name", "") or "")
        al = str(r.get("album_name", "") or "")
        return f"{t} {a} {al}".strip()
    meta["text"] = meta.apply(mk_text, axis=1)
    meta = meta[meta["text"].str.strip().astype(bool)].drop_duplicates("track_uri")
    return meta[["track_uri", "text"]]

def top_pop_filter(in_dir: Path, keep_top_frac: float = 0.5) -> pd.Series:
    pop = pd.read_csv(in_dir / "features" / "track_popularity.csv", usecols=["track_uri", "pop_count"])
    thr = pop["pop_count"].quantile(1 - keep_top_frac)
    keep = pop[pop["pop_count"] >= thr]["track_uri"]
    return keep.reset_index(drop=True)

def load_titles(parts_pl: List[Path], limit: int = 100_000) -> List[str]:
    titles: List[str] = []
    for p in parts_pl:
        df = pd.read_csv(p, usecols=["name"])
        titles.extend([str(x) for x in df["name"].dropna().astype(str)])
        if limit and len(titles) >= limit:
            break
    return titles[:limit] if limit else titles

def build_tfidf_index(track_text_df: pd.DataFrame, titles: List[str], max_features=20000, ngram_max=1):
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=3,
        max_df=0.4,
    )
    corpus = track_text_df["text"].tolist() + titles
    vectorizer.fit(corpus)
    X = vectorizer.transform(track_text_df["text"].tolist()).astype(np.float32)
    uri2row = {u: i for i, u in enumerate(track_text_df["track_uri"].tolist())}
    return vectorizer, X, uri2row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--mask_m", type=int, default=5)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--titles_vocab_limit", type=int, default=100_000)
    ap.add_argument("--tfidf_max_features", type=int, default=20000)
    ap.add_argument("--tfidf_ngram_max", type=int, default=1)
    ap.add_argument("--pop_keep_top_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    splits_dir, retr_dir, models_dir = output_directories(Path(args.out_dir))

    parts_pl = read_parts(in_dir, "playlists")
    parts_pt = read_parts(in_dir, "playlist_tracks")

    base_df = make_leave_last_m(parts_pl, parts_pt, mask_m=args.mask_m, seed=args.seed)
    splits_df = sklearn_splits(base_df, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    scen_df = expand_challenge_scenarios(splits_df, seed=args.seed)
    scen_path = write_scenarios(scen_df, splits_dir)
    print(f"[splits] scenarios → {scen_path} (splits=train,val,test)")

    keep_uris = top_pop_filter(in_dir, keep_top_frac=args.pop_keep_top_frac)
    track_text = load_track_text(in_dir).merge(keep_uris.to_frame("track_uri"), on="track_uri", how="inner")
    titles_sample = load_titles(parts_pl, limit=args.titles_vocab_limit)

    vectorizer, X_tracks, uri2row = build_tfidf_index(
        track_text, titles_sample, max_features=args.tfidf_max_features, ngram_max=args.tfidf_ngram_max
    )
    track_uris = track_text["track_uri"].tolist()

    sparse.save_npz(retr_dir / "X_tracks.npz", X_tracks)
    np.save(retr_dir / "track_uris.npy", np.array(track_uris))
    joblib.dump(uri2row, retr_dir / "uri2row.joblib")
    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")

    print(f"[tfidf] TF-IDF={X_tracks.shape} saved to retrieval/, vectorizer saved to models/")

if __name__ == "__main__":
    main()
