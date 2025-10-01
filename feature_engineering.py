import argparse, re
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

SEED = 42
N_WORKERS = max(1, cpu_count())
CHUNK_SIZE = 2000  

FEAT_COLS = [
    "K","has_title","title_len","seed_len",
    "pop_log1p","pop_count",
    "title_overlap_count","title_overlap_ratio",
    "artist_overlap","album_overlap",
]

_TOKEN_RE = re.compile(r"[^a-z0-9'\s]")

def tokenize(s: str):
    if not isinstance(s, str) or not s:
        return []
    s = s.lower()
    s = _TOKEN_RE.sub(" ", s)
    return [t for t in s.split() if len(t) >= 2]

def parse_listcol(s):
    if isinstance(s, str) and s.strip():
        return s.split()
    if isinstance(s, (list, tuple)):
        return [x for x in s if isinstance(x, str)]
    return []

def load_inputs(in_dir: Path, candidates_csv: Path):
    scen = pd.read_csv(
        in_dir / "splits" / "scenarios.csv",
        usecols=["pid","split","K","has_title","title","seeds","targets"],
        dtype={"pid":"int64","split":"category","K":"int16","has_title":"int8"},
        engine="c", memory_map=True
    )
    scen["title"] = scen["title"].fillna("").astype("string")

    cands = pd.read_csv(
        candidates_csv,
        usecols=["pid","split","K","has_title","candidates"],
        dtype={"pid":"int64","split":"category","K":"int16","has_title":"int8"},
        engine="c", memory_map=True
    )

    if not cands["candidates"].map(lambda s: isinstance(s, str)).all():
        raise RuntimeError("candidates.csv: 'candidates' must be a single space-joined string per row.")

    scen = scen.merge(cands, on=["pid","split","K","has_title"], how="inner")

    track_meta = pd.read_csv(
        in_dir / "features" / "track_meta.csv",
        usecols=["track_uri","track_name","artist_id","artist_name","album_id","album_name"],
        dtype={"track_uri":"string","track_name":"string","artist_id":"string","artist_name":"string",
               "album_id":"string","album_name":"string"},
        engine="c", memory_map=True
    )
    track_meta["artist_id"] = track_meta["artist_id"].astype("category")
    track_meta["album_id"]  = track_meta["album_id"].astype("category")

    track_features = pd.read_csv(
        in_dir / "features" / "track_features.csv",
        usecols=["track_uri","pop_count","pop_log1p"],
        dtype={"track_uri":"string","pop_count":"float32","pop_log1p":"float32"},
        engine="c", memory_map=True
    )
    return scen, track_meta, track_features

_TRACK_TOK = None
_TITLE_TOK = None

def _init_worker(track_tok, title_tok):
    global _TRACK_TOK, _TITLE_TOK
    _TRACK_TOK = track_tok
    _TITLE_TOK = title_tok

def _overlap_row(args):
    row_id, cand_uri = args
    tt = _TITLE_TOK.get(row_id, set())
    if not tt:
        return (0, 0.0)
    xt = _TRACK_TOK.get(cand_uri, set())
    inter = tt & xt
    cnt = len(inter)
    return (cnt, cnt / max(1, len(tt)))

def build_examples_split(scen_split: pd.DataFrame,
                         track_meta: pd.DataFrame,
                         track_features: pd.DataFrame) -> pd.DataFrame:
    if scen_split.empty:
        return pd.DataFrame(columns=[
            "pid","split","K","has_title","title_len","seed_len","cand_uri","y",
            "pop_count","pop_log1p",
            "title_overlap_count","title_overlap_ratio",
            "artist_overlap","album_overlap",
        ])

    df = scen_split.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    ex = df[["row_id","pid","split","K","has_title","title","seeds","targets","candidates"]].copy()
    ex["cand_uri"] = ex["candidates"].apply(parse_listcol)
    ex = ex.explode("cand_uri", ignore_index=True)
    ex = ex[ex["cand_uri"].notna()]

    pos = df[["row_id","targets"]].copy()
    pos["cand_uri"] = pos["targets"].apply(parse_listcol)
    pos = pos.explode("cand_uri", ignore_index=True)[["row_id","cand_uri"]]
    pos["y"] = 1.0
    ex = ex.merge(pos, on=["row_id","cand_uri"], how="left")
    ex["y"] = ex["y"].fillna(0.0).astype("float32")

    ex["title_len"] = ex["title"].fillna("").str.split().apply(len).astype("int16")
    seed_len_by_row = df["seeds"].apply(parse_listcol).apply(len).astype("int16")
    ex["seed_len"] = seed_len_by_row.reindex(ex["row_id"]).values

    tf = track_features.rename(columns={"track_uri":"cand_uri"})
    tm = track_meta.rename(columns={"track_uri":"cand_uri"})
    ex = ex.merge(tf, on="cand_uri", how="left")
    ex = ex.merge(tm[["cand_uri","artist_id","album_id","track_name","artist_name","album_name"]], on="cand_uri", how="left")
    ex[["pop_count","pop_log1p"]] = ex[["pop_count","pop_log1p"]].fillna(0).astype({"pop_count":"float32","pop_log1p":"float32"})

    seeds_tbl = df[["row_id","seeds"]].copy()
    seeds_tbl["seed_uri"] = seeds_tbl["seeds"].apply(parse_listcol)
    seeds_tbl = seeds_tbl.explode("seed_uri", ignore_index=True)
    seeds_tbl = seeds_tbl.rename(columns={"seed_uri":"track_uri"}).merge(
        track_meta[["track_uri","artist_id","album_id"]], on="track_uri", how="left"
    )
    art_counts = seeds_tbl.groupby(["row_id","artist_id"]).size().reset_index(name="artist_overlap")
    alb_counts = seeds_tbl.groupby(["row_id","album_id"]).size().reset_index(name="album_overlap")
    ex = ex.merge(art_counts, on=["row_id","artist_id"], how="left")
    ex = ex.merge(alb_counts, on=["row_id","album_id"], how="left")
    ex["artist_overlap"] = ex["artist_overlap"].fillna(0).astype("int16")
    ex["album_overlap"]  = ex["album_overlap"].fillna(0).astype("int16")

    txt = ex[["cand_uri","track_name","artist_name","album_name"]].drop_duplicates("cand_uri").copy()
    txt["text"] = (txt["track_name"].fillna("") + " " +
                   txt["artist_name"].fillna("") + " " +
                   txt["album_name"].fillna("")).str.strip()
    track_tok = {r.cand_uri: set(tokenize(r.text)) for r in txt.itertuples(index=False)}
    title_tok = {r.row_id: set(tokenize(r.title if isinstance(r.title, str) else ""))
                 for r in df[["row_id","title"]].itertuples(index=False)}

    pairs = list(ex[["row_id","cand_uri"]].itertuples(index=False, name=None))
    with Pool(processes=N_WORKERS, initializer=_init_worker, initargs=(track_tok, title_tok)) as pool:
        ov_list = pool.map(_overlap_row, pairs, chunksize=CHUNK_SIZE)

    ov = pd.DataFrame(ov_list, columns=["title_overlap_count","title_overlap_ratio"])
    ex["title_overlap_count"] = ov["title_overlap_count"].astype("int16")
    ex["title_overlap_ratio"] = ov["title_overlap_ratio"].astype("float32")

    keep = [
        "pid","split","K","has_title","title_len","seed_len",
        "cand_uri","y",
        "pop_count","pop_log1p",
        "title_overlap_count","title_overlap_ratio",
        "artist_overlap","album_overlap",
    ]
    return ex[keep].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Feature engineering (parallel, deterministic, tight dtypes).")
    ap.add_argument("--in_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--candidates_csv", type=str, default="artifacts/retrieval/candidates.csv")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = out_dir / "ranking"
    rank_dir.mkdir(parents=True, exist_ok=True)

    scen, track_meta, track_features = load_inputs(in_dir, Path(args.candidates_csv))

    splits = set(scen["split"].astype(str).unique().tolist())
    needed = {"train","val","test"}
    missing = needed - splits
    if missing:
        raise RuntimeError(f"Missing split(s) in scenarios/candidates: {sorted(missing)}")

    for split in ["train","val","test"]:
        part = scen[scen["split"] == split].copy()
        if part.empty:
            raise RuntimeError(f"{split.upper()} split is empty — generate {split} candidates first.")
        tbl = build_examples_split(part, track_meta, track_features)

        tbl["K"] = tbl["K"].astype("int16", errors="ignore")
        tbl["has_title"] = tbl["has_title"].astype("int8", errors="ignore")
        tbl["title_len"] = tbl["title_len"].astype("int16", errors="ignore")
        tbl["seed_len"] = tbl["seed_len"].astype("int16", errors="ignore")
        tbl["pop_count"] = tbl["pop_count"].astype("float32", errors="ignore")
        tbl["pop_log1p"] = tbl["pop_log1p"].astype("float32", errors="ignore")
        tbl["artist_overlap"] = tbl["artist_overlap"].astype("int16", errors="ignore")
        tbl["album_overlap"] = tbl["album_overlap"].astype("int16", errors="ignore")
        tbl["title_overlap_count"] = tbl["title_overlap_count"].astype("int16", errors="ignore")
        tbl["title_overlap_ratio"] = tbl["title_overlap_ratio"].astype("float32", errors="ignore")
        tbl["y"] = tbl["y"].astype("float32", errors="ignore")

        out_path = rank_dir / f"features_{split}.csv"
        tbl.to_csv(out_path, index=False)
        print(f"[features] {split}: rows={len(tbl)} → {out_path}")

if __name__ == "__main__":
    main()
