import argparse, json, gzip, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from annoy import AnnoyIndex

def output_directories(base_out: Path):
    subs = base_out / "submissions"
    subs.mkdir(parents=True, exist_ok=True)
    return subs

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def parse_listcol(s):
    if isinstance(s, str) and s.strip():
        return s.split()
    return []

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    import re
    s = text.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return [t for t in s.split() if len(t) >= 2]

def write_submission_gz(path: Path, team_name: str, team_email: str, pid_to_uris: dict[int, list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path.as_posix(), "wt", encoding="utf-8") as f:
        f.write(f"team_info, {team_name}, {team_email}\n")
        for pid in sorted(pid_to_uris.keys()):
            uris = pid_to_uris[pid]
            f.write(",".join([str(pid)] + uris) + "\n")

def load_challenge_json(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    rows = []
    for p in files:
        obj = json.loads(p.read_text())
        pls = obj["playlists"] if "playlists" in obj else obj
        for pl in pls:
            pid = int(pl["pid"])
            title = pl.get("name", "") or ""
            seeds = [t["track_uri"] for t in pl.get("tracks", [])]
            rows.append({"pid": pid, "title": title, "seed_uris": seeds})
    return pd.DataFrame(rows).sort_values("pid")

def load_challenge_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seed_uris" in df and df["seed_uris"].dtype == object:
        df["seed_uris"] = df["seed_uris"].apply(parse_listcol)
    elif "seeds" in df:
        df["seed_uris"] = df["seeds"].apply(parse_listcol)
    else:
        df["seed_uris"] = [[] for _ in range(len(df))]
    if "title" not in df:
        df["title"] = ""
    return df[["pid","title","seed_uris"]].sort_values("pid")

def require_file(p: Path, hint: str):
    if not p.exists():
        raise SystemExit(f"Missing required artifact: {p}\n→ {hint}")
    return p

def load_retrieval_artifacts(models_dir: Path):
    vec_path   = require_file(models_dir / "tfidf.joblib",
                              "Saved TfidfVectorizer from retrieval.py")
    svd_path   = require_file(models_dir / "svd.joblib",
                              "Saved TruncatedSVD from retrieval.py")
    ann_path   = require_file(models_dir / "ann_index.ann",
                              "Saved Annoy index from retrieval.py")
    items_path = require_file(models_dir / "ann_items.csv",
                              "Mapping between Annoy item index and track_uri (written by retrieval.py)")

    vectorizer = joblib.load(vec_path)
    svd = joblib.load(svd_path)
    dim = int(getattr(svd, "n_components", None) or getattr(svd, "n_components_", None) or 128)

    ann = AnnoyIndex(dim, metric="angular")
    if not ann.load(ann_path.as_posix()):
        raise SystemExit(f"Failed to load Annoy index: {ann_path}")

    items_df = pd.read_csv(items_path) 
    if "idx" not in items_df.columns or "track_uri" not in items_df.columns:
        raise SystemExit(f"{items_path} must have columns: idx,track_uri")
    idx_to_uri = dict(zip(items_df["idx"].astype(int), items_df["track_uri"].astype(str)))
    uri_to_idx = {v:k for k,v in idx_to_uri.items()}

    return vectorizer, svd, dim, ann, idx_to_uri, uri_to_idx


def load_ranker_artifacts(rank_dir: Path):
    scaler_path = require_file(rank_dir / "scaler.joblib",
                               "Saved StandardScaler from ranker.py")
    model_path  = require_file(rank_dir / "model.joblib",
                               "Saved scikit-learn ranker from ranker.py")
    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)
    return scaler, model

def norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def title_embedding(title: str, vectorizer, svd) -> np.ndarray | None:
    if not isinstance(title, str) or not title.strip():
        return None
    t = vectorizer.transform([title])
    z = svd.transform(t).astype(np.float32)[0]
    return norm(z)

def seed_mean_vector(seeds: list[str], ann: AnnoyIndex, uri_to_idx: dict[str,int], svd_dim: int) -> np.ndarray | None:
    vecs = []
    for u in seeds:
        i = uri_to_idx.get(u)
        if i is not None:
            v = np.array(ann.get_item_vector(i), dtype=np.float32)
            if v.size == svd_dim:
                vecs.append(v)
    if not vecs:
        return None
    return norm(np.mean(vecs, axis=0))

def build_query_vector(seeds: list[str], title: str, ann, uri_to_idx, svd_dim: int, vectorizer, svd, title_weight=0.25):
    v_seed  = seed_mean_vector(seeds, ann, uri_to_idx, svd_dim)
    v_title = title_embedding(title, vectorizer, svd)
    if v_seed is None and v_title is None:
        return None
    if v_seed is None:
        return v_title
    if v_title is None:
        return v_seed
    return norm(v_seed + title_weight * v_title)

def ann_retrieve(ann: AnnoyIndex, qvec: np.ndarray, topn: int) -> list[int]:
    idxs = ann.get_nns_by_vector(qvec.tolist(), topn, search_k=-1, include_distances=False)
    return idxs

FEAT_COLS = [
    "K","has_title","title_len","seed_len",
    "pop_log1p","pop_count",
    "title_overlap_count","title_overlap_ratio",
    "artist_overlap","album_overlap",
]

def build_examples(chall_df: pd.DataFrame,
                   pid_to_cands: dict[int, list[str]],
                   track_meta: pd.DataFrame,
                   track_features: pd.DataFrame):
    df = chall_df.copy().reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    rows = []
    for r in df.itertuples(index=False):
        cands = pid_to_cands.get(int(r.pid), [])
        if not cands:
            continue
        rows.append(pd.DataFrame({
            "row_id": r.row_id,
            "pid": int(r.pid),
            "K": len(r.seed_uris),
            "has_title": 1 if isinstance(r.title, str) and r.title.strip() else 0,
            "title": r.title if isinstance(r.title, str) else "",
            "seed_uris": " ".join(r.seed_uris),
            "cand_uri": cands
        }))
    ex = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["row_id","pid","K","has_title","title","seed_uris","cand_uri"])

    ex["title_len"] = ex["title"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0).astype(np.int16)
    ex["seed_len"]  = ex["seed_uris"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0).astype(np.int16)

    tf = track_features[["track_uri","pop_count","pop_log1p"]].rename(columns={"track_uri":"cand_uri"})
    tm = track_meta[["track_uri","artist_id","album_id","track_name","artist_name","album_name"]].rename(columns={"track_uri":"cand_uri"})
    ex = ex.merge(tf, on="cand_uri", how="left")
    ex = ex.merge(tm, on="cand_uri", how="left")
    ex[["pop_count","pop_log1p"]] = ex[["pop_count","pop_log1p"]].fillna(0)

    seeds = df[["row_id","seed_uris"]].copy()
    seeds["seed_uri"] = seeds["seed_uris"].str.split()
    seeds = seeds.explode("seed_uri", ignore_index=True)
    seeds = seeds.rename(columns={"seed_uri":"track_uri"}).merge(
        track_meta[["track_uri","artist_id","album_id"]], on="track_uri", how="left"
    )
    art_counts = seeds.groupby(["row_id","artist_id"]).size().reset_index(name="art_overlap")
    alb_counts = seeds.groupby(["row_id","album_id"]).size().reset_index(name="alb_overlap")
    ex = ex.merge(art_counts, on=["row_id","artist_id"], how="left")
    ex = ex.merge(alb_counts, on=["row_id","album_id"], how="left")
    ex["artist_overlap"] = ex["art_overlap"].fillna(0).astype(np.int16)
    ex["album_overlap"]  = ex["alb_overlap"].fillna(0).astype(np.int16)
    ex = ex.drop(columns=["art_overlap","alb_overlap"])

    txt = (ex[["cand_uri","track_name","artist_name","album_name"]].drop_duplicates("cand_uri").copy())
    txt["text"] = (txt["track_name"].fillna("") + " " + txt["artist_name"].fillna("") + " " + txt["album_name"].fillna("")).str.strip()
    track_tok = {r.cand_uri: set(tokenize(r.text)) for r in txt.itertuples(index=False)}
    title_tok = {r.row_id: set(tokenize(r.title)) for r in df[["row_id","title"]].itertuples(index=False)}

    def _overlap(row):
        tt = title_tok.get(row.row_id, set())
        if not tt: return (0, 0.0)
        xt = track_tok.get(row.cand_uri, set())
        inter = tt & xt
        cnt = len(inter)
        return (cnt, cnt / max(1, len(tt)))

    ov = ex[["row_id","cand_uri"]].apply(lambda r: _overlap(r), axis=1, result_type="expand")
    ex["title_overlap_count"] = ov[0].astype(np.int16)
    ex["title_overlap_ratio"] = ov[1].astype(np.float32)

    return ex.reset_index(drop=True)

def enforce_caps(ranked_uris: list[str], artist_of: dict[str,str], cap: int):
    if cap <= 0: 
        return ranked_uris
    out, counts = [], Counter()
    for u in ranked_uris:
        a = artist_of.get(u)
        if a is None or counts[a] < cap:
            out.append(u)
            if a is not None:
                counts[a] += 1
    i = 0
    while len(out) < len(ranked_uris) and i < len(ranked_uris):
        u = ranked_uris[i]; i += 1
        if u not in out:
            out.append(u)
    return out[:len(ranked_uris)]

def main():
    ap = argparse.ArgumentParser(description="Final inference (load-only): Annoy retrieval + sklearn ranker.")
    ap.add_argument("--in_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--challenge_json", type=str, default="")
    ap.add_argument("--challenge_csv", type=str, default="")
    ap.add_argument("--team_name", type=str, required=True)
    ap.add_argument("--team_email", type=str, required=True)
    ap.add_argument("--n_retrieved", type=int, default=1200, help="candidates per playlist (>=500)")
    ap.add_argument("--artist_cap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    subs_dir = output_directories(Path(args.out_dir))
    in_dir = Path(args.in_dir)

    if args.challenge_json:
        chall = load_challenge_json(Path(args.challenge_json))
    elif args.challenge_csv:
        chall = load_challenge_csv(Path(args.challenge_csv))
    else:
        raise SystemExit("Provide --challenge_json or --challenge_csv")

    track_meta = pd.read_csv(in_dir / "features" / "track_meta.csv")
    track_features = pd.read_csv(in_dir / "features" / "track_features.csv")
    pop_df = pd.read_csv(in_dir / "features" / "track_popularity_recency.csv")
    pop_list = pop_df.sort_values(["pop_count","pop_log1p"], ascending=False)["track_uri"].tolist()
    artist_of = track_meta.set_index("track_uri")["artist_id"].astype(str).to_dict()

    vectorizer, svd, svd_dim, ann, idx_to_uri, uri_to_idx = load_retrieval_artifacts(in_dir / "models")

    scaler, model = load_ranker_artifacts(in_dir / "ranking")

    pid_to_cands: dict[int, list[str]] = {}
    for r in chall.itertuples(index=False):
        seeds = [u for u in r.seed_uris if isinstance(u, str)]
        title = r.title if isinstance(r.title, str) else ""
        seed_set = set(seeds)

        q = build_query_vector(seeds, title, ann, uri_to_idx, svd_dim, vectorizer, svd, title_weight=0.25)
        if q is None:
            pid_to_cands[int(r.pid)] = pop_list[:args.n_retrieved]
            continue

        idxs = ann_retrieve(ann, q, topn=max(args.n_retrieved, 1200))
        uris = [idx_to_uri[i] for i in idxs if i in idx_to_uri]
        uris = [u for u in uris if u not in seed_set]
        pid_to_cands[int(r.pid)] = uris[:args.n_retrieved]

    ex = build_examples(chall, pid_to_cands, track_meta, track_features)
    X = scaler.transform(ex[FEAT_COLS].values.astype(np.float32))
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:,1].astype(np.float32)
    else:
        scores = model.predict(X).astype(np.float32)
    ex["score"] = scores

    pid_to_final = {}
    for pid, g in ex.groupby("pid"):
        ranked = g.sort_values("score", ascending=False)["cand_uri"].tolist()
        ranked = enforce_caps(ranked, artist_of, cap=args.artist_cap)
        uniq, seen = [], set()
        seed_set = set(next(iter(chall[chall["pid"] == pid]["seed_uris"]), []))
        for u in ranked:
            if u in seed_set or u in seen:
                continue
            seen.add(u); uniq.append(u)
            if len(uniq) >= 500:
                break
        if len(uniq) < 500:
            for u in pop_list:
                if u not in seen and u not in seed_set:
                    uniq.append(u); seen.add(u)
                    if len(uniq) >= 500:
                        break
        assert len(uniq) == 500, f"pid {pid}: got {len(uniq)}"
        pid_to_final[int(pid)] = uniq

    out_path = subs_dir / "submission.csv.gz"
    write_submission_gz(out_path, args.team_name, args.team_email, pid_to_final)

    cfg = {
        "seed": args.seed,
        "n_retrieved": args.n_retrieved,
        "artist_cap": args.artist_cap,
        "retrieval_artifacts": {
            "tfidf": str((in_dir / "models" / "tfidf.joblib").resolve()),
            "svd": str((in_dir / "models" / "svd.joblib").resolve()),
            "ann_index": str((in_dir / "models" / "ann_index.ann").resolve()),
            "ann_items": str((in_dir / "models" / "ann_items.csv").resolve()),
            "svd_dim": svd_dim,
        },
        "ranker_artifacts": {
            "scaler": str((in_dir / "ranking" / "scaler.pkl").resolve()),
            "model": str((in_dir / "ranking" / "model.pkl").resolve()),
        },
    }
    (out_path.with_suffix(".config.json")).write_text(json.dumps(cfg, indent=2))
    print(f"[done] submission → {out_path}")

if __name__ == "__main__":
    main()
