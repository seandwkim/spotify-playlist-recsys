import argparse, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

def top_global_pop(in_dir: Path, whitelist: pd.Series, n: int = 600) -> List[str]:
    pop = pd.read_csv(in_dir / "features" / "track_popularity.csv", usecols=["track_uri", "pop_count", "pop_log1p"])
    pop = pop.merge(whitelist.to_frame("track_uri"), on="track_uri", how="inner")
    pop = pop.sort_values(["pop_count", "pop_log1p"], ascending=False)
    return pop["track_uri"].head(n).tolist()

def top_pop_filter(in_dir: Path, keep_top_frac: float = 0.5) -> pd.Series:
    pop = pd.read_csv(in_dir / "features" / "track_popularity.csv", usecols=["track_uri", "pop_count"])
    thr = pop["pop_count"].quantile(1 - keep_top_frac)
    keep = pop[pop["pop_count"] >= thr]["track_uri"]
    return keep.reset_index(drop=True)

def ann_retrieve(ann: AnnoyIndex, qvec: np.ndarray, track_uris: List[str], topn: int, search_k: int = -1):
    if qvec is None:
        return []
    idxs, dists = ann.get_nns_by_vector(
        qvec.astype(np.float32).tolist(),
        n=topn,
        search_k=search_k,
        include_distances=True
    )
    return [(track_uris[i], -float(d)) for i, d in zip(idxs, dists)]

_TITLE_CACHE: Dict[str, np.ndarray] = {}

def _get_title_vec(title: str, vectorizer, svd, norm) -> np.ndarray | None:
    if not isinstance(title, str) or not title.strip():
        return None
    v = _TITLE_CACHE.get(title)
    if v is not None:
        return v
    t_sparse = vectorizer.transform([title])
    t_red = svd.transform(t_sparse)
    t_red = norm.transform(t_red)[0]
    _TITLE_CACHE[title] = t_red
    return t_red

def query_vector(vectorizer, svd, norm, seed_uris: List[str], title: str,
                 track_vecs: np.ndarray, uri2row: Dict[str, int], title_weight: float = 0.25):
    q = None
    seed_vecs = []
    for u in seed_uris:
        i = uri2row.get(u)
        if i is not None:
            seed_vecs.append(track_vecs[i])
    if seed_vecs:
        q = np.mean(seed_vecs, axis=0)

    t_red = _get_title_vec(title, vectorizer, svd, norm)
    if t_red is not None:
        q = t_red if q is None else (q + title_weight * t_red)

    if q is None:
        return None
    n = np.linalg.norm(q)
    return q if n == 0 else (q / n)

def _write_rows(rows: List[Dict], out_path: Path, wrote_header_flag: List[bool]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = False if (out_path.exists() or wrote_header_flag[0]) else True
    df.to_csv(out_path, mode="a", header=header, index=False)
    wrote_header_flag[0] = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--ann_trees", type=int, default=50)
    ap.add_argument("--ann_search_k", type=int, default=-1)
    ap.add_argument("--title_weight", type=float, default=0.25)
    ap.add_argument("--n_retrieved", type=int, default=800)
    ap.add_argument("--pop_keep_top_frac", type=float, default=0.5)
    ap.add_argument("--scenario_splits", type=str, default="train,val,test",
                    help="Comma-separated; default runs all (train,val,test)")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4)),
                    help="Parallel query workers; set 1 to disable parallelism")
    ap.add_argument("--flush_every", type=int, default=2000,
                    help="Append to CSV every N rows to keep memory low")
    ap.add_argument("--reset_output", action="store_true",
                    help="If set, delete existing retrieval/candidates.csv before writing")
    args = ap.parse_args()

    base_in = Path(args.in_dir)
    base_out = Path(args.out_dir)
    splits_dir = base_out / "splits"
    retr_dir = base_out / "retrieval"
    models_dir = base_out / "models"
    out_path = retr_dir / "candidates.csv"

    if args.reset_output and out_path.exists():
        out_path.unlink()

    scen = pd.read_csv(splits_dir / "scenarios.csv")
    scen["seeds"] = scen["seeds"].apply(lambda s: s.split() if isinstance(s, str) else [])
    scen["targets"] = scen["targets"].apply(lambda s: s.split() if isinstance(s, str) else [])

    want_splits = {s.strip() for s in (args.scenario_splits or "").split(",") if s.strip()}
    if want_splits:
        scen = scen[scen["split"].isin(want_splits)].reset_index(drop=True)

    track_uris = np.load(retr_dir / "track_uris.npy", allow_pickle=True).tolist()
    uri2row = joblib.load(retr_dir / "uri2row.joblib")
    track_vecs = np.load(retr_dir / "track_vecs.npy", mmap_mode="r")
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    svd = joblib.load(models_dir / "svd.joblib")
    norm = joblib.load(models_dir / "norm.joblib")

    f = track_vecs.shape[1]
    ann = AnnoyIndex(f, metric="angular")
    for i, vec in enumerate(track_vecs):
        ann.add_item(i, vec.astype(np.float32).tolist())
    ann.build(args.ann_trees)
    ann.save(str(retr_dir / "annoy.ann"))
    print(f"[annoy] built trees={args.ann_trees} on {len(track_uris)} tracks → retrieval/annoy.ann")

    keep_uris = top_pop_filter(base_in, keep_top_frac=args.pop_keep_top_frac)
    pop_list = top_global_pop(base_in, whitelist=keep_uris, n=max(600, args.n_retrieved))
    pop_hits_template = [(u, -999.0) for u in pop_list[: min(400, args.n_retrieved)]]

    def process_record(rec) -> Dict:
        seeds = [u for u in rec.seeds if isinstance(u, str)]
        title = rec.title if isinstance(rec.title, str) else ""
        seed_set = set(seeds)

        qvec = query_vector(vectorizer, svd, norm, seeds, title, track_vecs, uri2row, title_weight=args.title_weight)
        ann_hits = ann_retrieve(ann, qvec, track_uris, topn=args.n_retrieved, search_k=args.ann_search_k) if qvec is not None else []

        best: Dict[str, float] = {}
        for u, s in ann_hits:
            if u in seed_set:
                continue
            if (u not in best) or (s > best[u]):
                best[u] = s
        for u, s in pop_hits_template:
            if u in seed_set:
                continue
            if u not in best:
                best[u] = s

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
        candidates = [u for u, _ in ranked][: args.n_retrieved]

        return {
            "pid": rec.pid,
            "split": rec.split,
            "scenario": rec.scenario,
            "K": rec.K,
            "has_title": rec.has_title,
            "seed_uris": " ".join(seeds),
            "target_uris": " ".join(rec.targets if isinstance(rec.targets, list) else []),
            "candidates": " ".join(candidates),
        }

    wrote_header_flag = [out_path.exists() and out_path.stat().st_size > 0]
    batch: List[Dict] = []

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = {ex.submit(process_record, rec): i for i, rec in enumerate(scen.itertuples(index=False))}
            for fut in as_completed(futures):
                row = fut.result()
                batch.append(row)
                if len(batch) >= args.flush_every:
                    _write_rows(batch, out_path, wrote_header_flag)
                    batch.clear()
    else:
        for rec in scen.itertuples(index=False):
            row = process_record(rec)
            batch.append(row)
            if len(batch) >= args.flush_every:
                _write_rows(batch, out_path, wrote_header_flag)
                batch.clear()

    _write_rows(batch, out_path, wrote_header_flag)
    print(f"[retrieval] appended {len(scen)} rows → {out_path}")
    print("done")

if __name__ == "__main__":
    main()
