import argparse
import numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import joblib

def build_svd_ann(X_sparse, svd_dim=128, ann_trees=50):
    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    X_red = svd.fit_transform(X_sparse)
    norm = Normalizer(copy=False)
    X_red = norm.fit_transform(X_red)
    return svd, norm, X_red

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--svd_dim", type=int, default=128)
    args = ap.parse_args()

    base = Path(args.out_dir)
    X = sparse.load_npz(base / "retrieval" / "X_tracks.npz")

    svd, norm, X_red = build_svd_ann(X, svd_dim=args.svd_dim)
    np.save(base / "retrieval" / "track_vecs.npy", X_red.astype(np.float32))
    joblib.dump(svd, base / "models" / "svd.joblib")
    joblib.dump(norm, base / "models" / "norm.joblib")

    print(f"[svd] SVD {args.svd_dim} dims → track_vecs.npy saved; models/svd.joblib & models/norm.joblib saved")

if __name__ == "__main__":
    main()
