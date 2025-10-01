import argparse, re, math, glob
import pandas as pd
from collections import Counter
from pathlib import Path

def output_directories(base_out: Path):
    vocab_dir = base_out / "vocab"
    feat_dir = base_out / "features"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)
    return vocab_dir, feat_dir


def build_vocab(df: pd.DataFrame, key_col: str, name_col: str | None, out_csv: Path):
    id_col = key_col.replace("_uri", "_id") if key_col.endswith("_uri") else f"{key_col}_id"
    rows = []
    if name_col:
        rows.append({key_col: "__OOV__", id_col: 0, name_col: "__OOV__"})
    else:
        rows.append({key_col: "__OOV__", id_col: 0})
    keep_cols = [key_col] + ([name_col] if name_col else [])
    uniq = df[keep_cols].drop_duplicates().sort_values(key_col)
    for i, rec in enumerate(uniq.itertuples(index=False), start=1):
        if name_col:
            rows.append({key_col: getattr(rec, key_col), id_col: i, name_col: getattr(rec, name_col)})
        else:
            rows.append({key_col: getattr(rec, key_col), id_col: i})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def build_vocabs(in_dir: Path, vocab_dir: Path, feat_dir: Path):
    uniq_dir = in_dir / "unique"
    tracks_csv = uniq_dir / "tracks.csv"
    artists_csv = uniq_dir / "artists.csv"
    albums_csv = uniq_dir / "albums.csv"
    if not (tracks_csv.exists() and artists_csv.exists() and albums_csv.exists()):
        raise FileNotFoundError("Expected uniques at artifacts/unique/{tracks,artists,albums}.csv")
    tracks_df = pd.read_csv(tracks_csv, usecols=["track_uri", "track_name", "artist_uri", "album_uri"])
    artists_df = pd.read_csv(artists_csv, usecols=["artist_uri", "artist_name"])
    albums_df = pd.read_csv(albums_csv, usecols=["album_uri", "album_name"])
    track_vocab = build_vocab(tracks_df, "track_uri", "track_name", vocab_dir / "track_vocab.csv")
    artist_vocab = build_vocab(artists_df, "artist_uri", "artist_name", vocab_dir / "artist_vocab.csv")
    album_vocab = build_vocab(albums_df, "album_uri", "album_name", vocab_dir / "album_vocab.csv")
    meta = tracks_df.merge(track_vocab[["track_uri", "track_id"]], on="track_uri", how="left")
    meta = meta.merge(artists_df, on="artist_uri", how="left")
    meta = meta.merge(artist_vocab[["artist_uri", "artist_id"]], on="artist_uri", how="left")
    meta = meta.merge(albums_df, on="album_uri", how="left")
    meta = meta.merge(album_vocab[["album_uri", "album_id"]], on="album_uri", how="left")
    meta[["track_id", "artist_id", "album_id"]] = meta[["track_id", "artist_id", "album_id"]].fillna(0).astype(int)
    cols = ["track_id", "track_uri", "track_name", "artist_id", "artist_uri", "artist_name", "album_id", "album_uri", "album_name"]
    meta = meta[cols].drop_duplicates("track_uri").sort_values("track_id")
    meta.to_csv(feat_dir / "track_meta.csv", index=False)
    return track_vocab, artist_vocab, album_vocab, meta


def track_popularity(in_dir: Path, feat_dir: Path):
    parts_pt = sorted(glob.glob(str(in_dir / "parts" / "playlist_tracks" / "part-*.csv")))
    if not parts_pt:
        raise FileNotFoundError("parts not found under artifacts/parts/playlist_tracks/part-*.csv")
    pop_counter = Counter()
    for p in parts_pt:
        df = pd.read_csv(p, usecols=["track_uri"])
        pop_counter.update(df["track_uri"].dropna().astype(str).values)
    rows = []
    for uri, cnt in pop_counter.items():
        rows.append({"track_uri": uri, "pop_count": int(cnt), "pop_log1p": math.log1p(int(cnt))})
    pop_df = pd.DataFrame(rows)
    pop_df.to_csv(feat_dir / "track_popularity.csv", index=False)
    return pop_df


def track_features(track_meta: pd.DataFrame, pop_df: pd.DataFrame, feat_dir: Path):
    df = track_meta.merge(pop_df, on="track_uri", how="left")
    df[["pop_count", "pop_log1p"]] = df[["pop_count", "pop_log1p"]].fillna(0)
    keep = ["track_id", "track_uri", "artist_id", "album_id", "pop_count", "pop_log1p"]
    out = df[keep].sort_values("track_id")
    out.to_csv(feat_dir / "track_features.csv", index=False)
    return out


_TITLE_CLEAN_RE = re.compile(r"[^a-z0-9'\s]")


def tokenize_title(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = _TITLE_CLEAN_RE.sub(" ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    return toks[:20]


def title_prep(in_dir: Path, feat_dir: Path):
    parts_pl = sorted(glob.glob(str(in_dir / "parts" / "playlists" / "part-*.csv")))
    if not parts_pl:
        raise FileNotFoundError("parts not found under artifacts/parts/playlists/part-*.csv")
    token_counter = Counter()
    title_rows = []
    for p in parts_pl:
        df = pd.read_csv(p, usecols=["pid", "name"])
        for r in df.itertuples(index=False):
            toks = tokenize_title(getattr(r, "name"))
            if toks:
                token_counter.update(toks)
            title_rows.append({"pid": int(getattr(r, "pid")), "tokens": " ".join(toks)})
    tt = pd.DataFrame(title_rows).drop_duplicates("pid")
    tt.to_csv(feat_dir / "title_tokens.csv", index=False)
    tv = pd.DataFrame([{"token": k, "count": v} for k, v in token_counter.most_common()])
    tv.to_csv(feat_dir / "token_vocab.csv", index=False)
    return tt, tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="artifacts")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    vocab_dir, feat_dir = output_directories(out_dir)
    _, _, _, track_meta = build_vocabs(in_dir, vocab_dir, feat_dir)
    pop_df = track_popularity(in_dir, feat_dir)
    _ = track_features(track_meta, pop_df, feat_dir)
    _tt, _tv = title_prep(in_dir, feat_dir)
    print("done")
    print(f"{vocab_dir}/track_vocab.csv")
    print(f"{vocab_dir}/artist_vocab.csv")
    print(f"{vocab_dir}/album_vocab.csv")
    print(f"{feat_dir}/track_meta.csv")
    print(f"{feat_dir}/track_popularity.csv")
    print(f"{feat_dir}/track_features.csv")
    print(f"{feat_dir}/title_tokens.csv")
    print(f"{feat_dir}/token_vocab.csv")


if __name__ == "__main__":
    main()
