import argparse, json, re, sys
import pandas as pd
from pathlib import Path

SLICE_RE = re.compile(r"^mpd\.slice\.\d+-\d+\.json$")

def output_directories(out_dir: Path):
    parts_pl = out_dir / "parts" / "playlists"
    parts_pt = out_dir / "parts" / "playlist_tracks"
    uniq_dir = out_dir / "unique"
    for d in (parts_pl, parts_pt, uniq_dir):
        d.mkdir(parents=True, exist_ok=True)
    return parts_pl, parts_pt, uniq_dir

def iter_files(mpd_dir: Path):
    for p in sorted(mpd_dir.iterdir()):
        if p.is_file() and SLICE_RE.match(p.name):
            yield p

def write_csv(df: pd.DataFrame, path: Path, mode="w", header=True):
    df.to_csv(path, index=False, mode=mode, header=header)


def unique_csv(seen_keys: set, rows: list, cols: list, out_csv: Path):
    if not rows:
        return
    new = []
    for r in rows:
        k = r[0]
        if k and k not in seen_keys:
            seen_keys.add(k)
            new.append(r)
    if not new:
        return
    df = pd.DataFrame(new, columns=cols)
    exists = out_csv.exists()
    write_csv(df, out_csv, mode="a", header=not exists)

def processing(
    slice_path: Path,
    part_idx: int,
    parts_pl: Path,
    parts_pt: Path,
    uniq_dir: Path,
    seen_tracks: set,
    seen_artists: set,
    seen_albums: set,
):
    data = json.loads(slice_path.read_text(encoding="utf-8"))

    playlists_rows = []
    playlist_tracks_rows = []
    tracks_rows = []
    artists_rows = []
    albums_rows = []

    for pl in data["playlists"]:
        pid = pl.get("pid")
        playlists_rows.append({
            "pid": pid,
            "name": pl.get("name"),
            "description": pl.get("description"),
            "modified_at": pl.get("modified_at"),
            "num_artists": pl.get("num_artists"),
            "num_albums": pl.get("num_albums"),
            "num_tracks": pl.get("num_tracks"),
            "num_followers": pl.get("num_followers"),
            "num_edits": pl.get("num_edits"),
            "duration_ms": pl.get("duration_ms"),
        })

        for t in pl.get("tracks", []):
            track_uri = t.get("track_uri")
            artist_uri = t.get("artist_uri")
            album_uri  = t.get("album_uri")

            playlist_tracks_rows.append({
                "pid": pid,
                "pos": t.get("pos"),
                "track_uri": track_uri,
                "track_name": t.get("track_name"),
                "artist_uri": artist_uri,
                "artist_name": t.get("artist_name"),
                "album_uri": album_uri,
                "album_name": t.get("album_name"),
                "duration_ms": t.get("duration_ms"),
            })

            tracks_rows.append((track_uri, t.get("track_name"), artist_uri, album_uri, t.get("duration_ms")))
            artists_rows.append((artist_uri, t.get("artist_name")))
            albums_rows.append((album_uri, t.get("album_name")))

    pl_df = pd.DataFrame(playlists_rows, columns=[
        "pid","name","description","modified_at","num_artists","num_albums",
        "num_tracks","num_followers","num_edits","duration_ms"
    ])
    pt_df = pd.DataFrame(playlist_tracks_rows, columns=[
        "pid","pos","track_uri","track_name","artist_uri","artist_name","album_uri","album_name","duration_ms"
    ])

    write_csv(pl_df, parts_pl / f"part-{part_idx:05d}.csv")
    write_csv(pt_df, parts_pt / f"part-{part_idx:05d}.csv")

    unique_csv(seen_tracks, tracks_rows,
               ["track_uri","track_name","artist_uri","album_uri","duration_ms"],
               uniq_dir / "tracks.csv")
    unique_csv(seen_artists, artists_rows,
               ["artist_uri","artist_name"],
               uniq_dir / "artists.csv")
    unique_csv(seen_albums, albums_rows,
               ["album_uri","album_name"],
               uniq_dir / "albums.csv")

def main():
    ap = argparse.ArgumentParser(description="Minimal MPD ingest (pandas-only, CSV)")
    ap.add_argument("--mpd_dir", type=str, default="data", help="Directory with mpd.slice.*.json")
    ap.add_argument("--out_dir", type=str, default="artifacts", help="Output directory")
    args = ap.parse_args()

    mpd_dir = Path(args.mpd_dir)
    out_dir = Path(args.out_dir)

    if not mpd_dir.exists():
        print(f"[err] mpd_dir not found: {mpd_dir}")
        sys.exit(1)

    parts_pl, parts_pt, uniq_dir = output_directories(out_dir)

    seen_tracks, seen_artists, seen_albums = set(), set(), set()

    for idx, slice_path in enumerate(iter_files(mpd_dir), start=1):
        print(f"[ingest] {idx:05d} :: {slice_path.name}")
        processing(
            slice_path,
            part_idx=idx,
            parts_pl=parts_pl,
            parts_pt=parts_pt,
            uniq_dir=uniq_dir,
            seen_tracks=seen_tracks,
            seen_artists=seen_artists,
            seen_albums=seen_albums,
        )

    print("\n[done] Outputs:")
    print(f"  - {parts_pl}")
    print(f"  - {parts_pt}")
    print(f"  - {uniq_dir}")

if __name__ == "__main__":
    main()
