# Spotify Playlist Recommendation

End-to-end system for the Spotify playlist continuation challenge.  
Combines TF-IDF + SVD retrieval, ANN candidate generation, and LightGBM ranking with feature engineering.

---

## Highlights
- **TF-IDF + SVD retrieval** over track text  
- **Approximate Nearest Neighbors (Annoy)** to generate playlist candidates  
- **Rich feature engineering**: popularity, co-occurrence, title tokens, metadata joins  
- **LightGBM ranker** for supervised scoring  
- **Inference script** to run end-to-end and generate Kaggle-style submissions  

---

## Repository Structure

```
spotify-recsys/
├── ingest_mpd.py          # Parse MPD slices into parts/ & unique/ CSVs
├── vocab_features.py      # Build vocabularies, track meta, popularity, title tokens
├── retrieval_tfidf.py     # Build TF-IDF matrix over track text
├── retrieval_svd.py       # Reduce TF-IDF with SVD; save normalized track vectors
├── retrieval_ann.py       # Build/query ANN; produce candidates.csv
├── feature_engineering.py # Build playlist–track feature table from candidates
├── ranker.py              # Train LightGBM ranker; save scaler/model + preview submission
├── inference.py           # Final pipeline: retrieve + rank -> gzip submission
└── README.md
```

---

## Setup

Python ≥ 3.9. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `pandas`, `numpy`, `scikit-learn`, `lightgbm`
- `annoy`, `scipy`, `joblib`, `tqdm`

---

## Usage

### 0. Ingest MPD
Split raw Million Playlist Dataset (JSON slices) into CSV parts:

```bash
python ingest_mpd.py --mpd_dir data --out_dir artifacts
```

### 1. Vocab + Base Features
Generate vocabularies and track metadata:

```bash
python vocab_features.py --in_dir artifacts --out_dir artifacts
```

### 2. Retrieval (TF-IDF → SVD → ANN)
```bash
# Build TF-IDF matrix
python retrieval_tfidf.py --in_dir artifacts --out_dir artifacts

# Reduce with SVD and normalize
python retrieval_svd.py --out_dir artifacts --svd_dim 128

# Build ANN index and write candidates
python retrieval_ann.py --in_dir artifacts --out_dir artifacts --n_retrieved 800
```

### 3. Feature Engineering + Ranking
```bash
# Build features for playlist–track pairs
python feature_engineering.py \
  --in_dir artifacts \
  --out_dir artifacts \
  --candidates_csv artifacts/retrieval/candidates.csv

# Train LightGBM ranker and preview submission
python ranker.py --in_dir artifacts --out_dir artifacts
```

### 4. Inference (End-to-End)
Generate a submission for a challenge set:

```bash
python inference.py \
  --in_dir ./artifacts \
  --out_dir ./artifacts \
  --challenge_json /path/to/challenge_set.json \
  --team_name "your_team" \
  --team_email "you@example.com"
```

This produces a gzip file ready for submission.

---

## Notes
- Artifacts (vocabs, features, models) are written under `artifacts/`.  
- Default configs (e.g. ANN neighbors, SVD dimension) can be tuned for performance.

---

## References
- [Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)  
