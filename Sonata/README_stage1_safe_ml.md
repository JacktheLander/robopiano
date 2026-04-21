# Safe Stage 1 (medium RP1M 300) — Sonata

## Why the previous run failed

The legacy `medium.yaml` primitive profile combined **note-aligned segmentation** (many short segments per episode) with a **partitioned global GMM sweep** using **large `gmm_k_candidates` and full covariance**. That can fragment the motion space into **hundreds of primitives**, explode **segment row counts**, and spend **many CPU hours** fitting models—matching the failure mode (~743 primitives, ~80 GB, ~22 h, still not viable).

## What this path changes

- **Segmentation**: `learned_boundary_safe` uses bounded boundary evidence (actions, joint motion, goals/piano) with **hard caps** per episode and per song (`max_candidate_boundaries_per_episode`, `max_segments_per_song`, `max_total_segments`).
- **Storage**: **Slim cache only** by default (`save_raw_segment_chunks: false`, `write_slim_cache: true`). Optional `slim_feature_dtype: float16` reduces feature shard size.
- **Clustering**: `primitive_discovery_method: hdbscan_then_local_gmm` replaces the giant **K-sweep + full-covariance GMM** default. **HDBSCAN** finds coarse modes; **tiny diagonal local GMMs** (K∈{2,3}) split only very large clusters; **merge** enforces `max_clusters_total`.
- **Noise**: HDBSCAN label `-1` becomes `primitive_noise`, logged to `clustering/noise_segments.csv`, **excluded** from GMR and token tables.
- **GMR**: Capped samples per primitive (`max_segments_per_primitive_for_gmr`), skips unstable clusters (size, dispersion, MSE), and logs **kept vs skipped** counts.
- **Guards**: `storage_guard`, `runtime_guard_projected_walltime`, `max_episode_processing_seconds_*`, and `primitive_count_guard` stop unsafe runs early with explicit errors.

## Dataset (explicit)

- **Zarr root**: `/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr`
- **Profile**: RP1M 300 **medium** — `configs/data/medium.yaml` uses `max_episodes: 20` per song.
- Export on the cluster: `export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr`

## Commands (tmux → conda → Slurm)

```bash
# SSH to cluster, then:
bash Sonata/scripts/tmux_stage1_safe_ml.sh
tmux attach -t sonata_stage1_safe_ml

conda activate sonata
cd /WAVE/projects/ECEN-524-Wi26/robopiano/Sonata
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr

# Submit (adjust SONATA_ROOT / OUT_BASE if needed)
sbatch Sonata/scripts/slurm_stage1_safe_ml.sh

# Monitor
squeue -u "$USER"

# Logs (paths match #SBATCH in slurm script)
tail -f /WAVE/datasets/ccoelho_lab-jlanders/outputs_run3/stage1_safe_ml/logs/sonata_s1_safe-*.out
```

### Local smoke (no full dataset)

```bash
conda activate sonata
cd Sonata
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
python scripts/train_primitives.py --profile debug --config configs/primitive/debug.yaml
```

## Expected outputs

Under `OUT_BASE/primitives` (Slurm) or resolved `output_root` from the YAML:

- `slim/` — online slim shards (features, GMR targets, index CSVs)
- `clustering/segment_assignments*.csv` — per-segment labels
- `clustering/gmm_sweep*.csv` — diagnostics (HDBSCAN path uses JSON text columns)
- `library/primitive_library*.csv` + `primitive_*_prior.npz`
- `tokens/primitive_tokens*.csv` + `primitive_vocabulary.json`
- `metrics/stage1_metrics.json` — **inspect `final_primitive_count`, `noise_rows`, `num_skipped_clusters`, storage bytes first**

## Guard thresholds (defaults in `medium_safe_ml.yaml`)

See that file for authoritative numbers (`max_total_segments`, `max_clusters_total`, `max_storage_bytes_*`, `max_fit_rows`, `max_walltime_seconds`, etc.).

## Resume

- Keep `force: false`.
- Re-run the same `train_primitives.py` command; completed episodes are skipped via slim progress.
- If a slim chunk was partial, delete the incomplete chunk manifests listed in logs, then resume.

## Closed-loop (small eval only)

```bash
python scripts/closed_loop_primitives.py \
  --primitive-config configs/primitive/medium_safe_ml.yaml \
  --eval-config configs/evaluation/primitives_online_safe.yaml \
  --data-profile medium
```

## New dependency

- `hdbscan` (listed in `Sonata/requirements.txt`)

## Unit tests (conda `sonata` recommended)

```bash
conda activate sonata
cd Sonata
export PYTHONPATH=src
pytest -q tests/primitives/test_learned_boundary_safe.py tests/primitives/test_online_stage1.py
```
