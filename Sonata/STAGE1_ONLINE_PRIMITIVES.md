# Sonata Stage 1 Online Primitive Discovery

## Old flow

Stage 1 used to:

1. Segment each episode.
2. Write raw variable-length segment chunk files under `segments/segment_chunk_*.npz`.
3. Re-open those raw chunks later to extract clustering features and GMR trajectories.

That made raw segment chunks the dominant storage term and forced a second pass over heavy intermediates.

## New flow

The default Stage 1 path now computes compact downstream-ready rows online while scanning episodes:

1. Discover segment boundaries.
2. Slice one segment at a time from the source episode.
3. Immediately compute:
   - a fixed-size `feature_vector` for global GMM fitting
   - a fixed-size canonical `gmr_target` trajectory for primitive-library fitting
   - lightweight metadata for the segment index
4. Append those compact rows into the online store under `slim/`.
5. Discard the raw segment arrays unless raw chunk saving is explicitly enabled.

The GMM is still fit globally across all segments, and the GMR library is still fit from the shared compact store after clustering.

Stage 2 does not need Stage 1 to pre-write planner families. The factored planner derives deterministic primitive-family labels online from the cached Stage 1 token table plus primitive-library statistics, so this compact Stage 1 layout remains the canonical handoff to later stages.

## Storage rationale

The online store keeps only:

- compressed feature shards
- compressed canonical GMR target shards
- segment-index CSV rows
- chunk/store manifests and episode progress logs

Raw segment chunk files are no longer required in the default path. The compatibility/debug path is controlled by `save_raw_segment_chunks`.

## Resume behavior

- Completed episodes are tracked in `slim/progress/episode_progress.jsonl`.
- Only completed compact shards are loaded on resume.
- Incomplete shard artifacts are ignored rather than merged into the segment index.
- New shard indices continue after any existing shard name, so interrupted runs do not overwrite prior outputs.
- Existing legacy raw chunks can be migrated into the compact store with `migrate_existing_segment_chunks: true`.

## Config knobs

- `online_segment_processing`: enable the compact online Stage 1 path.
- `save_raw_segment_chunks`: keep legacy raw segment chunks for debugging.
- `online_storage_format`: currently `npz_shards`.
- `gmr_resample_steps`: fixed canonical horizon for saved GMR trajectories.
- `segment_chunk_size`: shard target size for compact online writes.
- `migrate_existing_segment_chunks`: backfill the online store from old raw chunk outputs.

Legacy aliases such as `write_slim_cache`, `write_full_segment_cache`, and `gmr_horizon` are still accepted.
