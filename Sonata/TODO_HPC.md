# Sonata-3 HPC Integration TODOs

These are the remaining items that depend on cluster-specific deployment details or on RoboPianist environment wiring that is not fully recoverable from RP1M files alone.

1. Export `RP1M_300_ROOT` to the real shared-storage RP1M path on the cluster before launching Sonata.
2. Decide where cached manifests, segment chunks, and feature bundles should live:
   recommended split is shared read-only dataset on network storage and writable caches on node-local SSD or scratch.
3. Validate whether the full RP1M installation also ships `.proto` or MIDI score files, or whether score lookup should always fall back to `goals`.
4. Add cluster launcher wrappers for Slurm or the local scheduler so each stage can run as a separate resumable job.
5. Finish a RoboPianist-to-MJX observation/reward adapter if full GPU rollout evaluation is required:
   current code only provides batched MuJoCo physics stepping.
6. Confirm the exact action dimensionality used by the target RoboPianist task on the cluster:
   RP1M actions are typically 39D while dm_control tasks may expect 45D with wrist or sustain padding.
7. If mixed precision is desired on the cluster GPUs, test diffusion stability before enabling it by default.
8. Add distributed data-parallel training only if single-node GPU throughput becomes the bottleneck; current code is single-process by design.
