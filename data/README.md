# Data

Real demonstration data is intentionally not stored in Git.

The pipeline has three data stages:

1. `data/raw/episode_XXXXXX/`
   - dual RealSense RGB-D
   - Piper follower state
   - timestamps
2. remote conversion to FP3 HDF5
3. validated `trajectory_pcd.h5`

For a public dataset release, use a separate dataset host or release mechanism
and publish checksums plus a manifest here.

See `docs/DATASET_SCHEMA.md`.
