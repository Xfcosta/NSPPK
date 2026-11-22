# Changelog

## [1.2.1] - 2026-11-22

- Replaced the SHA-256 primitive with seeded 32-bit MurmurHash3 for faster feature hashing. This changes generated feature IDs and is not compatible with vectors or models created with earlier versions.

## [1.2.0] - 2026-11-22

- Added graph loading and batch streaming for SMILES, SDF, and PyTorch Geometric files, with custom readers, sampling, balancing, and progress reporting.
- Added node-level feature importance projection and a subgraph provenance archive for inspecting hashed features.
- Added attribute embedding and discretization workflows, edge-triplet controls, and weighted distance contributions.
- Improved graph preprocessing performance and clarified parameter validation, scikit-learn parameter updates, and clustering reproducibility.
- Expanded API, I/O, attribute, hashing, and importance documentation.

Earlier Git tags exist, but detailed notes for those releases were not available when this changelog was started.
