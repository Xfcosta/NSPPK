# Changelog

## [3.5.0] - 2026-11-27

- Added Bayesian optimization for graph-level classifier and regressor wrappers to tune NSPPK encoder parameters with cross-validation, callbacks, and trial history.
- Added the optional `tuning` dependency extra, API and tuning guides, and tests for classifier and regressor searches.
- Added a runnable HIV Bayesian optimization notebook and refreshed the molecular learning-curve experiment to compare batch and incremental training workflows.

## [3.4.0] - 2026-11-27

- Added incremental `partial_fit` and source-streaming `partial_fit_from` methods to the graph-level classifier and regressor wrappers when their estimator supports incremental updates.
- Fit learned attribute preprocessing once on the first training batch and keep it fixed across subsequent updates; added batch callbacks for streaming metrics and progress hooks.
- Added the `SGDRegressor` shortcut for incremental regression while preserving Ridge as the `linear` default.
- Updated the molecular learning-curve notebook to train through `NSPPKClassifier.partial_fit_from` and compare incremental updates with regular estimator fitting.
- Expanded the predictor, streaming, and getting-started documentation for incremental training.

## [3.3.0] - 2026-11-26

- Added `parameters` lists to graph, node, and edge encoders so one output can accumulate features from multiple radius, distance, connector, weighting, and simplex configurations.
- Revised simplex fragment hashing to use cached edge hashes and order-invariant hash sets, with root-label and distance context. Positive-distance fragments now use the union of endpoint neighborhoods rather than all cross-neighborhood fragment pairs.
- Changed edge label hashing to preserve endpoint invariance for undirected edges while retaining directed source/target order.
- Added a 4D node-attribute Gaussian-noise notebook demonstration and refreshed simplex and recursive-feature examples.
- Documented the feature-union API and revised hashing formulas.

## [3.2.0] - 2026-11-25

- Added inclusive `(min, max)` tuple ranges for `radius`, `distance`, and `connector`, while preserving scalar cumulative ranges from zero.
- Optimized distance-zero simplex features by enumerating fragments once as unary neighborhood features instead of forming self-pairs.
- Added bitset-based clique enumeration through simplex dimension 3, with NetworkX fallback for higher dimensions.
- Documented tuple ranges, distance-zero simplex signatures, and clique enumeration behavior.

## [3.1.0] - 2026-11-25

- Added optional per-graph root-node filtering to graph-level and node-level NSPPK transforms.
- Added `RecursiveNSPPKClassifier` with importance-guided stages, cumulative feature output, and scikit-learn classifier compatibility.
- Added a configurable HIV experiment comparing multiple unfiltered baselines with individual and cumulative recursive stages.
- Documented the recursive classifier and root-filtering API.

## [3.0.0] - 2026-11-25

- Replaced `use_edges_as_features` with `use_simplex_features` and `simplex_feature_max_order` for optional node, edge, triangle, and higher-order simplex fragments.
- Added cached simplex fragment pairing for endpoint neighborhoods and expanded shortest-path unions, with directed arc orientation and multigraph edge patterns retained in hashes.
- Added a dense-graph fragment and HIV runtime/accuracy notebook.
- Removed the legacy `use_edges_as_features` parameter; this is a breaking API change.

## [2.1.0] - 2026-11-25

- Added support for directed graphs and multigraphs, preserving edge direction, parallel edges, and self-loops in structural features and archives.
- Added a runnable notebook comparing multigraph, directed, and undirected graph vectorizations.
- Documented directed and multigraph semantics across the API and hashing guides.

## [2.0.0] - 2026-11-22

- Added `EdgeNSPPK`, which returns one structural feature row per edge by summing the endpoint node rows.
- Activated `edge_attribute_key` for graph-level edge attribute weighting and added independent edge embedding and clustering settings.
- Added Read the Docs guidance and a runnable notebook covering edge row order, interaction columns, learned labels, graph aggregation, and empty graphs.
- Enabling `edge_attribute_key` now changes graph-vector widths and values; leave it unset to retain the node-only layout.

## [1.3.6] - 2026-11-22

- Improved structural feature extraction by caching connector provenance, skipping archive-only work during ordinary transforms, and reducing repeated induced-edge visits.
- Tuned the graph-level multiprocessing threshold to avoid pool overhead on smaller batches.
- Preserved feature outputs while reducing median structural extraction time by 2.36x on the documented 64-graph HIV benchmark.


## [1.3.5] - 2026-11-22

- Updated the README layout and centered project logo.
- Added benchmark profiles and results for current graph workloads.

## [1.3.4] - 2026-11-22

- Regenerated the README and documentation logo PNG from the project SVG source.

## [1.3.3] - 2026-11-22

- Updated the README and documentation landing page to use the project logo asset at a compact size.

## [1.3.2] - 2026-11-22

- Added the NSPPK graph mark to the README and documentation landing page.
- Refreshed the PyPI project description with the centered README logo.

## [1.3.1] - 2026-11-22

- Prepared the first PyPI distribution and added `pip install nsppk` installation instructions.
- Added project links and package classifiers to the distribution metadata.

## [1.3.0] - 2026-11-22

- Added `NSPPKClassifier` and `NSPPKRegressor` wrappers for graph-level supervised learning with built-in `linear` and `rf` shortcuts or custom scikit-learn estimators.
- Switched structural feature hashing from MurmurHash3 to BLAKE2s with a direct 32-bit digest conversion. Feature IDs differ from version 1.2.1.

## [1.2.1] - 2026-11-22

- Replaced the SHA-256 primitive with seeded 32-bit MurmurHash3 for faster feature hashing. This changes generated feature IDs and is not compatible with vectors or models created with earlier versions.

## [1.2.0] - 2026-11-22

- Added graph loading and batch streaming for SMILES, SDF, and PyTorch Geometric files, with custom readers, sampling, balancing, and progress reporting.
- Added node-level feature importance projection and a subgraph provenance archive for inspecting hashed features.
- Added attribute embedding and discretization workflows, edge-triplet controls, and weighted distance contributions.
- Improved graph preprocessing performance and clarified parameter validation, scikit-learn parameter updates, and clustering reproducibility.
- Expanded API, I/O, attribute, hashing, and importance documentation.

Earlier Git tags exist, but detailed notes for those releases were not available when this changelog was started.
