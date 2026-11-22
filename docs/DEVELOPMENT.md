# Development

## Set up

Use Python 3.10 or newer. From the repository root, install the project in editable mode:

```bash
python -m pip install -e .
```

The repository uses standard-library unittest tests. Run them from the root with:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Reader tests that require RDKit are skipped when RDKit is not installed. Install it with `python -m pip install rdkit` to run those cases. PyTorch and PyTorch Geometric are only needed for the `pyg_pt` reader.

## Build a distribution

Install the build frontend, then create source and wheel distributions:

```bash
python -m pip install build
python -m build
```

Artifacts are written to `dist/`. Build output, test caches, virtual environments, and local tool binaries are ignored by Git.

## Release checklist

The package version is recorded in both `pyproject.toml` and `src/nsppk.py` as `__version__`. Keep them equal.

For a release:

1. Choose a semantic version. Bug fixes increment the patch component; compatible features increment the minor component; breaking API changes increment the major component.
2. Add a `CHANGELOG.md` entry for the version.
3. Update the project version in `pyproject.toml` and `__version__` in `src/nsppk.py`.
4. Run the test suite and build the distributions.
5. Commit the release changes.
6. Create an annotated `vX.Y.Z` tag on that commit and push both the branch and tag.

The existing release tags use the `v` prefix.
