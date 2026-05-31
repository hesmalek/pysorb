# Release Checklist

This package publishes to PyPI as `pysorb`.

## Version

Update the package version in:

- `pyproject.toml`
- `src/pysorb/__init__.py`

Use semantic versioning:

- Patch fixes: `0.2.1`
- New backwards-compatible APIs: `0.3.0`
- Stable public API: `1.0.0`

## Local Checks

Run the smoke test from the package root:

```bash
set PYTHONPATH=src
set PYTHONDONTWRITEBYTECODE=1
python tests/smoke_test.py
```

On PowerShell:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:NUMBA_CACHE_DIR = Join-Path $env:TEMP "pysorb_numba_cache"
python .\tests\smoke_test.py
```

## Build

The project may need to be built outside OneDrive if setuptools cannot write
`.egg-info` metadata in-place.

Standard build command:

```bash
python -m build
```

Offline/no-isolation build command:

```bash
python -m build --no-isolation
```

If OneDrive blocks metadata writes, copy `pyproject.toml`, `README.md`,
`setup.cfg`, `src/pysorb/*.py`, and `tests/*.py` to a temporary folder outside
OneDrive and run:

```bash
python -m build --wheel --sdist --no-isolation
```

## Validate Distributions

```bash
python -m twine check dist/*
```

## Upload

Upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Then test installation from TestPyPI in a clean environment.

Upload to PyPI:

```bash
python -m twine upload dist/*
```

PyPI versions cannot be overwritten. If an upload succeeds and a problem is
found later, increment the version and publish a new release.
