# vLLM-Omni Development Guide

## Quick Start

```bash
# 1. Create environment (Python 3.12 recommended, compatible with 3.10-3.12)
uv venv --python 3.12 --seed
source .venv/bin/activate

# 2. Install vLLM (the base dependency)
uv pip install vllm --torch-backend=auto

# 3. Install vllm-omni in editable mode
uv pip install -e .

# 4. Install dev tools
uv pip install pre-commit
pre-commit install
```

## Do I Need to Rebuild After Code Changes?

**No.** vllm-omni is a pure Python package — there are no C/CUDA extensions in `vllm_omni/`. The editable install (`-e .`) creates a symlink to your source tree, so any Python code changes take effect immediately.

You **only** need to re-run `uv pip install -e .` when:

- `pyproject.toml` or `setup.py` changes (e.g., new dependencies, entry points)
- Package metadata or version configuration changes

## Linting

vllm-omni uses `pre-commit` with `ruff` (line length 120, Google Python style guide).

```bash
pre-commit run                # runs on staged files
pre-commit run --all-files    # runs on all files
```

## Testing

vllm-omni uses `pytest`. Most tests require GPU access.

```bash
pytest                        # run all tests
pytest tests/path/to/test.py  # run specific test file
```

Test markers include `core_model` (L1/L2, run per PR), `advanced_model` (L3/L4, nightly), `diffusion`, `omni`, etc. See `pyproject.toml` for the full list.

## Commits

Commits must include a `Signed-off-by` header (DCO requirement):

```bash
git commit -s -m "your message"
```

## Pull Request Conventions

### Title Prefixes

| Prefix | Use For |
|--------|---------|
| `[Bugfix]` | Bug fixes |
| `[Model]` | Adding/improving models (include model name) |
| `[Core]` | Core logic (OmniProcessor, OmniARScheduler, etc.) |
| `[Frontend]` | Frontend changes (OpenAI API server, Omni/AsyncOmni) |
| `[Kernel]` | CUDA/compute kernel changes |
| `[Doc]` | Documentation |
| `[CI/Build]` | CI or build improvements |
| `[Hardware][Vendor]` | Hardware-specific (e.g., `[Hardware][Ascend]`) |
| `[Misc]` | Other |

### Before Requesting Review

- Run L1 and L2 tests locally and attach results
- Pass all linter checks
- For large changes (>500 LOC excluding kernel/data/config/test), open an RFC issue first

## Documentation

```bash
uv pip install -e ".[docs]"
mkdocs serve                                        # full build (~10 min)
API_AUTONAV_EXCLUDE=vllm_omni mkdocs serve          # skip API ref (~15 sec)
```

Preview at http://127.0.0.1:8000/

## Weekly Meetings

Developer meetings are held **every Tuesday at 19:30 PDT**. See [meeting notes](https://tinyurl.com/vllm-omni-meeting).
