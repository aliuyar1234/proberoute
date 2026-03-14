# Public Release Checklist

Use this checklist before pushing the repo to GitHub.

## Required
- Verify tests still pass:
  - `.\.venv\Scripts\python -m pytest tests\unit tests\integration`
- Confirm `git status --ignored` does not show runtime artifacts being tracked.
- Confirm `outputs/`, `_archive/`, `.venv/`, and temporary bundles are ignored.
- Confirm no generated paper drafts or experiment outputs are staged.
- Confirm no model weights or dataset payloads are present in tracked files.

## Recommended
- Review [README.md](README.md) as the public landing page.
- Review [LICENSE_AND_COMPLIANCE.md](LICENSE_AND_COMPLIANCE.md).
- Add a real software `LICENSE` file before publishing if the repo will be public.
- Optionally add `.github/` metadata such as issue templates, CI, and a repo description.

## Current Public Surface
- implementation code in `src/`
- configs in `configs/`
- tests in `tests/`
- smoke fixtures in `fixtures/`
- stable design/build docs in the repo root and `MODULE_SPECS/`
- paper-generation templates in `PAPER_TEMPLATES/`
