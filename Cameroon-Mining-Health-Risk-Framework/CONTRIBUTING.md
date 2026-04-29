# Contributing

Contributions, bug reports and dataset extensions are welcome.

## How to contribute

1. Fork and clone the parent repository (`rogerpanel/CV`).
2. Create a feature branch from `claude/environmental-health-risk-modeling-pS8PC`.
3. Run the test suite locally:

   ```bash
   cd Cameroon-Mining-Health-Risk-Framework
   pytest tests/
   ```

4. Open a pull request that:
   * adds a unit test if you change behaviour;
   * updates the relevant doc page (`docs/`);
   * does **not** modify YAML defaults without a methodological justification.

## Style

* Black-formatted (88 cols), `ruff` lint, type-annotated where practical.
* Docstrings in NumPy style.
* New analytical methods belong in their own subpackage with an `__init__.py`
  that re-exports the public API.

## Adding a new method

1. Implement under `src/<topic>/your_method.py` with a docstring including the
   formula and the original citation.
2. Add classification thresholds (if any) to `config/thresholds.yaml`.
3. Re-export the function in the subpackage's `__init__.py`.
4. Add unit tests under `tests/`.
5. Update `docs/METHODOLOGY.md` and `docs/REFERENCES.md`.
