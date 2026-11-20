# Adding SODE-Guard to the RobustIDPS.ai platform as a detector branch

This wires the **real** trained SODE-Guard into the platform backend, alongside
the built-in methods (CT-TGNN, MambaShield, …), so it appears in `/api/model_info`
and answers at its own endpoint. Nothing in the platform's existing models is
changed; SODE-Guard is added as an extra detector.

`robustidps.ai` is a separate repository, so these files live here in the
SODE-Guard repo and are copied into the backend image at build time. Three small,
reversible edits to `backend/main.py` complete the wiring.

## 1. Files to copy into `backend/`

```
sode_guard_detector.py      # loads the trained model, scores flows
platform_router.py          # FastAPI routes + the model_info branch entry
```

Copy both into the platform's `backend/` directory (next to `main.py`).

## 2. Mount the model and the SODE-Guard code

The detector needs the SODE-Guard package (for `src/` and `scripts/v2_run.py`)
and one saved run folder with weights. Add to the backend image / compose:

```yaml
# docker-compose.prod.yml, backend service
  environment:
    - SODE_GUARD_ROOT=/opt/sode-guard
    - SODE_GUARD_RUN=/opt/sode-guard-run        # meta.json, full.npz, seed_*/*.pt
    - SODE_GUARD_MODEL=sode_at                  # or sode_rs_c2 for the certified head
  volumes:
    - ./vendor/sode-guard:/opt/sode-guard:ro    # the SODE-Guard checkout (src + scripts)
    - ./vendor/sode-guard-run:/opt/sode-guard-run:ro
```

`sode-guard-run` is the whole run folder brought back from Colab for the corpus
whose schema matches the platform's uploads (CIC-IoT-2023 is the natural choice,
since the platform's primary schema is CIC-IoT-2023). It must contain
`meta.json`, `full.npz` and `seed_0/<model>.pt`.

Add the SODE-Guard runtime deps to `backend/requirements.txt` (already mostly
present): `torch`, `torchdiffeq`, `scikit-learn`, `pandas`, `numpy`.

## 3. Three edits to `backend/main.py`

```python
# (a) near the other imports
from platform_router import router as sode_router, branch_entry

# (b) right after `app = FastAPI(...)` and the CORS middleware
app.include_router(sode_router)

# (c) in model_info(), add SODE-Guard to the advertised branches
    "branch_names": SurrogateIDS.BRANCH_NAMES + [branch_entry()],
```

That is all. Rebuild: `docker compose -f docker-compose.prod.yml up -d --build`.

## 4. Verify

```bash
curl localhost:8000/api/sode_guard/info                       # {"ready": true, ...}
curl -F file=@sample_data/ciciot_sample.csv \
     localhost:8000/api/predict/sode_guard | head            # per-flow predictions
```

The Live Monitor's method list will show **SODE-Guard (Neural SDE, certified)**,
and the testbed's `scoring/score_platform.py` can point `--url` at this backend to
score realised traffic through SODE-Guard and the platform's own ensemble on the
same flows.

## Notes

* The detector aligns the platform's uploaded columns to SODE-Guard's feature
  schema **by name** and applies the training-set scaling from `full.npz`; absent
  features are imputed to the training median. Scores are only meaningful when the
  uploaded schema matches the run's corpus (CIC-IoT-2023 ↔ CIC-IoT-2023).
* If the weights are not mounted, `/api/sode_guard/info` reports `ready: false`
  and the predict route returns HTTP 503 instead of crashing the app — the rest
  of the platform is unaffected.
* This integration is read-only toward the platform's own models; remove the
  three edits to revert.
