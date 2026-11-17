# Local Kepler inputs

Obtain the original processed course dataset and keep it in this directory. Preserve the order of rows in `kepler_candidates.csv` to reproduce the seeded stratified split.

| Path | Needed by | Contents |
|---|---|---|
| `kepler_candidates.csv` | Both experiments | Seven engineered features, binary `label`, and stellar identifier `kepid` |
| `koi_catalog.csv` | CNN benchmark | Stellar identifiers and KOI dispositions for the eligibility audit |
| `candidates/candidatev2_*.npz` | CNN benchmark | Extracted phase-folded curves on the common 201-bin grid |
| `candidates.tgz` | Optional | Local archive of the curve files |

Use the supplied extracted curves for the benchmark. `script/kepler_pipeline2.py` records the upstream processing approach and may fetch observations if run separately.
