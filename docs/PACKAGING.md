# Source packaging record

Prepared on 2026-11-17 for the paper's current three-LLM, two-game experiment.
The package is self-contained and does not depend on the original checkout.

## Retained implementation

- The complete `bsi` inference/client/sandbox modules and `p2_opponent` game and
  agent modules. Their computational behavior is unchanged.
- The existing matrix runner, generic matrix analysis, outcome/source-review
  analysis, and offline demo. Compatible earlier agent options remain inside
  these shared modules; release configurations select the paper's v6 methods.
- Synthetic framework tests, including v6 archive/heredity/residual/exploration
  behavior, the No-write operation restrictions, and GLM's JSON protocol.
- The paper's existing candidate-funnel accounting, extracted into
  `scripts/paper_behavior.py` with its synthetic regression tests.

No run directories, original review annotations, experiment reports, generated
model programs, manuscript assets, API keys, caches, or repository history were
copied. The README's numerical tables are reported paper aggregates, not
bundled episode records.

## Configuration mapping

| Release configuration | Original configuration source |
|---|---|
| `intern_full.yaml` | `matrix_v6_s0.yaml` + `matrix_v6_s1.yaml`, Full arm |
| `deepseek_full.yaml` | `matrix_v6_deepseek.yaml`, Full arm |
| `glm_flash_full.yaml` | `matrix_v6_glm53_flash_json.yaml`, completed main-arm protocol |
| `*_no_write.yaml` | Corresponding Full settings with `menu_mode: select_only`, `n_variation: 0` |
| `controls.yaml` | `matrix_v6_controls.yaml`, `menu_all_e30` only |

Intern's seed files are consolidated into seeds `[0, 1]`; deterministic job keys
and sampling remain unchanged. The No-write configs include an Intern
reproduction recipe as well as DeepSeek and GLM-Flash. Only the three paper
model identities are distributed as executable experiment configurations.

Each LLM matrix has explicit proposer/decider roles. The original token budgets,
timeouts, reasoning defaults, response repairs, and GLM validation repairs are
retained. Runtime worker/request limits are reduced to 2/4, with 40 RPM and
2.5M TPM per process. Model aliases and credential-file paths from local launch
records are not carried over.

## Packaging changes

- Distribution name changed to `unknown-coverage-game`; Python imports remain
  `bsi` and `p2_opponent`.
- Added English README, reproduction instructions, `.env.example`, and ignore
  rules for secrets and runtime output. `requirements-tested.txt` records the
  complete dependency versions used in the clean validation environment.
- Added `summarize_paper.py`, a portable adapter around the existing outcome and
  behavior functions. It computes paper means, family effects, 30-round windows,
  and both stability windows from fresh traces, without saved manuscript inputs.
- Source-review export defaults to 80 rounds and accepts `--review-tail 50`.
  Existing episode-level analysis retains its 50-round default.
- Removed the one regression test requiring historical committed review data;
  added a synthetic export/edit/load review roundtrip instead. Updated the v6
  configuration test to use the released Full/control files. Live API tests are
  not included.

## Validation

Validation used Python 3.12.3, a separate virtual environment, and temporary
output directories:

- Editable installation succeeded. All 245 offline tests passed.
- All seven configuration dry runs yielded 56 episodes each.
- One actual 150-round fixed-menu episode completed without API calls or failed
  attempts.
- Synthetic traces exercised the source-review and paper-summary CLIs, with
  numerical checks and rejection of mixed-model inputs.
- The README result table was checked against the manuscript. Its window and
  stability trend counts were checked against the analysis data.
- An independent copy outside the original checkout installed successfully;
  package imports resolved inside that copy and its three release integration
  tests passed with the pinned dependencies.
- The implementation packages match the original source modules. The release
  inventory excludes credentials, original experiment records, caches, and
  build artifacts.

Validation outputs are outside this directory. No LLM experiment or API health
request was made during packaging.
