# Unknown Coverage Games

Code for **How LLM-Based Agents Evolve Opponent Beliefs in Repeated Games under Unknown Model Coverage**.

An LLM manages executable opponent models during repeated play. It can select a
menu model, write a new program, revise an explanation, or retire it. Fixed code
fits parameters, updates model weights, and chooses actions from the combined
prediction. The agent does not know whether its initial menu contains the true
opponent mechanism.

## Start here

Use **Linux and Python 3.12** for the tested environment. The program sandbox uses POSIX process
limits. Install from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-tested.txt -e '.[dev]'
python -m pytest -q
```

Tests use synthetic episodes and mocked LLM responses. No API credentials are
needed. `requirements-tested.txt` pins the environment validated for this
release; `pyproject.toml` also records the supported dependency ranges.
To inspect the paper experiment without running it:

```bash
python scripts/run_matrix.py --config configs/intern_full.yaml --dry-run
```

Run a small **offline** check of the fixed-menu implementation:

```bash
python scripts/run_matrix.py --config configs/controls.yaml \
  --run-dir runs/smoke/controls --max-jobs-per-method 1
```

The smoke directory is separate from the full experiment. For the complete
reproduction sequence, see [Reproducing the experiments](docs/REPRODUCING.md).

## Method and experimental design

The paper uses rock–paper–scissors (`rps3`) and five-location hide-and-seek
(`hideseek5`). Each episode lasts 150 rounds. Seven opponent families, two
parameter draws, and two environment seeds give **56 episodes per LLM and
condition**. The three Full cohorts contain 168 episodes.

**Available** means the initial menu can represent the opponent's structure.
The available families are one-step response, win-stay/lose-shift, and windowed
fictitious play. **Unavailable** families use two-step responses, loss-triggered
switching, transition-based best responses, or blockwise probability drift.
Opponent identity, parameters, and coverage are hidden from the agent in both
settings.

The menu contains five model families: own-action frequency, one-step response,
win-stay/lose-shift, level-k reasoning, and fixed cycles. Every LLM arm starts
with only a statistical reference active. The reference predicts from the
previous joint action using a conditional count table, smoothed toward the
Laplace-smoothed marginal distribution with mass 2.

| Condition | Method key | Behavior |
|---|---|---|
| Full | `v6_full_e30` | LLM model selection and writing; reusable archive, mutation, residual feedback, and scheduled candidate generation |
| No-write | `v6_noexpand_e30` | Menu selection, parameter-grid changes, and retirement; no new code, code revision, or scheduled generation |
| Fixed menu | `menu_all_e30` | All five menu families active; fixed-code fitting and weighting, without LLM model management |

Full and No-write use the same model, opponents, seeds, decision checkpoints,
and exploration settings. Decisions occur after rounds
10, 24, 38, 52, 66, 80, 94, 108, 122, and 136. Full makes two scheduled generation
calls before each decision. No-write makes none. At most three named programs
are active in either LLM arm. The fixed-menu control is shared across LLMs.
Exploration uses rate 0.30 within the configured ten-round windows; its action
selection is implemented in `p2_opponent/explore.py`.

### Core implementation

```text
bsi/                    Inference, LLM client, cache, sandbox, and trace storage
p2_opponent/            Games, true opponents, model programs, and the LLM agent
configs/                Full / No-write configurations for three LLMs; shared control
scripts/run_matrix.py   Seeded experiment execution and resume
scripts/summarize_paper.py  Paper metrics from newly generated traces
scripts/analyze_outcomes.py Source-review inventory and episode trajectories
tests/                  Offline synthetic regression tests
docs/                   Reproduction and packaging notes
```

`p2_opponent/menu_agent.py` owns the model-management loop;
`menu_prompt.py` defines its prompts and allowed operations. `archive.py` tracks
candidate programs and lineage. Parameter fitting lives in `hypotheses.py`;
`bsi/posterior.py` computes weights. Ground-truth rules are in `bots.py` and the
initial menu is in `seed_programs.py`.

## Paper results

The following values are the manuscript's reported aggregates, included here
as a reference for fresh experiments. Historical traces, generated programs,
review annotations, response caches, and credentials are not distributed.
The No-write means were supplied for the manuscript as aggregate measurements;
they are preserved separately from metrics computed from the Full records.

Terminal KL is averaged over rounds **137–150**, then across episodes. Lower
is better. `Δ fixed−Full` is the mean paired fixed-menu minus Full difference.
Mechanism-match rates divide by all episodes in a cell. Decision rates use
parseable checkpoints and can overlap. `S80` and `S50` are median fractions of
rounds supporting the final mechanism in rounds **71–150** and **101–150**.
All columns after No-write describe Full.

| Model | Game | Setting | n | KL Full | KL No-write | Δ fixed−Full | Match full | Partial | Mismatch | Menu | Write | S80 | S50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Intern | RPS | Available | 12 | 0.023 | 0.021 | -0.011 | 0.417 | 0.417 | 0.167 | 0.233 | 0.217 | 0.54 | 0.39 |
| Intern | RPS | Unavailable | 16 | 0.163 | 0.259 | +0.234 | 0.250 | 0.438 | 0.312 | 0.175 | 0.269 | 0.94 | 1.00 |
| Intern | H&S | Available | 12 | 0.048 | 0.076 | -0.035 | 0.417 | 0.583 | 0.000 | 0.208 | 0.192 | 0.70 | 1.00 |
| Intern | H&S | Unavailable | 16 | 0.197 | 0.331 | +0.048 | 0.062 | 0.688 | 0.250 | 0.195 | 0.176 | 0.35 | 0.56 |
| DeepSeek | RPS | Available | 12 | 0.114 | 0.072 | -0.101 | 0.250 | 0.583 | 0.167 | 0.088 | 0.491 | 0.77 | 0.80 |
| DeepSeek | RPS | Unavailable | 16 | 0.256 | 0.249 | +0.141 | 0.062 | 0.750 | 0.188 | 0.081 | 0.635 | 0.50 | 0.64 |
| DeepSeek | H&S | Available | 12 | 0.063 | 0.051 | -0.051 | 0.083 | 0.833 | 0.083 | 0.077 | 0.581 | 0.57 | 0.76 |
| DeepSeek | H&S | Unavailable | 16 | 0.145 | 0.290 | +0.100 | 0.125 | 0.562 | 0.312 | 0.139 | 0.649 | 0.41 | 0.56 |
| GLM-Flash | RPS | Available | 12 | 0.029 | 0.016 | -0.017 | 0.333 | 0.667 | 0.000 | 0.072 | 0.299 | 1.00 | 1.00 |
| GLM-Flash | RPS | Unavailable | 16 | 0.293 | 0.382 | +0.105 | 0.250 | 0.500 | 0.250 | 0.039 | 0.484 | 0.58 | 0.87 |
| GLM-Flash | H&S | Available | 12 | 0.036 | 0.049 | -0.024 | 0.250 | 0.750 | 0.000 | 0.040 | 0.337 | 1.00 | 1.00 |
| GLM-Flash | H&S | Unavailable | 16 | 0.295 | 0.453 | -0.050 | 0.062 | 0.562 | 0.375 | 0.045 | 0.444 | 0.53 | 0.84 |

Full has lower terminal KL than No-write in **5 of 6 Unavailable** model/game
cells. No-write has lower KL in **4 of 6 Available** cells. The Unavailable
exception is DeepSeek/RPS (0.256 versus 0.249).

Family-level effects below compare **Full with the fixed menu**, not with
No-write. Each entry uses four episodes (two parameter settings × two seeds).
The percentage is `100 × (mean Full KL / mean fixed-menu KL − 1)`.

| Opponent family | Game | Intern | DeepSeek | GLM-Flash |
|---|---|---:|---:|---:|
| Two-move response (`react2`) | RPS | −89.6% | −88.9% | −88.1% |
| Block drift (`block_drift`) | H&S | +23.9% | +2.5% | +17.7% |

Across all 12 Full model/game/coverage cells, final-window KL is below
first-window KL (30-round windows). Median S50 exceeds S80 in nine cells,
is equal in two, and is lower in one. Autonomous decision-stage writing occurs
in 49/56 Intern, 56/56 DeepSeek, and 54/56 GLM-Flash episodes. Scheduled candidate
generation is counted separately.

## Run the paper configurations

Set credentials for your OpenAI-compatible provider in your shell. Model IDs
in the YAML files must be available at that provider:

```bash
export BSI_BASE_URL='https://your-provider.example/v1'
read -rsp 'API key: ' BSI_API_KEY; echo
export BSI_API_KEY
```

The `.env.example` file documents variable names; it is not loaded automatically.
An explicit `BSI_API_KEY_FILE` is also supported. Credentials are never part of
the released configurations.

Run matrices sequentially so that one process owns the configured request
budget. These commands perform real API calls, except for the shared control:

```bash
python scripts/run_matrix.py --config configs/controls.yaml
for model in intern deepseek glm_flash; do
  python scripts/run_matrix.py --config "configs/${model}_full.yaml"
  python scripts/run_matrix.py --config "configs/${model}_no_write.yaml"
done
```

The exact model identities are `Intern-S2-Preview-397B`,
`deepseek-v4.1-flash`, and `glm-5.3-flash`. Intern and DeepSeek use the text
protocol; GLM-Flash uses `json_v1`. Role token budgets and response-repair
settings are preserved in each matrix. Runtime concurrency is set to two
sandbox episodes and four in-flight requests. See
[configuration provenance](docs/PACKAGING.md) for the source mapping.

Repeat the same command to resume an interrupted run. Completed successful
jobs are skipped. Outputs are written under `runs/matrix/<config name>/`.
Changing a configuration requires a new run directory.

## Analyze new results

First compute numerical outcomes; source reviews can be added afterward:

```bash
python scripts/summarize_paper.py \
  --full runs/matrix/intern_full \
  --no-write runs/matrix/intern_no_write \
  --controls runs/matrix/controls \
  --out outputs/intern
```

Repeat for `deepseek` and `glm_flash`. The analysis reads each LLM separately
and rejects mixed-model inputs or unmatched episode keys. It creates:

- `cells.csv`: terminal means, match rates, decision rates, and S80/S50;
- `prediction_windows.csv`: five consecutive 30-round KL means;
- `family_effects.csv`: per-family Full versus fixed-menu effects;
- `paper_metrics.json`: per-episode mechanisms, decision histories, and candidate funnels;
- `summary.md`: an English table of the new results.

To evaluate mechanism recovery and stability, export the dominant generated
sources from the last 80 rounds, review their actual code, and rerun with
`--reviews`. The [review workflow](docs/REPRODUCING.md#5-review-mechanisms-and-stability)
explains the annotation fields. Without complete mechanism labels, the affected
stability aggregate is `NA`; numeric prediction and decision metrics are still
computed. Program names are not used as mechanism labels.

## Release contents

This directory is an independent source tree. It contains no original Git
history, local API settings, paper build, or experiment artifacts. `.gitignore`
excludes locally generated outputs and credential files. The packaging record
lists the retained source modules, configuration adaptations, and validation
steps in [docs/PACKAGING.md](docs/PACKAGING.md).
