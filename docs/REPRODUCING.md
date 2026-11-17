# Reproducing the paper experiments

## 1. Install and inspect

Use Linux and Python 3.12 for the pinned environment. From the project root, install with
`python -m pip install -c requirements-tested.txt -e '.[dev]'` in a
virtual environment. Run `python -m pytest -q` before launching experiments.
All distributed tests are offline.

Every paper matrix uses the same two games, seven opponent families, two
parameter settings, seeds 0 and 1, and 150 rounds. Model-specific Full and
No-write files share the same inference settings. The Full method is
`v6_full_e30`; the No-write method is `v6_noexpand_e30`.

```bash
for config in controls intern_full intern_no_write deepseek_full deepseek_no_write glm_flash_full glm_flash_no_write; do
  python scripts/run_matrix.py --config "configs/${config}.yaml" --dry-run
done
```

Each configuration lists 56 episodes. There are 168 Full episodes, 168 No-write
episodes, and 56 shared fixed-menu episodes in a complete fresh reproduction.
The published results in the README remain the manuscript values, independent
of the new output directories.

## 2. Configure the provider

Export `BSI_API_KEY` and `BSI_BASE_URL` as shown in the README. Use the model IDs
from the corresponding YAML file. The main calls retain reasoning settings;
bounded repair calls have their own token budgets and reasoning overrides.
GLM-Flash's `json_v1` protocol also retains its validation-repair budget.

Each LLM configuration specifies two episode workers, four simultaneous
requests, 40 requests/minute, and 2.5 million tokens/minute. These limits apply
per process. Start with one matrix at a time and adjust runtime limits to your
provider allocation before creating a run directory.

## 3. Generate new episodes

```bash
python scripts/run_matrix.py --config configs/controls.yaml
python scripts/run_matrix.py --config configs/intern_full.yaml
python scripts/run_matrix.py --config configs/intern_no_write.yaml
python scripts/run_matrix.py --config configs/deepseek_full.yaml
python scripts/run_matrix.py --config configs/deepseek_no_write.yaml
python scripts/run_matrix.py --config configs/glm_flash_full.yaml
python scripts/run_matrix.py --config configs/glm_flash_no_write.yaml
```

The runner samples opponent parameters from deterministic configuration keys.
Environment seeds are paired across methods. Exploration opportunities also
share deterministic keys; actions can differ as the learned models diverge.

A run directory contains `config.json`, `summaries.jsonl`, compressed traces,
and progress/error files. Successful summaries identify completed episodes.
Large trace payloads can be stored in overflow files; keep those with the
traces when analyzing your own runs. No original traces or response cache are
required to start.

Resume with the same command. A changed configuration uses a new `--run-dir`.
Use `--max-jobs-per-method 1` with a separate `runs/smoke/...` directory for an
initial check. Only the controls are API-free; a Full or No-write smoke test
makes real requests.

## 4. Compute prediction and behavioral results

```bash
for model in intern deepseek glm_flash; do
  python scripts/summarize_paper.py \
    --full "runs/matrix/${model}_full" \
    --no-write "runs/matrix/${model}_no_write" \
    --controls runs/matrix/controls \
    --out "outputs/${model}"
done
```

The script joins by game, opponent family, parameter index, and seed. It checks
model identity before reading multiple LLM directories. Analyze separate LLMs
in separate invocations; do not pass three model directories to the generic
matrix loader as if they were one cohort.

`cells.csv` follows the paper's game/coverage groups. Terminal KL averages
rounds 137–150 within each episode, then averages episodes. The advantage is
fixed-menu minus Full for paired keys. No-write uses the same terminal window.
Family effects compare the two family means, not the mean of episode-level
percentage changes.

`prediction_windows.csv` averages rounds 1–30, 31–60, 61–90, 91–120, and
121–150. Plot `end` against `mean_kl` for the paper's prediction trajectories.
The JSON retains per-episode decision histories and reviewed dominant states.
`behavior` distinguishes generation, adoption, dominance, code revision,
parameter-only revision, and retirement. Scheduled generation and autonomous
writing have separate candidate funnels.

Menu/Write counts include rejected operations in parseable decisions, but not
unparseable replies. Multiple actions can occur at one checkpoint. Divide the
summed counts by the summed parseable decisions, not by episodes or all rounds.

## 5. Review mechanisms and stability

Export each Full cohort separately. The default 80-round inventory includes
every generated source dominating in rounds 71–150:

```bash
python scripts/analyze_outcomes.py runs/matrix/intern_full \
  --export-review reviews/intern.json --review-tail 80
```

Read each entry's `source` and compare it with the listed opponent rule in
`p2_opponent/bots.py`. Fill in:

- `mechanism`: a consistent identity for the implemented rule;
- `label`: a short readable name;
- `assessments[family].match`: `match`, `partial`, `mismatch`, or `unknown`;
- `assessments[family].reason`: the code-level basis for that classification.

Preserve `source`, `id`, and `uses`. Source content and recorded episode use
bind the annotation. Assign the same mechanism identity to different program
versions implementing the same rule. Parameter-grid changes alone do not
create a new mechanism. A full match recovers the core rule; a partial match
adds or omits dependencies or alters the response; a mismatch uses another
rule.

Rerun the paper summary with the completed review:

```bash
python scripts/summarize_paper.py \
  --full runs/matrix/intern_full \
  --no-write runs/matrix/intern_no_write \
  --controls runs/matrix/controls \
  --reviews reviews/intern.json \
  --out outputs/intern
```

Repeat the export and review for DeepSeek and GLM-Flash. Match proportions use
all episodes, including any `unknown` entries. S80/S50 compare the dominant
mechanism in each round with the terminal mechanism, then take the median
across episodes. Missing identities leave the affected median `NA`; the CSV
reports how many episodes have complete reviews. Predictive accuracy and
mechanism identity remain separate measurements.

## 6. Expected completion evidence

For each LLM, confirm 56 matched episodes, with 12 Available and 16 Unavailable
per game. There should be five prediction windows per method/cell. Inspect the
reviewed source and decision timeline alongside terminal outcomes. Keep the
three model identities separate in tables and plots.

The released README contains the original manuscript aggregates. Fresh LLM
responses produce new programs and trajectories; record the provider, model
identity, configuration, and generated reviews with your new outputs.
