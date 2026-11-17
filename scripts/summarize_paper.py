"""Compute paper metrics from new runs, keeping each base LLM separate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import analyze_outcomes as A
from paper_behavior import behavior_record

FULL = 'v6_full_e30'
NO_WRITE = 'v6_noexpand_e30'
CONTROL = 'menu_all_e30'


def config_key(record):
    return '|'.join(record['job_key'].split('|')[1:])


def index_records(records):
    indexed = {config_key(r): r for r in records}
    if len(indexed) != len(records):
        raise ValueError('Duplicate opponent/parameter/seed keys')
    for r in records:
        if r['horizon'] != 150 or len(r['kl_per_round']) != 150:
            raise ValueError('Paper metrics require all 150 rounds')
    return indexed


def complete_median(values):
    return median(values) if all(v is not None for v in values) else None


def summarize(model, full, controls, no_write=None, reviews=None):
    full, controls = index_records(full), index_records(controls)
    no_write = index_records(no_write) if no_write is not None else None
    if set(full) != set(controls) or (no_write is not None and set(full) != set(no_write)):
        raise ValueError('Full, fixed-menu, and No-write must have matching episode keys')
    episodes = []
    for key, r in full.items():
        ep = A.episode(r, reviews or {})
        if ep['terminal_rounds'] != [137, 150]:
            raise ValueError('Full must retain the round-136 final decision')
        ep['model'] = model
        ep['stability_80'] = A.episode(r, reviews or {}, tail=80)['stability']
        ep['behavior'] = behavior_record(r)
        ep['fixed_terminal_kl'] = mean(controls[key]['kl_per_round'][136:150])
        ep['no_write_terminal_kl'] = mean(no_write[key]['kl_per_round'][136:150]) if no_write is not None else None
        ep['prediction_windows'] = {}
        for name, record in [('full', r), ('fixed_menu', controls[key]),
                             ('no_write', no_write[key] if no_write is not None else None)]:
            if record is not None:
                ep['prediction_windows'][name] = [mean(record['kl_per_round'][i:i+30]) for i in range(0, 150, 30)]
        episodes.append(ep)
    cells, windows, families = [], [], []
    for game in ('rps3', 'hideseek5'):
        for available in (True, False):
            rows = [r for r in episodes if r['game'] == game and r['available'] == available]
            if not rows:
                continue
            usable = sum(r['behavior']['parseable'] for r in rows)
            cell = dict(model=model, game=game, setting='Available' if available else 'Unavailable', n=len(rows),
                        kl_full=mean(r['terminal_kl'] for r in rows),
                        kl_no_write=mean(r['no_write_terminal_kl'] for r in rows) if no_write is not None else None,
                        delta_fixed_minus_full=mean(r['fixed_terminal_kl']-r['terminal_kl'] for r in rows))
            cell.update({k: sum(r['final']['match'] == k for r in rows)/len(rows)
                         for k in ('match', 'partial', 'mismatch', 'unknown')})
            cell.update({k: sum(r['behavior']['counts'][k] for r in rows)/usable if usable else None
                         for k in ('select', 'write')})
            cell.update(s80=complete_median([r['stability_80'] for r in rows]),
                        s50=complete_median([r['stability'] for r in rows]), parseable_decisions=usable,
                        reviewed_s80_episodes=sum(r['stability_80'] is not None for r in rows),
                        reviewed_s50_episodes=sum(r['stability'] is not None for r in rows))
            cells.append(cell)
            for method in rows[0]['prediction_windows']:
                for i in range(5):
                    windows.append(dict(model=model, game=game, setting=cell['setting'], method=method,
                                        start=30*i+1, end=30*(i+1),
                                        mean_kl=mean(r['prediction_windows'][method][i] for r in rows)))
        for family in sorted({r['family'] for r in episodes if r['game'] == game}):
            rows = [r for r in episodes if r['game'] == game and r['family'] == family]
            f, c = mean(r['terminal_kl'] for r in rows), mean(r['fixed_terminal_kl'] for r in rows)
            families.append(dict(model=model, game=game, family=family, n=len(rows), full=f, fixed_menu=c,
                                 relative_change_percent=100*(f/c-1) if c else None))
    return dict(model=model, cells=cells, prediction_windows=windows, family_effects=families, episodes=episodes)


def run_model(directory):
    cfg = json.loads((directory / 'config.json').read_text())['config']
    role = cfg.get('proposer', 'proposer_xl')
    return cfg['llm']['roles'][role]['model']


def write_outputs(result, out):
    out.mkdir(parents=True, exist_ok=True)
    (out/'paper_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for name in ('cells', 'prediction_windows', 'family_effects'):
        rows = result[name]
        with (out/(name+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    cols = ['game', 'setting', 'n', 'kl_full', 'kl_no_write', 'delta_fixed_minus_full',
            'match', 'partial', 'mismatch', 'unknown', 'select', 'write', 's80', 's50']
    def display(v):
        return 'NA' if v is None else f'{v:.3f}' if isinstance(v,float) else str(v)
    lines = ['# '+result['model'], '', '| '+' | '.join(cols)+' |',
             '| '+' | '.join(['---']*len(cols))+' |']
    lines += ['| '+' | '.join(display(r[c]) for c in cols)+' |' for r in result['cells']]
    lines += ['', 'KL and paired differences are episode means. S80/S50 are episode medians.',
              'Review dominant generated sources in rounds 71–150 to populate missing stability values.',
              'Decision rates use parseable checkpoints; structural rates use every episode.', '']
    (out/'summary.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full', type=Path, nargs='+', required=True)
    ap.add_argument('--controls', type=Path, required=True)
    ap.add_argument('--no-write', type=Path, nargs='+')
    ap.add_argument('--reviews', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    models = {run_model(d) for d in args.full + (args.no_write or [])}
    if len(models) != 1:
        ap.error('Analyze one base LLM at a time; input directories contain different model identities')
    result = summarize(next(iter(models)), A.read_records(args.full, FULL),
                       A.read_records([args.controls], CONTROL),
                       A.read_records(args.no_write, NO_WRITE) if args.no_write else None,
                       A.load_reviews(args.reviews))
    write_outputs(result, args.out)
    print(f"{result['model']}: {len(result['episodes'])} matched episodes -> {args.out}")


if __name__ == '__main__':
    main()
