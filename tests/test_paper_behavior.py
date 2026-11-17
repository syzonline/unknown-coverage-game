import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import paper_behavior as P

def candidate(name='a', t=1, src='code', call='variation:0', op='create', grid=None):
    return dict(event='archive_add', name=name, t=t, src=src, param_grid=grid or {},
                call=call, op=op, ok=True, parent=None)


def add(pid='a#1', t=1, src='code', source='archived', grid=None):
    return dict(event='add', id=pid, t=t, src=src, param_grid=grid or {}, source=source, ok=True)


def decision(t=1, ops=(), **kwargs):
    return dict(event='decision', t=t, ops=list(ops), **kwargs)


def belief(pid=None, weight=0, other=1):
    return dict(map=pid, map_w=weight, other=other, n=int(pid is not None))


def run(events, beliefs):
    return P.behavior_record(dict(job_key='test', events=events, belief=beliefs))


def test_generation_adoption_and_support_are_distinct():
    events = [candidate(), candidate('unused'), add(),
              decision(ops=[dict(op='activate', name='a', pid='a#1', ok=True)])]
    # Named MAP loses to reference: not dominant in the first predictive round.
    r = run(events, [belief(), belief('a#1', .4, .6), belief('a#1', .9, .1)])
    assert r['funnel']['variation'] == dict(accepted=2, adopted=1, dominant=1)
    assert r['versions'][0]['dominant_rounds'] == [3]
    assert r['counts']['activate'] == 1
    assert r['counts']['write'] == 0


def test_reused_name_keeps_distinct_archive_versions():
    events = [candidate(), add(), decision(ops=[dict(op='activate', name='a', pid='a#1', ok=True)]),
              candidate(t=2), add('a#2', t=2),
              decision(2, [dict(op='activate', name='a', pid='a#2', ok=True)])]
    r = run(events, [belief(), belief('a#1', .9, .1), belief('a#2', .9, .1)])
    assert r['funnel']['variation'] == dict(accepted=2, adopted=2, dominant=2)
    assert r['versions'][0]['active_ids'] == ['a#1']
    assert r['versions'][1]['active_ids'] == ['a#2']


def test_code_revision_is_archived_after_active_add_but_grid_change_is_not_new_code():
    events = [candidate(call='decision'), add(source='created'),
              decision(ops=[dict(op='create', name='a', pid='a#1', ok=True)]),
              add('a#2', t=2, src='revision', source='revised'),
              candidate('a_r1', t=2, src='revision', call='decision', op='revise'),
              decision(2, [dict(op='revise', name='a', pid='a#2', ok=True, new_code=True)]),
              add('a#3', t=3, src='revision', source='revised', grid={'x': [1, 2]}),
              decision(3, [dict(op='revise', name='a', pid='a#3', ok=True, new_code=False)])]
    r = run(events, [belief(), belief('a#1', .8, .2), belief('a#2', .8, .2), belief('a#3', .8, .2)])
    assert r['funnel']['decision'] == dict(accepted=2, adopted=2, dominant=2)
    assert r['counts']['write'] == 2
    assert r['counts']['code_revision'] == 1
    assert r['counts']['parameter_only'] == 1
    assert r['parameter_only_ids'] == ['a#3']


def test_failed_decision_is_not_noop_and_rejected_writing_still_counts():
    rejected = {**candidate(call='decision'), 'ok': False}
    events = [rejected, decision(ops=[dict(op='create', name='a', ok=False)]),
              decision(2, error='unusable reply'), decision(3)]
    r = run(events, [belief()] * 3)
    assert r['recorded'] == 3 and r['parseable'] == 2
    assert r['counts']['write'] == 1
    assert r['funnel']['decision'] == dict(accepted=0, adopted=0, dominant=0)


def test_multiple_actions_and_repairs_count_once_per_checkpoint():
    events = [decision(ops=[dict(op='create', name='a', ok=False),
                           dict(op='create', name='a', ok=False, repair=True),
                           dict(op='retire', name='old', ok=True)])]
    r = run(events, [belief()])
    assert r['counts']['write'] == r['counts']['retire'] == 1
    assert r['parseable'] == 1
