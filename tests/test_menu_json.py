"""Opt-in JSON envelope: same model operations, state constraints and sandbox contract."""
import json

import pytest

from bsi.llm import LLMResult
from bsi.sandbox import SandboxPool
from p2_opponent.bots import Cycle
from p2_opponent.games import make_game
from p2_opponent.menu_agent import MenuChoiceAgent
from p2_opponent.menu_prompt import (DecisionError, V6Rules, parse_decision, parse_variation,
                                     output_block, PromptState)
from p2_opponent.runner import run_episode
from p2_opponent.seed_programs import SEED_LIBRARY

IDS = {s.id for s in SEED_LIBRARY}
CODE = "def model(history, params, game):\n    # a literal ``` marker stays in the program\n    return np.ones(game['n_opp'])\n"


def reply(key, ops, code=None):
    return json.dumps({key: ops, 'code': code or {}, 'rationale': 'Based on interaction history.'})


def decision(text, **kw):
    return parse_decision(text, 'select_create', IDS, {}, 3, V6Rules(frozenset(), True, 1),
                          output_protocol='json_v1', **kw)


def test_json_select_and_create_have_the_same_operation_meaning():
    ops, _ = decision(reply('actions', [{'op': 'select', 'menu_id': 'reactive'},
                                      {'op': 'create', 'id': 'new_model', 'param_grid': {'a': [.1, .5]}}],
                           {'new_model': CODE}))
    assert [(o.op, o.name) for o in ops] == [('select', 'reactive'), ('create', 'new_model')]
    assert ops[1].code == CODE and ops[1].param_grid == {'a': [.1, .5]}


def test_json_keeps_empty_actions_and_explicit_keep_valid():
    assert decision(reply('actions', []))[0] == []
    assert decision(reply('actions', [{'op': 'keep'}]))[0] == []


@pytest.mark.parametrize('text,problem', [
    ('{"actions": null, "code": {}, "rationale": "r"}', 'root.actions'),
    ('{"op": "keep"}', 'root.actions'),
    ('{"actions": [], "rationale": "r"}', 'root.code'),
    ('{"actions": [], "code": {}, "rationale": []}', 'root.rationale'),
    ('{"actions": [], "code": {"x": []}, "rationale": "r"}', 'root.code'),
    ('prefix {"actions": [], "code": {}, "rationale": "r"}', 'root:'),
    (reply('actions', [{'op': 'create', 'id': 'missing'}]), 'code.missing'),
    (reply('actions', [{'op': 'revise', 'target': 'x', 'new_code': 'false'}]), 'new_code'),
])
def test_json_envelope_errors_are_actionable(text, problem):
    with pytest.raises(DecisionError, match=problem):
        decision(text)


def test_json_does_not_relax_the_existing_active_model_limit():
    with pytest.raises(DecisionError, match='more than the limit 3'):
        decision(reply('actions', [{'op': 'select', 'menu_id': name} for name in sorted(IDS)[:4]]))


def test_json_variation_retains_parent_grid_and_source_checks():
    rules = V6Rules(frozenset({'parent'}), True, 1)
    ops, _ = parse_variation(reply('programs', [{'op': 'mutate', 'parent': 'parent', 'id': 'child'}], {'child': CODE}),
                             IDS, rules, 2, output_protocol='json_v1')
    assert ops[0].parent == 'parent' and ops[0].code == CODE
    with pytest.raises(DecisionError, match='not in the archive'):
        parse_variation(reply('programs', [{'op': 'mutate', 'parent': 'missing', 'id': 'child'}], {'child': CODE}),
                        IDS, rules, 2, output_protocol='json_v1')
    with pytest.raises(DecisionError, match='sandbox rules'):
        decision(reply('actions', [{'op': 'create', 'id': 'bad'}], {'bad': 'import os\n' + CODE}))


def test_only_output_section_changes_in_json_prompt():
    from p2_opponent.menu_prompt import build_messages
    args = (make_game('rps3'), list(SEED_LIBRARY), {}, PromptState(10, 150, 0, 10, [], None, 0), (),
            'select_create', 'full', 40, 3, False)
    text = build_messages(*args)[1]['content']
    js = build_messages(*args, output_protocol='json_v1')[1]['content']
    assert text.split('Output format.')[0] == js.split('Output format (json_v1).')[0]
    assert 'First one JSON object' in output_block('select_create', False)
    assert 'root' in js and '"code"' in js


def test_json_runs_variation_adoption_and_prediction_in_both_games():
    class Fake:
        def __init__(self):
            self.calls = []

        def chat(self, messages, seed_tag='', validate=None, **kw):
            self.calls.append(kw)
            if ':v' in seed_tag:
                name = 'a' if seed_tag.endswith(':v0') else 'b'
                text = reply('programs', [{'op': 'create', 'id': name}], {name: CODE})
            else:
                text = reply('actions', [{'op': 'select', 'menu_id': 'habit'}, {'op': 'activate', 'archive_id': 'a'}])
            result = LLMResult(text, '', 'fake', 10, 10, 'stop', 0)
            assert validate(result) is None
            return result

    with SandboxPool(n_workers=1) as pool:
        for game in ('rps3', 'hideseek5'):
            llm = Fake()
            g = make_game(game)
            agent = MenuChoiceAgent(pool, SEED_LIBRARY, llm, archive=True, heredity=True,
                                    n_variation=2, decision_at=[2], output_protocol='json_v1')
            r = run_episode(g, Cycle(g, [0, 1], .1), agent, 4, seed=0, episode_key=game)
            assert all(c['response_format'] == {'type': 'json_object'} for c in llm.calls)
            assert len([e for e in r.events if e['event'] == 'archive_add' and e['ok']]) == 2
            d = next(e for e in r.events if e['event'] == 'decision')
            assert d['output_protocol'] == 'json_v1' and all(o['ok'] for o in d['ops'])
            assert r.belief[-1]['active'] == ['a', 'habit']
            agent.close()
