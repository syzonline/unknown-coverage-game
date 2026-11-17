"""Source-only release contracts and paper aggregation with synthetic records."""
import copy
import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_matrix as R
import summarize_paper as P
from test_outcomes import record


def test_matched_release_configs():
    expected = ['wsls','fictitious','reactive','lose_k_shift','react2','seq_br','block_drift']
    models = ['Intern-S2-Preview-397B','deepseek-v4.1-flash','glm-5.3-flash']
    for label,model in zip(['intern','deepseek','glm_flash'],models):
        full = yaml.safe_load((ROOT/'configs'/f'{label}_full.yaml').read_text())
        nw = yaml.safe_load((ROOT/'configs'/f'{label}_no_write.yaml').read_text())
        assert full['families'] == nw['families'] == expected
        assert full['llm'] == nw['llm']
        assert full['llm']['roles']['proposer_xl']['model'] == model
        a,b = full['methods'][P.FULL],nw['methods'][P.NO_WRITE]
        assert a['n_variation']==2 and b['n_variation']==0 and b['menu_mode']=='select_only'
        assert {k:v for k,v in a.items() if k!='n_variation'} == {k:v for k,v in b.items() if k not in ('n_variation','menu_mode')}
        assert a.get('output_protocol','text') == ('json_v1' if label=='glm_flash' else 'text')
        for cfg in [full,nw]:
            assert R.config_problems(cfg,list(cfg['methods'])) == []
            assert len(R.build_jobs(cfg,list(cfg['methods']))) == 56
    ctl = yaml.safe_load((ROOT/'configs/controls.yaml').read_text())
    assert list(ctl['methods']) == [P.CONTROL]
    assert len(R.build_jobs(ctl,list(ctl['methods'])))==56


def test_paper_summary_and_outputs(tmp_path):
    full = record(); full['events'][-1]['ops']=[{'op':'select','name':'reactive','ok':True}]
    ctl=copy.deepcopy(full); ctl['job_key']=ctl['job_key'].replace(P.FULL,P.CONTROL)
    ctl['kl_per_round']=[.4]*150
    nw=copy.deepcopy(full); nw['job_key']=nw['job_key'].replace(P.FULL,P.NO_WRITE)
    nw['kl_per_round']=[.3]*150
    result=P.summarize('test-model',[full],[ctl],[nw])
    c=result['cells'][0]
    assert c['kl_full']==pytest.approx(.2) and c['kl_no_write']==pytest.approx(.3)
    assert c['delta_fixed_minus_full']==pytest.approx(.2)
    assert c['s80']==c['s50']==1 and c['select']==.5 and c['write']==0
    assert c['match']==1 and c['unknown']==0
    assert result['family_effects'][0]['relative_change_percent']==pytest.approx(-50)
    assert len(result['prediction_windows'])==15
    P.write_outputs(result,tmp_path)
    assert set(p.name for p in tmp_path.iterdir())=={'paper_metrics.json','cells.csv','prediction_windows.csv','family_effects.csv','summary.md'}
    assert json.loads((tmp_path/'paper_metrics.json').read_text())['model']=='test-model'
    with pytest.raises(ValueError,match='matching episode keys'):
        P.summarize('test-model',[full],[],[nw])
    assert P.complete_median([None,1]) is None


def test_source_review_roundtrip_without_bundled_records(tmp_path):
    import analyze_outcomes as A
    r=record(); r['events'][0].update(source='created',src='def model(history, params, game):\n    return np.ones(3) / 3\n')
    for b in r['belief']: b['map_src']='created'; b['by_src']={'created':.9}
    path=tmp_path/'review.json'; A.export_review([r],path,80)
    data=json.loads(path.read_text()); entry=data['entries'][0]
    entry.update(mechanism='uniform',label='Uniform actions')
    entry['assessments']['reactive']={'match':'mismatch','reason':'No last-agent-action conditioning'}
    path.write_text(json.dumps(data))
    reviews=A.load_reviews(path)
    assert A.episode(r,reviews,tail=80)['stability']==1
    assert A.episode(r,reviews)['final']['match']=='mismatch'
