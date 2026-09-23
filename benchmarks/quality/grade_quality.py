"""Grade frozen labels, not teacher agreement. Raw predictions stay untouched."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def score_answer(answer, target):
    kind, label = target['type'], target['label']
    assert answer['type'] == kind
    if kind == 'noul':
        p = float(answer['noul'])
        assert math.isfinite(p) and 0 <= p <= 1
        pred = int(p >= 0.5)
        return pred, int(pred == label), (p - label) ** 2, max(p, 1-p), None
    probs = answer['probabilities']
    vals = np.array(list(probs.values()), dtype=float)
    assert len(vals) >= 2 and np.isfinite(vals).all()
    assert np.all((vals >= 0) & (vals <= 1)) and abs(vals.sum()-1) <= 0.005
    key = max(probs, key=probs.get)
    pred = key if kind == 'choice' else int(key)
    if kind == 'choice':
        assert answer['choice'] == key
        assert label in probs
    else:
        assert set(probs) == {str(i) for i in range(5)}
    brier = sum((float(v)-int(k == str(label)))**2 for k,v in probs.items())
    mae = None
    if kind == 'score':
        score = float(answer['score'])
        assert math.isfinite(score) and 0 <= score <= 4
        expected_score = sum(int(k)*float(v) for k,v in probs.items())
        assert abs(score - expected_score) <= 0.02
        mae = abs(score-label)
    return pred, int(pred == label), brier, float(max(vals)), mae


def summarize(rows):
    assert rows
    valid = [r for r in rows if r['error'] is None]
    bins = []
    for bucket in range(10):
        group = [r for r in valid if min(9, int(r['confidence'] * 10)) == bucket]
        if group:
            bins.append(len(group)/max(1,len(valid))*abs(np.mean([r['correct'] for r in group])-np.mean([r['confidence'] for r in group])))
    def mean_field(rs, field):
        vs = [r[field] for r in rs if r[field] is not None]
        return float(np.mean(vs)) if vs else None
    return {'decisions':len(rows), 'valid':len(valid), 'errors':len(rows)-len(valid),
            'correct':sum(r['correct'] for r in rows),
            'accuracy_errors_as_wrong':mean_field(rows,'correct'),
            'accuracy_parsed_only':mean_field(valid,'correct'),
            'binary_brier':mean_field([r for r in valid if r['type']=='noul'],'brier'),
            'multiclass_brier':mean_field([r for r in valid if r['type']!='noul'],'brier'),
            'score_mae':mean_field(valid,'score_mae'),
            'confidence_ece_10bin_directional':float(sum(bins)) if valid else None,
            'confident_wrong_90':sum(r['correct']==0 and r['confidence']>=0.9 for r in valid)}


def grade(path, expected, protocol):
    raw = json.loads(Path(path).read_text())
    results = raw['results']
    assert len(results) == len(expected), (len(results),len(expected))
    by_id = {r['id']:r for r in results}
    assert len(by_id)==len(results) and set(by_id)==set(expected)
    assert raw['suite_sha256']==protocol['suite_sha256']
    rows=[]
    for tid, task in expected.items():
        result=by_id[tid]
        assert result['family']==task['family']
        output=result['output']
        answers=output.get('answers',{}) if isinstance(output,dict) else {}
        for qid,target in task['questions'].items():
            key=f'{tid}/{qid}'
            row={'key':key,'id':tid,'question':qid,'family':task['family'],'split':task['split'],
                 'type':target['type'],'target':target['label'],'excluded':key in protocol['exclusions'],
                 'prediction':None,'correct':0,'brier':None,'confidence':None,'score_mae':None,'error':None}
            try:
                pred,correct,brier,conf,mae=score_answer(answers[qid],target)
                row.update(prediction=pred,correct=correct,brier=brier,confidence=conf,score_mae=mae)
            except (KeyError,TypeError,ValueError,AssertionError) as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            rows.append(row)
    assert len(rows)==protocol['expected_decisions']
    clean=[r for r in rows if not r['excluded']]
    families={f:summarize([r for r in clean if r['family']==f]) for f in sorted({r['family'] for r in clean})}
    summary={'model':raw['model'],'source':str(path),'original_labels':summarize(rows),
             'predeclared_clean':summarize(clean),'families':families,
             'macro_family_accuracy':float(np.mean([v['accuracy_errors_as_wrong'] for v in families.values()])),
             'prior_test_split':summarize([r for r in clean if r['split']=='test']),
             'usage':raw.get('usage'),'calibration_probe':raw.get('calibration_probe')}
    return rows,summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('results',nargs='+'); p.add_argument('--output',default=str(ROOT.parent/'results'/'quality_summary.json'))
    args=p.parse_args()
    expected=json.loads((ROOT/'quality_expected.json').read_text())
    protocol=json.loads((ROOT/'quality_protocol.json').read_text())
    assert expected and len(expected)==protocol['expected_cases']
    all_rows=[]; summaries={}; per_arm={}
    for path in args.results:
        rows,summary=grade(path,expected,protocol)
        name=Path(path).stem
        assert name not in summaries
        summaries[name]=summary; per_arm[name]=rows
        all_rows.extend(dict(arm=name,**r) for r in rows)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    paired={}
    if len(per_arm)==2:
        names=list(per_arm); a,b=(per_arm[n] for n in names)
        assert [r['key'] for r in a]==[r['key'] for r in b]
        raw_a, raw_b = (json.loads(Path(p).read_text()) for p in args.results)
        requests_a = {r['id']: {k: r['request'][k] for k in ('state', 'questions')} for r in raw_a['results']}
        requests_b = {r['id']: {k: r['request'][k] for k in ('state', 'questions')} for r in raw_b['results']}
        assert requests_a == requests_b, 'Arms did not receive identical states and questions'
        pairs=[(x,y) for x,y in zip(a,b) if not x['excluded']]
        deltas=defaultdict(list)
        for x,y in pairs: deltas[x['id']].append(y['correct']-x['correct'])
        groups=list(deltas.values()); rng=np.random.default_rng(20260920)
        boot=[]
        for _ in range(5000):
            sample=[groups[i] for i in rng.integers(len(groups),size=len(groups))]
            boot.append(sum(sum(g) for g in sample)/sum(len(g) for g in sample))
        paired={'difference':f'{names[1]} minus {names[0]}','accuracy_difference':float(np.mean([y['correct']-x['correct'] for x,y in pairs])),
                'case_cluster_bootstrap_95_ci':np.quantile(boot,[0.025,0.975]).tolist(),
                'first_only_correct':sum(x['correct'] and not y['correct'] for x,y in pairs),
                'second_only_correct':sum(y['correct'] and not x['correct'] for x,y in pairs),
                'both_correct':sum(x['correct'] and y['correct'] for x,y in pairs),
                'neither_correct':sum(not x['correct'] and not y['correct'] for x,y in pairs)}
    out.write_text(json.dumps({'protocol':protocol,'arms':summaries,'paired':paired},indent=2))
    with out.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader();writer.writerows(all_rows)
    print(json.dumps({'arms':{k:{'model':v['model'],'overall':v['predeclared_clean'],'families':v['families']} for k,v in summaries.items()},'paired':paired},indent=2))


if __name__=='__main__':
    main()
