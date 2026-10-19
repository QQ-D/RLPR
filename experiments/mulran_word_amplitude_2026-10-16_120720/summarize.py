"""Verify completed evidence and export paired amplitude results."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = PROJECT / 'src_spectral'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

r = json.loads((ROOT / 'run.json').read_text())
assert r['status'] == 'COMPLETE', 'Training and benchmarks are still pending'
assert r['paired_sampling_streams_match']
assert digest(ROOT / 'protocol.json') == r['protocol_sha256']
assert digest(ROOT / 'selection.json') == r['selection_sha256']
p = json.loads((ROOT / 'protocol.json').read_text())
assert all(digest(SOURCE / n) == h for n, h in p['source_sha256'].items())
assert all(digest(v['path']) == v['sha256'] for v in p['initialization'].values())
official = json.loads((PROJECT / 'runs/mulran_official_eval_2026-10-14/results.json').read_text())
previous = json.loads((PROJECT / 'runs/mulran_word_hardneg_2026-10-16_003626/run.json').read_text())
rows = []
for split in ('dcc', 'riverside'):
    rows.append(dict(dataset=split, method='paper', **official['splits'][split]['paper_metrics']))
    rows.append(dict(dataset=split, method='released_official_model', **official['splits'][split]['metrics']))
    epoch = previous['selection']['selected_epoch']
    rows.append(dict(dataset=split, method='previous_validation_selected_hardneg', cross_epoch=epoch,
                     **previous['epochs'][str(epoch)]['benchmarks'][split]['metrics']))
    for arm in ('pure_phase', 'amplitude'):
        a = r['arms'][arm]
        epoch = a['cross']['selected_epoch']
        selected = a['cross']['epochs'][str(epoch)]
        assert digest(selected['checkpoint']) == selected['checkpoint_sha256']
        rows.append(dict(dataset=split, method=arm, cross_epoch=epoch,
                         lidar_additional_epoch=a['l']['selected_epoch'],
                         radar_additional_epoch=a['r']['selected_epoch'],
                         **a['cross']['benchmarks'][split]))
fields = ['dataset', 'method', 'cross_epoch', 'lidar_additional_epoch', 'radar_additional_epoch',
          'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent', 'F1']
with (ROOT / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(rows)
curves = []
for arm, stages in r['arms'].items():
    for stage, s in stages.items():
        for epoch, e in s['epochs'].items():
            curves.append(dict(arm=arm, stage=stage, epoch=int(epoch), loss=e['loss'],
                               selected=int(epoch) == s['selected_epoch'], **e['validation']))
with (ROOT / 'curves.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['arm', 'stage', 'epoch', 'loss', 'selected',
                                         'Recall@1', 'Recall@5', 'Recall@10', 'Recall@20', 'Recall@1percent', 'F1'])
    writer.writeheader();writer.writerows(curves)
deltas = {split: {metric: r['arms']['amplitude']['cross']['benchmarks'][split][metric]
                             - r['arms']['pure_phase']['cross']['benchmarks'][split][metric]
                 for metric in ('Recall@1', 'Recall@5', 'Recall@10', 'F1')}
          for split in ('dcc', 'riverside')}
summary = dict(status='COMPLETE', run_root=str(ROOT), protocol_sha256=r['protocol_sha256'],
               evidence_verified=True, comparison=rows, amplitude_minus_matched_pure_phase=deltas,
               caveat='Single seed; exploratory benchmark comparison; teacher selection and cross selection use Sejong only',
               training='Each arm model-only warmstarts historical L5/R5, fresh Adam/scheduler, trains each unary branch three additional epochs and cross stage two epochs; strict seed0')
(ROOT / 'results_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
