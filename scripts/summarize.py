"""Summarize newly evaluated LRR outputs without changing the paper's frozen CSVs."""
from _common import ROOT
import argparse,csv,json,statistics
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outputs',type=Path,default=ROOT/'outputs')
    p.add_argument('--destination',type=Path,default=ROOT/'outputs/summary.csv');a=p.parse_args()
    groups={}
    for c in (ROOT/'configs/main').glob('*.json'):
        config=json.loads(c.read_text());result=a.outputs/(c.stem+'.pt')
        if not result.exists():continue
        payload=torch.load(result,map_location='cpu',weights_only=False)
        if 'test_metrics' not in payload:raise ValueError(f'no final test metrics: {result}')
        key=tuple(str(config[k]) for k in ['dataset','backbone','deletion_mode','deletion_ratio'])
        groups.setdefault(key,[]).append((config['seed'],payload['test_metrics']))
    rows=[]
    for key,values in sorted(groups.items()):
        row=dict(zip(['dataset','backbone','location','ratio'],key));row.update(n_seeds=len(values),seeds=';'.join(str(x[0]) for x in sorted(values)))
        for metric in ['DT_AUC','DT_AUP','Df_AUC','Df_AUP','J_AUC','J_AUP']:
            scores=[float(x[1][metric]) for x in values]
            row[metric+'_mean']=statistics.mean(scores);row[metric+'_sample_sd']=statistics.stdev(scores) if len(scores)>1 else ''
        rows.append(row)
    if not rows:raise ValueError('no evaluated outputs')
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    with a.destination.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(a.destination)


if __name__=='__main__':main()
