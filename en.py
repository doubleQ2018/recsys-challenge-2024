import shutil
import tempfile
import pandas as pd
import polars as pl
from argparse import ArgumentParser
from datetime import datetime

version = datetime.now().strftime("%m%d%H%M")

def grank(x):
    scores = x["score"].tolist()
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return rank

def grank(scores):
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return rank

def gen_submit(df):
    df = ( df
            .group_by(["impression_id", "user_id"], maintain_order=True)
            .agg(
                pl.col("score").apply(grank).alias("rank")
            )
        )
    return df

parser = ArgumentParser()
parser.add_argument("--files", nargs="+", default=[])
parser.add_argument("--weights", nargs="+", default=[])
parser.add_argument("--output", type=str, default='ensemble')
parser.add_argument('--norm', action='store_true')
args = parser.parse_args()
args.weights = [float(w) for w in args.weights]
print(args.files)
if not args.weights:
    args.weights = [1.0/len(args.files)] * len(args.files)
print(args.weights)
ans = None
for w, f in zip(args.weights, args.files):
    df = pd.read_csv(f)
    print(f)
    if args.norm and (df['score'].min() < 0 or df['score'].max() > 1):
        df['max_score'] = df.groupby(['impression_id', 'user_id'])['score'].transform(max)
        df['min_score'] = df.groupby(['impression_id', 'user_id'])['score'].transform(min)
        df['score'] = (df['score'] - df['min_score']) / (df['max_score'] - df['min_score'])
        df = df.drop(columns=['max_score', 'min_score'])
    print(df)
    if ans is None:
        ans = df
        ans['score'] = w * ans['score']
    else:
        ans['score'] = ans['score'] + w * df['score']
print(ans)
ans = gen_submit(pl.from_pandas(ans))
with tempfile.TemporaryDirectory() as tmpdir:
    with open(f'{tmpdir}/predictions.txt', "w") as fout:
        out = [str(row['impression_id']) + " " + row['rank'] for row in ans.iter_rows(named=True)]
        fout.write("\n".join(out))
    shutil.make_archive(f'result/{args.output}', 'zip', tmpdir, 'predictions.txt')
