import argparse, json
import datasets
import numpy as np

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--src', type=str, metavar='N',
                    help='source file')
parser.add_argument('--hyp', type=str, metavar='N',
                    help='hypothesis file')
parser.add_argument('--refs', type=str, metavar='N',
                    help='reference file')
parser.add_argument('--outfile', type=str, metavar='N',
                    help='output file')
args = parser.parse_args()

def read_jsonl(infile, extract_key=None):
    f = open(infile, 'r')
    if extract_key is None:
        out = [json.loads(line.strip()) for line in f]
    else:
        out = [json.loads(line.strip())[extract_key] for line in f]
    f.close()
    return out

def score(src, hyp, refs, outfile):
    src = read_jsonl(src, 'src')
    hyp = read_jsonl(hyp, 'hyp')
    refs = read_jsonl(refs, 'refs')
    scores = []
    model = datasets.load_metric("bertscore")
    batch_size = 20
    nb_batches = len(refs)//batch_size
    nb_refs = len(refs[0])
    if len(refs) - nb_batches*batch_size > 0:
        nb_batches += 1
    for i in range(nb_batches):
        scores_batch = [model.compute(
                        predictions=hyp[i*batch_size:(i+1)*batch_size],
                        references=[x[j] for x in refs[i*batch_size:(i+1)*batch_size]],
                        #lang="de")['f1'] for j in range(nb_refs)
                        lang="en")['f1'] for j in range(nb_refs)
                        ]
        scores.extend(list(np.array(scores_batch).max(axis=0)))
        # take max for multi-reference situations
    with open(outfile, 'wt') as fout:
        for score in scores:
            fout.write(str(score))
            fout.write('\n')


if __name__ == '__main__':
    score(args.src, args.hyp, args.refs, args.outfile)
