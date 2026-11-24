"""Download GPT-2-tokenized FineWeb10B shards (100M tokens each) from kjj0/fineweb10B-gpt2.

Same source as modded-nanogpt's data/cached_fineweb10B.py. The 5,100-step baseline
(512 x 1024 tokens/step = 2.67B tokens) needs 27 train shards; 28 leaves a margin.
"""
import argparse
import os

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--train-shards', type=int, default=28)
parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fineweb10B'))
args = parser.parse_args()


def get(fname):
    if not os.path.exists(os.path.join(args.data_dir, fname)):
        hf_hub_download(repo_id='kjj0/fineweb10B-gpt2', filename=fname,
                        repo_type='dataset', local_dir=args.data_dir)


get('fineweb_val_%06d.bin' % 0)
for i in range(1, args.train_shards + 1):
    get('fineweb_train_%06d.bin' % i)
print(f'done: {args.data_dir}')
