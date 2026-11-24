"""Small token shards in train_gpt2.py's .bin format, for single-device runs (--device mps|cpu).

    python make_local_data.py chars    # Complete Works of Shakespeare, one token per character -> data/shk_chars
    python make_local_data.py bpe      # the same text in GPT-2 BPE (tiktoken)                  -> data/shk_bpe
    python make_local_data.py fineweb PATH/fineweb_val_000000.bin  # split one FineWeb shard  -> data/fineweb_local

The text is ../../data/shakespeare_complete.txt (written by experiments/nanogpt.py --data complete). The last 10%
(5% for FineWeb) is the validation shard. The files are named fineweb_{train,val}_000000.bin, the glob
train_gpt2.py reads under --data-dir. A shard is a 256-int32 header (magic 20240520, version 1, token count)
followed by uint16 tokens.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.path.join(HERE, '..', '..', 'data', 'shakespeare_complete.txt')


def write_shard(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[:3] = 20240520, 1, len(tokens)
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype=np.uint16).tobytes())


def read_shard(path):
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1, path
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == header[2], path
    return tokens


def write(name, tokens, val_frac):
    out = os.path.join(HERE, 'data', name)
    os.makedirs(out, exist_ok=True)
    n = int((1 - val_frac) * len(tokens))
    write_shard(os.path.join(out, 'fineweb_train_000000.bin'), tokens[:n])
    write_shard(os.path.join(out, 'fineweb_val_000000.bin'), tokens[n:])
    print(f'{out}: {n:,} train / {len(tokens) - n:,} val tokens, max token id {int(np.max(tokens))}')


if __name__ == '__main__':
    kind = sys.argv[1]
    if kind == 'chars':
        text = open(TEXT, encoding='utf-8').read()
        chars = sorted(set(text))
        lookup = {c: i for i, c in enumerate(chars)}
        write('shk_chars', np.array([lookup[c] for c in text]), 0.1)
    elif kind == 'bpe':
        import tiktoken
        write('shk_bpe', np.array(tiktoken.get_encoding('gpt2').encode_ordinary(open(TEXT, encoding='utf-8').read())), 0.1)
    elif kind == 'fineweb':
        write('fineweb_local', read_shard(sys.argv[2]), 0.05)
    else:
        raise ValueError(kind)
