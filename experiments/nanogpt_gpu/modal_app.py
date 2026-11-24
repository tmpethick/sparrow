"""Run train_gpt2.py on Modal (image and FineWeb volume as in AccSODA's modal_train_gpt_moda.py).

    modal run modal_app.py --action smoke [--optimizers aurora]  # 10 steps each of Muon and Sparrow, 1 H100
    modal run -d modal_app.py --train-args "--seed 0 --run-name muon-s0"
    modal run -d modal_app.py --train-args "--optimizer sparrow --seed 0 --run-name sparrow-s0"

Training uses 4 H100s (global batch 512 via two accumulation steps) and is killed
after TRAIN_TIMEOUT_MINUTES; a run takes about 33 min. Functions are not retried.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = '/root/project'
DATA_VOLUME = modal.Volume.from_name(os.environ.get('MODAL_DATA_VOLUME_NAME', 'accmoda-fineweb10b'))
LOGS_VOLUME = modal.Volume.from_name('halpern-tbptt-nanogpt-logs', create_if_missing=True)
TRAIN_GPU = os.environ.get('MODAL_GPU', 'H100:4')
TRAIN_TIMEOUT_MINUTES = int(os.environ.get('MODAL_TRAIN_TIMEOUT_MINUTES', '60'))

image = (
    modal.Image.debian_slim(python_version='3.11')
    .run_commands(
        'python -m pip install --upgrade pip',
        'python -m pip install torch --index-url https://download.pytorch.org/whl/cu124',
    )
    .pip_install('numpy', 'wandb')
    .workdir(REMOTE)
    .add_local_file(HERE.parent.parent / 'sparrow_torch.py', f'{REMOTE}/sparrow_torch.py')
    .add_local_dir(HERE, REMOTE, ignore=['**/__pycache__', 'data/**', 'logs/**', 'wandb/**', 'slurm-*.out'])
)
app = modal.App('halpern-tbptt-nanogpt')


def torchrun(nproc, train_args):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={nproc}',
               'train_gpt2.py', *shlex.split(train_args)]
    print('$', shlex.join(command), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    subprocess.run(command, cwd=REMOTE, env=env, check=True)


@app.function(image=image, gpu='H100', timeout=20 * 60, retries=0,
              volumes={f'{REMOTE}/data/fineweb10B': DATA_VOLUME})
def smoke(optimizers: str = 'muon,sparrow', extra: str = ''):
    for o in optimizers.split(','):
        torchrun(1, f'--num-iterations 10 --val-loss-every 0 --no-wandb --seed 0 --optimizer {o} --run-name smoke-{o} {extra}')


@app.function(image=image, gpu=TRAIN_GPU, timeout=TRAIN_TIMEOUT_MINUTES * 60, retries=0,
              volumes={f'{REMOTE}/data/fineweb10B': DATA_VOLUME, f'{REMOTE}/logs': LOGS_VOLUME},
              secrets=[modal.Secret.from_name('wandb-secret')])
def train(train_args: str):
    import torch
    try:
        torchrun(torch.cuda.device_count(), train_args)
    finally:
        LOGS_VOLUME.commit()


@app.local_entrypoint()
def main(action: str = 'train', train_args: str = '', optimizers: str = 'muon,sparrow'):
    if action == 'smoke':
        smoke.remote(optimizers, train_args)  # train_args: extra flags for the smoke runs
    elif action == 'train':
        train.remote(train_args)
    else:
        raise ValueError(action)
