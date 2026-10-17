# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import random
import subprocess
import sys

from mmengine.logging import print_log

import xtuner

MODES = ('train',)

CLI_HELP_MSG = f"""
    Arguments received: {str(['xtuner'] + sys.argv[1:])}. Supported syntax:

        xtuner MODE MODE_ARGS

    Supported modes:
        - train

    Common usage:
        xtuner train $CONFIG

    Run special commands:
        xtuner help
        xtuner version
"""

special = {
    'help': lambda: print_log(CLI_HELP_MSG, 'current'),
    'version': lambda: print_log(xtuner.__version__, 'current'),
}
special = {
    **special,
    **{f'-{k[0]}': v for k, v in special.items()},
    **{f'--{k}': v for k, v in special.items()},
}


def train():
    from xtuner.tools import train as train_tool
    return train_tool.__file__


modes = {
    'train': train,
}


def cli():
    args = sys.argv[1:]
    if not args:
        print_log(CLI_HELP_MSG, 'current')
        return

    command = args[0].lower()
    if command in special:
        special[command]()
        return

    if command not in modes:
        print_log('WARNING: command error!', 'current', logging.WARNING)
        print_log(CLI_HELP_MSG, 'current', logging.WARNING)
        raise SystemExit(2)

    try:
        fn = modes[command]
        nnodes = int(os.environ.get('NNODES', 1))
        nproc_per_node = int(os.environ.get('NPROC_PER_NODE', 1))

        if nnodes == 1 and nproc_per_node == 1:
            subprocess.run([sys.executable, fn()] + args[1:], check=True)
            return

        port = os.environ.get('PORT')
        if port is None:
            port = random.randint(20000, 29999)
            print_log(f'Use random port: {port}', 'current', logging.WARNING)

        torchrun_args = [
            f'--nnodes={nnodes}',
            f"--node_rank={os.environ.get('NODE_RANK', 0)}",
            f'--nproc_per_node={nproc_per_node}',
            f"--master_addr={os.environ.get('ADDR', '127.0.0.1')}",
            f'--master_port={port}',
        ]
        subprocess.run(
            ['torchrun'] + torchrun_args + [fn()] + args[1:] + ['--launcher', 'pytorch'],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except Exception as exc:
        print_log(f"WARNING: command error: '{exc}'!", 'current', logging.WARNING)
        print_log(CLI_HELP_MSG, 'current', logging.WARNING)
        raise


if __name__ == '__main__':
    cli()
