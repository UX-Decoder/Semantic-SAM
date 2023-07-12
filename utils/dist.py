import functools
import io
import os
import random 
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import json, time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

import colorsys
def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and 
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])

        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        # args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))        
        # local_world_size = int(os.environ['GPU_PER_NODE_COUNT'])
        # args.world_size = args.world_size * local_world_size
        # args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        # args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        if os.environ.get('HAND_DEFINE_DIST_URL', 0) == '1':
            pass
        else:
            import util.hostlist as uh
            nodenames = uh.parse_nodelist(os.environ['SLURM_JOB_NODELIST'])
            gpu_ids = [int(node[3:]) for node in nodenames]
            fixid = int(os.environ.get('FIX_DISTRIBUTED_PORT_NUMBER', 0))
            # fixid += random.randint(0, 300)
            port = str(3137 + int(min(gpu_ids)) + fixid)
            args.dist_url = "tcp://{ip}:{port}".format(ip=uh.nodename_to_ip(nodenames[0]), port=port)

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))


    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        world_size=args.world_size, 
        rank=args.rank,
        init_method=args.dist_url,
    )

    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")