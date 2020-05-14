from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np
import time

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--num-parallel-steps', type=int, default=5,
                    help='number of sgd steps done in parallel')
parser.add_argument('--bb-l2-ratio', type=float, default=-1.0,
                    help='Ratio of l2 norm to collect to break the barrier')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.bb_l2_ratio < 0.0:
    import horovod.torch as hvd
else:
    import horovod.torch.broken_barrier_new as hvd


hvd.init()

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

# By default, Adasum doesn't need scaling up learning rate.
lr_scaler = hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
if args.bb_l2_ratio < 0.0:
    optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum if args.use_adasum else hvd.Average)
else:
    optimizer = hvd.BrokenBarrier(model,
        optimizer, named_parameters=model.named_parameters(), num_steps=args.num_warmup_batches + args.num_batches_per_iter * args.num_iters,
        l2_barrier_cond_ratio=args.bb_l2_ratio, num_parallel_steps=args.num_parallel_steps)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()


time_forward = 0
time_backward = 0
time_sync = 0
time_zero = 0
def benchmark_step():
    global time_forward, time_backward, time_sync, time_zero
    # torch.cuda.synchronize()
    s = time.time()
    # zero grad must be called before backward!!!
    optimizer.zero_grad()
    time_zero += time.time() - s
    s = time.time()
    output = model(data)
    time_forward += time.time() - s
    loss = F.cross_entropy(output, target)
    s = time.time()
    loss.backward()
    time_backward += time.time() - s
    s = time.time()
    optimizer.step()
    time_sync += time.time() - s

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
log('Running warmup...')
total_time = timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    step_time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    total_time += step_time
    img_sec = args.batch_size * args.num_batches_per_iter / step_time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log("Total time: {}, forward time: {}, backward time {}, sync time {}, norm time {}. time_zero {}".format(total_time, time_forward, time_backward, time_sync, getattr(optimizer, 'norm_time', 0.0), time_zero))

if hvd.rank() == 0:
    hvd.print_times()