import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
from torchvision import models
import timeit
import numpy as np


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
parser.add_argument('--local_rank', default=-1, type=int, help="Using apex")
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training')
parser.add_argument('--use-apex', action='store_true', default=False,
                    help='use apex')
parser.add_argument('--use-hvd', action='store_true', default=False,
                    help='use apex')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.use_hvd:
    import horovod.torch as hvd
    hvd.init()
else:
    from apex import amp
    dist.init_process_group(backend='nccl', init_method='env://')

if args.use_apex:
    from apex.parallel import DistributedDataParallel

if args.use_amp:
    from apex import amp

if not args.use_hvd:
    rank = args.local_rank
    world_size = dist.get_world_size()
else:
    rank = hvd.local_rank()
    world_size = hvd.size()

if args.cuda:
    # pin GPU to local rank.
    torch.cuda.set_device(rank)

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()

# By default, Adasum doesn't need scaling up learning rate.
# lr_scaler = world_size if not args.use_adasum else 1
#
if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    # if args.use_adasum and hvd.nccl_built():
    #     lr_scaler = hvd.local_size()

optimizer = optim.SGD(model.parameters(), lr=0.01)


# Horovod: wrap optimizer with DistributedOptimizer.
if args.use_hvd and not args.use_apex:
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # compression = hvd.Compression.fp32
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

if args.use_amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

if args.use_apex:
    if args.use_hvd:
        model = DistributedDataParallel(model, hvd_dist=True, allreduce_always_fp32=True)
    else:
        model = DistributedDataParallel(model, num_allreduce_streams=4, fake_comp_ratio=1.0)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()


def benchmark_step():
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    if args.use_apex or (args.use_amp and not args.use_hvd):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
    else:
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                optimizer.synchronize()
            with optimizer.skip_synchronize():
                optimizer.step()
        else:
            loss.backward()
            optimizer.step()


def log(s, nl=True):
    if rank != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, world_size))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

# Benchmark
log('Running benchmark...')
img_secs = []
for x in range(args.num_iters):
    time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)

# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (world_size, device, world_size * img_sec_mean, world_size * img_sec_conf))
