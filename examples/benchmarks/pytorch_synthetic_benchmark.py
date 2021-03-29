import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torch.distributed as dist
from torchvision import models
import timeit
import numpy as np
from dataloaders import get_dali_train_loader
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
parser.add_argument('--use-apex-ddp', action='store_true', default=False,
                    help='use apex')
parser.add_argument('--use-hvd', action='store_true', default=False,
                    help='use apex')
parser.add_argument('--use-torch-ddp', action='store_true', default=False,
                    help='use torch')

parser.add_argument("--data-loader", type=str, choices=["synthetic", "dali-cpu"], default="dali-cpu")
parser.add_argument('--workers', type=int, default=4,
                    help='number of dataloader workers')
parser.add_argument('--dataset-dir', default='imagenet',
                    help='path to training data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.use_hvd ^ args.use_apex_ddp

if args.use_hvd:
    import horovod.torch as hvd
    hvd.init()
else:
    dist.init_process_group(backend='nccl', init_method='env://')

if args.use_apex_ddp:
    from apex.parallel import DistributedDataParallel
elif args.use_torch_ddp:
    from torch.nn.parallel import DistributedDataParallel

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

optimizer = optim.SGD(model.parameters(), lr=0.01)


# Horovod: wrap optimizer with DistributedOptimizer.
if args.use_hvd:
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

if args.use_amp:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", verbosity=0)

if args.use_apex_ddp:
    model = DistributedDataParallel(model)
elif args.use_torch_ddp:
    model = DistributedDataParallel(model, device_ids=[rank])

if args.data_loader == "dali-cpu":
    get_train_loader = get_dali_train_loader(True)
    train_loader, train_size = get_train_loader(args.dataset_dir, args.batch_size, 1000, one_hot=False,
                                                workers=args.workers)
    train_iter = enumerate(train_loader)
else:
    # Set up fixed fake data
    data_ = torch.randn(args.batch_size, 3, 224, 224)
    target_ = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data_, target_ = data_.cuda(), target_.cuda()

def get_data():
    if args.data_loader == "dali-cpu":
        i, (data, target) = next(train_iter)
        return data, target
    else:
        return data_, target_

time_data = 0
time_forward = 0
time_sync = 0
time_backward = 0


def benchmark_step_time():
    global time_data, time_forward, time_sync, time_backward
    optimizer.zero_grad()
    start = time.time()
    data, target = get_data()
    torch.cuda.synchronize()
    time_data += time.time() - start
    start = time.time()
    output = model(data)
    loss = F.cross_entropy(output, target)
    torch.cuda.synchronize()
    time_forward += time.time() - start
    start = time.time()
    if args.use_amp and not args.use_hvd:
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
    torch.cuda.synchronize()
    time_backward += time.time() - start

def benchmark_step():
    optimizer.zero_grad()
    data, target = get_data()
    output = model(data)
    loss = F.cross_entropy(output, target)
    if args.use_amp and not args.use_hvd:
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


log('Model: %s, size: %d' % (args.model, count_parameters(model)))
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, world_size))

# Warm-up
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)

time_data = 0
time_forward = 0
time_sync = 0
time_backward = 0
# Benchmark
log('Running benchmark...')
img_secs = []
total_time = time.time()
for x in range(args.num_iters):
    time_step = timeit.timeit(benchmark_step_time, number=args.num_batches_per_iter)
    img_sec = args.batch_size * args.num_batches_per_iter / time_step
    log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
    img_secs.append(img_sec)
total_time = time.time() - total_time
log("Total time %f"%(total_time))
# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (world_size, device, world_size * img_sec_mean, world_size * img_sec_conf))
# f = lambda x:  x * 100 / total_time
f = lambda x:  x * 1000 / (args.num_batches_per_iter * args.num_iters)
# log("Data time %.2f %%, forward time %.2f %%, sync time %.2f %%, backward time %.2f%%"%
#     tuple(map(f, [time_data, time_forward, time_sync, time_backward])))
log("Data time %f, forward time %f, sync time %f , backward time %f"%
    tuple(map(f, [time_data, time_forward, time_sync, time_backward])))