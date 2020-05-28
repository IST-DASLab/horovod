from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
from models import *
import os

from tqdm import tqdm
from distutils.version import LooseVersion
import time
from aqsgd.qlevels import LevelsEst

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset-dir', default=os.path.expanduser('./cifar10'),
                    help='path to training data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('--checkpoint-dir', default=None,
                    help='checkpoint directory')
parser.add_argument('--resume-from', default='',
                    help='Resume from model')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=256,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay')

parser.add_argument('--num-parallel-steps', type=int, default=10,
                    help='number of sgd steps done in parallel')
parser.add_argument('--bb-l2-ratio', type=float, default=-1.0,
                    help='Ratio of l2 norm to collect to break the barrier')
parser.add_argument('--quantization-bits', type=int, default=4,
                    help='number of sgd steps done in parallel')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.bb_l2_ratio < 0.0:
    import horovod.torch as hvd
else:
    import horovod.torch.broken_barrier_new as hvd


allreduce_batch_size = args.batch_size * args.batches_per_allreduce
print(args)
hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
checkpoint_format = "checkpoint-{epoch}.pth.tar"
if os.path.exists(args.resume_from):
    for try_epoch in range(args.epochs, 0, -1):
        basename = os.path.basename(args.resume_from)
        if basename == checkpoint_format.format(epoch=try_epoch):
            resume_from_epoch = try_epoch
            break

# Horovod: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()

# Horovod: print logs on the first worker.
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: write TensorBoard logs on first worker.
try:
    if LooseVersion(torch.__version__) >= LooseVersion('1.2.0'):
        from torch.utils.tensorboard import SummaryWriter
    else:
        from tensorboardX import SummaryWriter
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None
except ImportError:
    log_writer = None

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform_train)
# Horovod: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=hvd.size()` and `rank=hvd.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

val_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform_test)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)


# Set up standard ResNet-20 model.
# model = resnet32(num_classes=100)
model = resnet18(num_classes=10)

# By default, Adasum doesn't need scaling up learning rate.
# For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = args.batches_per_allreduce * hvd.local_size()

# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          lr_scaler),
                      momentum=args.momentum, weight_decay=args.wd)
nuq_level_est = LevelsEst(model, train_loader, args)

# Horovod: wrap optimizer with DistributedOptimizer.
if args.bb_l2_ratio < 0.0:
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average)
else:
    optimizer = hvd.BrokenBarrier(model,
                                  optimizer, named_parameters=model.named_parameters(), num_steps=len(train_loader) * args.epochs,
                                  backward_passes_per_step=args.batches_per_allreduce, l2_barrier_cond_ratio=args.bb_l2_ratio)



# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.resume_from
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def adjust_learning_rate(epoch, batch_idx):
    # if epoch < 60:
    #     lr_adj = 1.
    # elif epoch < 120:
    #     lr_adj = 2e-1
    # elif epoch < 160:
    #     lr_adj = 4e-2
    # else:
    #     lr_adj = 8e-3
    # if epoch < 81:
    #     lr_adj = 1.
    # elif epoch < 122:
    #     lr_adj = 1e-1
    # elif epoch < 164:
    #     lr_adj = 1e-2
    # else:
    #     lr_adj = 1e-3
    if epoch < 150:
        lr_adj = 1.
    elif epoch < 250:
        lr_adj = 1e-1
    else:
        lr_adj = 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * args.batches_per_allreduce * lr_adj


def train(epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1)) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            nuq_level_est.update_levels(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = criterion(output, target)
            train_loss.update(loss)
            loss.backward()
            optimizer.step()
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)


def validate(epoch):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(criterion(output, target))

                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

start_time = time.time()
for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
num_images = allreduce_batch_size * len(train_loader) * args.epochs
elapsed_time = time.time() - start_time
print("Elapsed time: ", elapsed_time)
print("{} Imgs/sec".format(num_images / elapsed_time))
