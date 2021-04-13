import argparse
import horovod.torch as hvd
import pickle
import numpy.random as rnd
import numpy as np
import torch
import os

arrays = []

class MaxMinQuantizer():
    def __init__(self, q_bits, bucket_size):
        self.q = q_bits
        self.num_levels = 1 << self.q
        self.bucket_size = bucket_size

    def compress(self, a):
        if self.q == 32:
            return a
        numel = a.numel()
        if self.bucket_size == -1:
            a[:] = self.quantize_bucket(a)
        else:
            main_chunk_size = (numel // self.bucket_size) * self.bucket_size
            if main_chunk_size > 0:
                a[:main_chunk_size] = self.quantize_bucket(a[:main_chunk_size].view((-1, self.bucket_size))).view(-1)
            if numel - main_chunk_size > 0:
                a[main_chunk_size:] = self.quantize_bucket(a[main_chunk_size:])
        return a

    def quantize_bucket(self, a):
        non_2 = False
        if a.dim() != 2:
            a = a[None, :]
            non_2 = True
        if a.dim() == 2:
            fmin = torch.min(a, dim=1)[0]
            fmax = torch.max(a, dim=1)[0]
            unit = (fmax - fmin) / (self.num_levels - 1)
            unit = unit[:, None]
            fmin = fmin[:, None]
            s = torch.Tensor([1e-11]).expand_as(unit).to(a.device)
        unit = torch.max(unit, s)
        a -= fmin
        a /= unit
        a += torch.empty(a.size(), device=a.device).uniform_(0, 1)
        # log(a.cpu().numpy())
        torch.floor_(a)
        a *= unit
        a += fmin
        if non_2:
            return a[0]
        return a

class NormUniformQuantizer(MaxMinQuantizer):
    def __init__(self, q_bits, bucket_size):
        super().__init__(q_bits, bucket_size)
        self.num_levels = self.num_levels // 2

    def quantize_bucket(self, a):
        non_2 = False
        if a.dim() != 2:
            a = a[None, :]
            non_2 = True
        if a.dim() == 2:
            vnorm = torch.norm(a, p=float("inf"), dim=1)
            vnorm = vnorm[:, None]
            s = torch.Tensor([1e-11]).expand_as(vnorm).to(a.device)
        else:
            vnorm = torch.norm(a, p=float("inf"))
            s = torch.Tensor([1e-11]).to(a.device)

        vnorm = torch.max(vnorm, s)
        sign = torch.sign(a)
        # cast sign to 1 bit
        sign.add_(1).div_(2)
        sign.mul_(2).add_(-1)
        if self.num_levels > 1:
            q = torch.abs(a / vnorm)
            r = torch.rand(a.shape, device=a.device)
            q.mul_((self.num_levels - 1))
            q.add_(r)
            torch.floor_(q)
            q.div_((self.num_levels - 1))
            res = q * vnorm * sign
        else:
            res = vnorm * sign
        if non_2:
            return res[0]
        else:
            return res


def log(msg):
    if hvd.rank() == 0:
        print(msg)

def generate_arrays(size):
    rnd.seed(43)
    global arrays
    sum = np.zeros(size)
    for i in range(num_nodes):
        array = rnd.normal(size=size, scale=0.1)
        sum = np.add(sum, array)
        arrays.append(array)
    arrays.append(sum)


def get_array(idx):
    return arrays[idx]


def get_expected_result():
    quantizer = MaxMinQuantizer(args.q, args.bucket_size)
    # quantizer = NormUniformQuantizer(args.q, args.bucket_size)
    rank = hvd.rank()
    array = torch.tensor(get_array(rank), device="cuda")
    # print(array)
    quantizer.compress(array)
    # print(array.cpu().numpy())
    a = hvd.allgather(array, "allgather").view(-1, *array.shape)
    return torch.sum(a, dim=0)


def run_allreduce(args, num, res):
    array = get_array(hvd.rank())
    res = get_array(hvd.size())
    # print(array[:8])
    tensors = []
    for i in range(num):
        if args.no_cuda:
            tensor = torch.tensor(array, device='cpu').float()
        else:
            tensor = torch.tensor(array, device='cuda').float()
        tensors.append(tensor)
    torch.cuda.synchronize()
    if args.fp16:
        tensor = tensor.half()
    handles = []
    for i in range(num):
        handles.append(hvd.allreduce_async_(tensors[i], name='test.{}'.format(i), op=hvd.Sum))
        #tensors[i] = hvd.synchronize(handles[-1])
        #tensors[i] = hvd.allreduce_(tensors[i], name='test.{}'.format(i), op=hvd.Sum)
        #tensors[i] = hvd.allreduce(tensors[i], name='test.{}'.format(i),
        #                       compression=hvd.Compression.fp16 if args.quantization_bits == 16 else hvd.Compression.none)
        #for i in range(num_nodes):
        #    print(i, get_array(i)[:8])
        #print("Base sum: ", result[:8])
        #print("Hvd: ", avg[:8])
    for i in range(num):
        h = handles[i]
        avg_tensor = hvd.synchronize(h)
        #avg_tensor = tensors[i] 
        #print(avg_tensor[:8])
        #avg_tensor = tensors[i]
        avg = avg_tensor.cpu().numpy()
        if hvd.rank() == 0:
            diff = np.linalg.norm(res - avg)
            if diff > res.size * 5e-2:
                log("L2 error: {}".format(diff))
                log("Base sum: {}".format(res[:8]))
                log("Hvd: {}".format(avg[:8]))
                hvd.broadcast_object(False, 0, "Result")
                return False
            else:
                hvd.broadcast_object(True, 0, "Result")
        else:
            res = hvd.broadcast_object(None, 0, "Result")
            if not res:
                return False
    return True

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument("--array-size", type=int, default=32,
                    help="array size (default: 32)")
parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
parser.add_argument('--fp16', action='store_true', default=False,
                            help='Casts tensors to fp16')
parser.add_argument('-q', type=int, default=32, help="quantization bits")
parser.add_argument('--bucket-size', type=int, default=512, help="quantization bucket size")


args = parser.parse_args()

# os.environ["HOROVOD_QUANTIZATION_BITS"] = str(args.q)
# os.environ["HOROVOD_COMPRESSION_BUCKET_SIZE"] = str(args.bucket_size)
hvd.init()
num_nodes = hvd.size()
torch.cuda.set_device(hvd.rank())
generate_arrays(args.array_size)

# res = get_expected_result()
# res = res.cpu().numpy()
res = None

num_layers = 1
num_batches = 100
for i in range(num_batches):
    if not run_allreduce(args, num_layers, res):
        log("Failed")
        break
else:
    log("Passed")
