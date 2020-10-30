# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch

class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    def compress(self, param, step):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(self, tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    def compress(self, param, step):
        """Returns the tensor unmodified."""
        if not param.requires_grad:
            return param, None
        else:
            return param.grad, None

    def decompress(self, tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    def compress(self, param, step):
        """Downcasts the tensor to 16-bit."""
        if not param.requires_grad:
            return param, None
        tensor_compressed = param.grad
        if tensor_compressed.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = param.grad.type(torch.float16)
        return tensor_compressed, param.grad.dtype

    def decompress(self, tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class TopKCompressor(Compressor):
    def __init__(self, k_perc=1.0, bucket_size=512, enable_error_correction=None, warmup_steps=1000, ef_q=None):
        super().__init__()
        self.k_perc = k_perc
        self.enable_error_correction = enable_error_correction
        self.eps = 1e-10
        self.bucket_size = bucket_size
        self.state = {}
        self.warm_up = warmup_steps
        self.ef_q = ef_q
        assert not (self.enable_error_correction and (self.ef_q is not None))

    def compress(self, p, step):
        if not p.requires_grad:
            return p, None

        grad = p.grad.data.detach().clone()
        grad_ = grad.view(-1)
        if p not in self.state:
            self.state[p] = {}
            self.state[p]["error_correction"] = torch.full_like(grad_, 0.0, memory_format=torch.preserve_format)
        state = self.state[p]
        e_c = state["error_correction"]
        if step < self.warm_up:
            return grad, None

        if self.enable_error_correction:
            # add error correction
            grad_.add_(e_c)
            # update error correction before subtraction
            e_c.copy_(grad_)

        numel = grad_.numel()
        bucket_size = self.bucket_size if self.bucket_size else numel
        # if numel < bucket_size:
        #     return p.grad, None
        main_chunk_size = (numel // bucket_size) * bucket_size
        tail_chunk_size = numel - main_chunk_size
        if main_chunk_size > 0:
            g_ = grad_[:main_chunk_size].view((-1, bucket_size))
            self.compress_chunk(g_, bucket_size)
        if tail_chunk_size > 0:
            g_ = grad_[main_chunk_size:]
            self.compress_chunk(g_, tail_chunk_size)
        if self.enable_error_correction:
            e_c.sub_(grad_)
        return grad, None

    def compress_chunk(self, g_, numel):
        num_zero = numel - max(int(self.k_perc * numel), 1)
        _, indices = torch.topk(g_.abs(), num_zero, dim=g_.dim() - 1, largest=False)
        if self.ef_q is not None:
            values = g_.gather(g_.dim() - 1, indices)
            self.quantize(values.view(-1))
            g_.scatter_(g_.dim() - 1, indices, values)
        else:
            g_.scatter_(g_.dim() - 1, indices, 0.0)

    def quantize(self, buf):
        q_bits = self.ef_q["bits"]
        if q_bits == 32 or q_bits <= 0:
            return buf
        levels = 1 << q_bits
        numel = buf.numel()
        bucket_size = self.ef_q["bucket_size"] if "bucket_size" in self.ef_q else numel
        main_chunk_size = (numel // bucket_size) * bucket_size
        tail_chunk_size = numel - main_chunk_size
        if main_chunk_size > 0:
            r_ = buf[:main_chunk_size].view((-1, bucket_size))
            self.quantize_bucket(r_, levels)
        if tail_chunk_size > 0:
            r_ = buf[main_chunk_size:]
            self.quantize_bucket(r_, levels)
        return buf

    def quantize_bucket(self, a, levels):
        if a.dim() == 2:
            fmin = torch.min(a, dim=0)[0]
            fmax = torch.max(a, dim=0)[0]
            unit = (fmax - fmin) / (levels - 1)
            unit = unit[None, :]
            fmin = fmin[None, :]
            s = torch.Tensor([1e-11]).expand_as(unit).to(a.device)
        else:
            fmin = torch.min(a)
            fmax = torch.max(a)
            unit = (fmax - fmin) / (levels - 1)
            s = torch.Tensor([1e-11]).to(a.device)

        unit = torch.max(unit, s)
        a -= fmin
        a /= unit
        a += torch.empty(a.size(), device=a.device).uniform_(0, 1)
        torch.floor_(a)
        a *= unit
        a += fmin
        return a

    def decompress(self, tensor, ctx):
        return tensor

class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor

    topk = TopKCompressor
