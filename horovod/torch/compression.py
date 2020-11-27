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
    def __init__(self):
        pass
    """Interface for compressing and decompressing a given tensor."""
    def compress(self, tensor, step):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    def decompress(self, tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    def compress(self, p, step=0):
        """Returns the tensor unmodified."""
        if p.requires_grad:
            return p.grad, None
        else:
            return p, None

    def decompress(self, tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    def compress(self, p, step=0):
        """Downcasts the tensor to 16-bit."""
        if p.requires_grad:
            tensor = p.grad
        else:
            tensor = p
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    def decompress(self, tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class FP32Compressor(Compressor):
    """Decompress all floating 32 gradients to 32-bit."""
    def compress(self, p, step=0):
        """Upcasts the tensor to 32-bit."""
        if p.requires_grad:
            tensor = p.grad
        else:
            tensor = p
        tensor_compressed = tensor
        if tensor.dtype == torch.float16:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float32)
        return tensor_compressed, tensor.dtype

    def decompress(self, tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor()

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor()

    """Compress all floating point gradients to 32-bit."""
    fp32 = FP32Compressor()