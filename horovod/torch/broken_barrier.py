from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.torch.compression import Compression
from horovod.torch.mpi_ops import allreduce_async_, Average
from horovod.torch.mpi_ops import poll, synchronize
from horovod.torch.mpi_ops import size, rank, local_rank

import threading
import logging

try:
    import queue
except ImportError:
    import Queue as queue
import time
import math
import torch
import horovod.torch as hvd
import numpy as np

_DistributedOptimizer = hvd._DistributedOptimizer
_hvd_DistributedOptimizer = hvd.DistributedOptimizer
broadcast_parameters = hvd.broadcast_parameters
broadcast_optimizer_state = hvd.broadcast_optimizer_state


class _BrokenBarrier(_DistributedOptimizer):
    def __init__(self, model, hvd_opt, num_steps=10 ** 6, num_parallel_steps=10, l2_barrier_cond_ratio=0.5):
        """Construct a new Optimizer with broken Barriern, which uses horovod optimizer under the hood for averaging gradients
         across all workers.
        Args:
            model: The training model. BytePS uses the model object to register hooks.
            hvd_opt: Optimizer to use for averaging gradients and applying updates.
            num_steps: The maximum number of training steps. BytePS needs to know when to stop cross-iteration
            scheduling.
            num_parallel_steps: The maximum number of training steps done in parallel with broken barrier.
        """
        if size() <= 1:
            raise RuntimeError("Broken Barrier doesn't support single node execution")
        self._model = model
        self._opt = hvd_opt
        self._logger = logging.getLogger("BrokenBarrier")

        self._logger.info("BrokenBarrier is enabled.")
        self._logger.debug("size {}, rank {}".format(size(), rank()))
        self._desc = "rank {}".format(rank())

        # Track training steps
        self._step = 0
        self._final_step = num_steps
        self._num_parallel_steps = num_parallel_steps
        self._l2_barrier_cond_ratio = l2_barrier_cond_ratio
        self._counters = {}
        self._buffers = {}
        self._ready_grads = {}
        self._current_version = {}
        self._sync_state = {}
        for param_group in self.param_groups:
            for p in param_group['params']:
                if not p.requires_grad:
                    continue
                self._ready_grads[p] = queue.Queue()
                self._current_version[p] = 0
                q = queue.Queue()
                for i in range(self._num_parallel_steps):
                    q.put(i)
                self._counters[p] = q
                self._buffers[p] = []
                # States. In case backward hook didn't work, we can find out if we need to try to sync on synchronize().
                # False - The version of this parameter is not in sync now. True - already synchronized.
                self._sync_state[p] = [False for i in range(self._num_parallel_steps)]
        # Constraints for barrier. They are set per step.
        self._barrier_constraint_lock = threading.Lock()
        self._barrier_constraint = {
            "step": 0,
            "synced_value": 0.0,
            "target_value": 0.0
        }
        self._barrier_broken = True
        # self.norm_time = 0.0
        if size() > 1:
            self._register_forward_hooks()
            self._register_hooks()

            # Poll whether the tensor allreduce is finished.
            self._event_queue = queue.Queue()
            self._poller = threading.Thread(target=self._poll, args=())
            self._poller.start()

    def __getattr__(self, item):
        return getattr(self._opt, item)

    def step(self, closure=None):
        """Override the default step function."""
        # Step 0 is called for parameter initialization after parameter broadcast
        if size() > 1 and self._step > 0:
            self._synchronize()
            # if it is the final training step, wait for the completion of all tensors
            if self._step == self._final_step:
                self._logger.debug("final step {}, waiting for allreduce completion.".format(self._final_step))
                while not self._event_queue.empty():
                    time.sleep(0.001)
                self._event_queue.put((None, None, None))
                self._poller.join()
                # Finish all updates.
                for param_group in self.param_groups:
                    for p in param_group['params']:
                        self._apply_gradients(p)
                self._logger.info("training finished!")
            loss = None
            if closure is not None:
                loss = closure()
            self._step += 1
            return loss
        else:
            # Optimizer.step() will be triggered when user calls horovod.broadcast_optimizer_sate()
            super(self._opt.__class__, self._opt).step()
            self._step += 1

    def _make_hook(self, p):
        def hook(*ignore):
            assert not p.grad.requires_grad
            self._allreduce_grad_async(p)
        return hook

    def zero_grad(self):
        """Override the default zero_grad function.
        Clears the gradients of all optimized tensors.
        """
        if size() > 1 and self._step > 0:
            pass
            # for param_group in self.param_groups:
            #     for p in param_group['params']:
            #         self._zero_one_grad(p)
        else:
            self._opt.zero_grad()


    def _get_parameter_name(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        return name

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _synchronize(self):
        """Push pull missing parameters"""
        # Reset constrains related to the previous step. _step is not incremented yet.
        missing_p = self._requires_update - set(self._sync_state.keys())
        for p in missing_p:
            self._logger.info("{} is not in sync".format(self._get_parameter_name(p)))
            self._allreduce_grad_async(p)

        for p in self._requires_update:
            if self._sync_state[p][self._current_version[p]] is None:
                self._logger.info("{} is not in sync".format(self._get_parameter_name(p)))
                self._allreduce_grad_async(p)

    def _get_version(self, p):
        while True:
            try:
                v = self._counters[p].get()
                return v
            except queue.Empty:
                self._logger.error("Counter queue is empty")

    def _put_version(self, p, v):
        self._counters[p].put(v)

    def _init_buffers(self, p, tensor):
        buffers = self._buffers[p]
        if len(buffers) == 0:
            # first call
            buffers.append(tensor)
            for i in range(self._num_parallel_steps - 1):
                buffers.append(torch.zeros_like(tensor))

    def _get_buffer(self, p, v):
        return self._buffers[p][v]

    def _apply_gradient(self, p, ctx):
        version = ctx["version"]
        buffers = self._buffers[p]
        buffer = buffers[version]
        if isinstance(self._opt, torch.optim.SGD):
            self._sgd(p, buffer)
        else:
            raise ValueError("Invalid optimizer! Only support SGD.")
        buffer.zero_()
        self._sync_state[p][version] = None
        self._put_version(p, version)
        return ctx["step"]

    def _apply_gradients(self, p):
        q = self._ready_grads[p]
        latest_step = 0
        while not q.empty():
            ctx = q.get()
            latest_step = max(self._apply_gradient(p, ctx), latest_step)
        return latest_step

    def _reset_barrier_constraint(self):
        with self._barrier_constraint_lock:
            if self._barrier_constraint["step"] != self._step:
                self._barrier_constraint["step"] = self._step
                self._barrier_constraint["synced_value"] = torch.tensor([0.0], device=torch.cuda.current_device())
                self._barrier_constraint["target_value"] = torch.tensor([0.0], device=torch.cuda.current_device())
                self._barrier_broken = False

    def _update_barrier_constraint_target(self, tensor, ctx):
        l2_norm = torch.norm(tensor.view(-1), p=2)
        self._barrier_constraint["target_value"].add_(l2_norm)
        ctx["l2_norm"] = l2_norm

    def _update_barrier_constraint_synced(self, ctx):
        self._barrier_constraint["synced_value"] += ctx["l2_norm"]

    def _barrier_constraint_satisfied(self):
        # We check if updates of the previous step synchronization reached a threshold
        return self._barrier_constraint["synced_value"] >= self._l2_barrier_cond_ratio * self._barrier_constraint["target_value"]

    def _on_grad_ready(self, p, output, ctx):
        buffers = self._buffers[p]
        version = ctx["version"]
        buffers[version].set_(output)
        with self._barrier_constraint_lock:
            if self._barrier_constraint["step"] == ctx["step"] and not self._barrier_broken:
                # if rank() == 0:
                #     print("Ready updating {}".format(self._get_parameter_name(p)))
                #     print("Step", ctx["step"])
                #     print("Barrier broken", self._barrier_broken)
                self._update_barrier_constraint_synced(ctx)
                self._barrier_broken = self._barrier_constraint_satisfied()
        self._sync_state[p][version] = True
        self._ready_grads[p].put(ctx)

    def _allreduce_grad_async(self, p):
        """Call byteps API to allreduce gradient asynchronously
        Arguments:
            tensor: The tensor to allreduce.
            name: The name of the tensor.
        Returns:
            an allreduce handle and context
        """
        name = self._get_parameter_name(p)
        tensor = p.grad
        # s = time.time()
        # tensor_compressed, compression_ctx = self._compression.compress(tensor)
        v = self._current_version[p]
        # reset barrier constraints, update step there.
        # Barrier is broken, so no need to track constraints of the previous step
        self._reset_barrier_constraint()

        # Initialize buffers if they are not initialized yet.
        # self._init_buffers(p, p.grad.data)
        b = self._get_buffer(p, v)
        handle = allreduce_async_(b, op=Average, name="Gradient.{}.{}".format(name, v))
        ctx = {}
        # ctx["compress"] = compression_ctx
        ctx["version"] = v
        ctx["step"] = self._step
        self._update_barrier_constraint_target(tensor, ctx)
        self._sync_state[p][v] = False
        # Add to queue to poll completion
        self._event_queue.put((p, handle, ctx))
        # self.norm_time += time.time() - s

    def log_if_fc(self, p, msg):
        l = '{} '.format(self._get_parameter_name(p))
        if ("fc.bias" in l) and rank() == 0:
            self._logger.debug(l + msg)

    def _poll(self):
        """Poll the completion of the tensor's backward or allreduce from a FIFO event_queue"""
        while True:
            try:
                p, handle, ctx = self._event_queue.get()
            except queue.Empty:
                self._logger.info("Event queue is empty.")
                continue
            if p is None:
                self._logger.debug("poller exits.")
                break
            # Check whether the allreduce is finished. If so, start updating parameters.
            if handle is not None and poll(handle):
                output = synchronize(handle)
                # p.grad.set_(self._compression.decompress(output, ctx["compress"]))
                self._on_grad_ready(p, output, ctx)
            else:
                self._event_queue.put((p, handle, ctx))

    def _register_forward_hooks(self):
        """Add hook before forward propagation of each layer to block forward computation until the allreduce and
        parameter update is finished. The blocking is implemented using a lock."""
        # Recursively find all submodules
        submodules = []
        q = queue.LifoQueue()
        for mod in self._model.children():
            q.put(mod)
        while not q.empty():
            mod = q.get()
            if len(list(mod.children())) == 0:
                submodules.append(mod)
            else:
                for m in mod.children():
                    q.put(m)

        def pre_forward_hook(mod, input):
            waited_params = set(mod.parameters())
            prev_step = self._step - 1
            while True:
                # check if there are updates and apply them.
                synced_params = set()
                for p in waited_params:
                    latest_updated = self._apply_gradients(p)
                    if latest_updated == prev_step:
                        synced_params.add(p)
                    with self._barrier_constraint_lock:
                        if self._barrier_broken:
                            break
                waited_params.difference_update(synced_params)
                if len(waited_params) > 0:
                    # yield
                    time.sleep(0)
                else:
                    break
            for p in waited_params:
                self._apply_gradients(p)

            for p in mod.parameters():
                self._current_version[p] = self._get_version(p)
                self._init_buffers(p, p.grad.data)
                p.grad.data = self._get_buffer(p, self._current_version[p])
            self._logger.debug("{} Starts forward {}.".format(self._desc, mod))


        def after_forward_hook(mod, input, result):
            self._logger.debug("{} finished forward {}.".format(self._desc, mod))

        # Register pre-hook and hook for each module
        for mod in reversed(submodules):
            self._logger.debug("{} registers forward hook on module {}".format(self._desc, mod))
            mod.register_forward_pre_hook(pre_forward_hook)
            mod.register_forward_hook(after_forward_hook)

    def _zero_one_grad(self, p):
        """Clears the gradient of one variable as torch accumulates gradients by default.
        Arguments:
            p: the parameter.
        """
        if p.grad is not None:
            if p in self._locks:
                with self._locks[p]:
                    p.grad.detach_()
                    p.grad.zero_()
            else:
                p.grad.detach_()
                p.grad.zero_()


    """Below are the implementations of optimizers, e.g., SGD, Adam, RMSprop.
    The implementation is derived from Torch's code, except that we update one parameter each time."""

    def _sgd(self, p, grad_data):
        """Performs a single optimization step using SGD optimizer on a parameter.
        Arguments:
            p: The parameter to be updated.
        """
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for gp in group['params']:
                if self._get_parameter_name(p) != self._get_parameter_name(gp) or gp.shape != p.shape:
                    continue
                if p.grad is None:
                    continue
                d_p = grad_data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
                break

def _init_bsc():
    """Replace _register_hook() function in _DistributedOptimizer with empty function."""

    def hijack(obj, func_name):
        orig_func = getattr(obj, func_name)

        # print("hijack function {}".format(orig_func))

        def wrapped_func(*args, **kwargs):
            # print("function {} is hijacked to do nothing.".format(orig_func))
            return

        setattr(obj, func_name, wrapped_func)

    hijack(_DistributedOptimizer, '_register_hooks')


def _init_logger():
    logger = logging.getLogger("BrokenBarrier")
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(levelname)s: %(message)s',
                                  '%H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler('broken_barrier.log', 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    logger.setLevel(logging.INFO)


def BrokenBarrier(model,
                  optimizer,
                  named_parameters=None,
                  compression=Compression.none,
                  backward_passes_per_step=1,
                  num_steps=10 ** 6, num_parallel_steps=10, l2_barrier_cond_ratio=0.5):
    """Wrap Torch optimizer using BytePS DistributedOptimizer and _BrokenBarrier."""
    hvd_opt = _hvd_DistributedOptimizer(optimizer, named_parameters, compression, backward_passes_per_step)
    return _BrokenBarrier(model, hvd_opt, num_steps, num_parallel_steps, l2_barrier_cond_ratio)


_init_bsc()
_init_logger()
