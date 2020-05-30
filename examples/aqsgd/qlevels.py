from .nuq import NUQEstimator
import horovod.torch as hvd
import ctypes
import torch
import time

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        if key in self.d:
            return self.d[key]
        else:

            return None


def convert_to_ctypes(arr):
    c_double_p = ctypes.POINTER(ctypes.c_double)
    a = arr.ctypes.data_as(c_double_p)
    return a


def get_minvar_loader(train_loader, opt):
    kwargs = {'num_workers': opt.workers,
              'pin_memory': True} if opt.cuda else {}
    idxdataset = train_loader.dataset
    train_loader = torch.utils.data.DataLoader(
        idxdataset,
        batch_size=opt.g_batch_size,
        shuffle=True,
        drop_last=False, **kwargs)
    return train_loader


class LevelsEst:
    def __init__(self, model, train_loader, args):
        self.model = model
        self.opt = {
            "nuq_number_of_samples": 10,
            "nuq_ig_sm_bkts": True,
            "nuq_bucket_size": args.bucket_size,
            "workers": 4,
            "cuda": True,
            "g_batch_size": args.batch_size,
            "nuq_method": "amq_nb",
            "nuq_learning_rate": 0.7,
            "nuq_truncated_interval": 1,
            "nuq_cd_epochs": 30,
            "nuq_layer": 0,
            "g_estim": 'nuq',
            "update_epochs": [2, 30, 60, 80],
            "nuq_bits": args.quantization_bits,
            'nuq_mul': 0.5,
            "nuq_amq_lr": 0.7,
            "nuq_amq_epochs": 50,
            "nuq_sym": False,
            "nuq_inv": False,
            "delay_epoch_start": 10,  # number of steps to wait after lr decaying epoch starts
            "logger_name": None,
            "dist_num": 350
        }
        print("ALQ opts:", self.opt)
        self.opt = DictWrapper(self.opt)
        self.gest_used = False
        data_loader = get_minvar_loader(train_loader, self.opt)
        self.gestim = NUQEstimator(data_loader, self.opt)

    def update_levels(self, epoch, batch_idx):
        if (epoch + 1) in self.opt.update_epochs and batch_idx == self.opt.delay_epoch_start:
            opt = self.opt
            model = self.model
            if hvd.rank() == 0:
                print("Epoch: {}, step: {}".format(epoch + 1, batch_idx))
                if opt.g_estim == 'nuq' and opt.nuq_method != 'none':
                    stats = self.gestim.snap_online_mean(model)
                    self.gestim.qdq.set_mean_variance(stats)

                isamq = opt.nuq_method == 'amq' or opt.nuq_method == 'amq_nb'
                isalq = opt.nuq_method == 'alq' or opt.nuq_method == 'alq_nb'
                if isamq or isalq:
                    self.gestim.qdq.update_levels()
                qlevels_t = self.gestim.qdq.get_levels()
                print(qlevels_t)
            else:
                time.sleep(30)
                qlevels_t = torch.tensor([0.0]*(1 << (self.opt.nuq_bits-1)), dtype=torch.float32)
            qlevels_t = hvd.broadcast(qlevels_t, root_rank=0)
            qlevels = qlevels_t.numpy()
            qlevels = convert_to_ctypes(qlevels)
            hvd.set_quantization_levels(qlevels)
