import subprocess
import os
DIR="scale_logs"
os.makedirs(DIR, exist_ok=True)

models_name = ["resnet18", "resnet34", "resnet50", "resnet152",
               "vgg16", "vgg19",
               "wide_resnet50_2", "wide_resnet101_2",
               "resnext50_32x4d", "resnext101_32x8d",
               "densenet121", "densenet161", "densenet201",
               "mnasnet1_0", "alexnet"]


bb = [-1.0, 0.0, 0.01, 0.5]

reduction_algos = ["none", "Ring", "ScatterAllgather"]

env = os.environ.copy()
# env["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
env["HOROVOD_CACHE_CAPACITY"] = "0"

for name in models_name:
    for b in bb:
        for r in reduction_algos:

            command = "horovodrun -np 2 -H localhost:2 --reduction-type {} ".format(r)
            if r == "none":
                q = 32
            else:
                q = 4
            command += "--compression-type expL2 --quantization-bits {} python pytorch_synthetic_benchmark.py --num-iters 5 --model {} --bb-l2-ratio {} ".format(q, name, b)
            command += "--num-parallel-steps 5"
            file_name=name
            if r != "none":
                file_name += "_" + r
            if b >= 0.0:
                file_name += "_" + str(b)
            FILE = os.path.join(DIR, file_name)
            print(command, FILE)
            with open(FILE, "w") as f:
                f.write("Reduction: {}, bb: {}\n".format(r, b))
                p = subprocess.Popen(command, env=env, stdout=f, shell=True)
            p.communicate()