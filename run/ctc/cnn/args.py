import argparse, sys, os

sys.path.append(os.path.join("..", "..", ".."))

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", "-e", type=int, default=1000)
parser.add_argument("--grad-clip", "-gc", type=float, default=1) 
parser.add_argument("--weight-decay", "-wd", type=float, default=1e-5) 
parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
parser.add_argument("--lr-decay", "-decay", type=float, default=1)
parser.add_argument("--final-learning-rate", type=float, default=1e-6)
parser.add_argument("--momentum", "-mo", type=float, default=0.9)
parser.add_argument("--optimizer", "-opt", type=str, default="adam")

parser.add_argument("--augmentation", "-augmentation", default=False, action="store_true")

parser.add_argument("--sampling-rate", "-rate", type=int, default=16000) 
parser.add_argument("--frame-width", "-fwidth", type=float, default=0.032) 
parser.add_argument("--frame-shift", "-fshift", type=float, default=0.01) 
parser.add_argument("--num-mel-filters", "-nmel", type=int, default=40) 
parser.add_argument("--window-func", "-winf", type=str, default="hanning")
parser.add_argument("--using-delta", dest="using_delta", default=True, action="store_true") 
parser.add_argument("--not-using-delta", dest="using_delta", action="store_false") 
parser.add_argument("--using-delta-delta", dest="using_delta_delta", default=True, action="store_true") 
parser.add_argument("--not-using-delta-delta", dest="using_delta_delta", action="store_false") 

parser.add_argument("--ndim-h", "-dh", type=int, default=128)
parser.add_argument("--ndim-dense", "-dd", type=int, default=320)
parser.add_argument("--num-conv-layers", "-nconv", type=int, default=4)
parser.add_argument("--num-dense-layers", "-ndense", type=int, default=4)
parser.add_argument("--wgain", "-w", type=float, default=1)

parser.add_argument("--nonlinear", type=str, default="relu")
parser.add_argument("--dropout", "-dropout", type=float, default=0)
parser.add_argument("--weightnorm", "-weightnorm", default=False, action="store_true")
parser.add_argument("--architecture", "-arch", type=str, default="zhang")

parser.add_argument("--gpu-device", "-g", type=int, default=0) 
parser.add_argument("--working-directory", "-cwd", type=str, default=None)

parser.add_argument("--buckets-limit", type=int, default=None)
parser.add_argument("--dataset-path", "-data", type=str, default=None)
parser.add_argument("--apply-cmn", "-cmn", default=False, action="store_true")
parser.add_argument("--seed", "-seed", type=int, default=0)
args = parser.parse_args()