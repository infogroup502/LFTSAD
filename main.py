# 这是一个示例 Python 脚本。
import os
import argparse
import numpy as np
from torch.backends import cudnn
from utils.utils import *
from solver import Solver
import time
import warnings
warnings.filterwarnings('ignore')
import json
import sys
import time
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 示例用法
set_seed(42)

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # 确保立即输出到终端
        mode = 'a' if self.add_flag else 'w'
        with open(self.filename, mode) as log:
            log.write(message)
            log.flush()  # 确保立即写入文件

    def flush(self):
        self.terminal.flush()


def str2bool(v):
    return v.lower() in ('true')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(array[idx - 1])

def main(config):
    cudnn.benchmark = True

    solver = Solver(vars(config))

    end_time = time.time()
    solver.train()
    print(f"{config.dataset}\n win_size: {config.win_size},  patch_size: {config.patch_size},patch_seq: {config.patch_seq},batch_size: {config.batch_size},seq_size:{config.seq_size}"
          f"\n num_epochs: {config.num_epochs},  d_model: {config.d_model},  anormly_ratio: {config.anormly_ratio}"
          f"\n sw_max_mean: {config.sw_max_mean}, sw_loss： {config.sw_loss }, p_seq : {config.p_seq}")
    solver.test()
    elapsed_time = end_time - start_time
    print(f"程序运行时间: {elapsed_time}秒")


    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    dataset_name = 'MSL'
    parser.add_argument('--win_size', type=int, default=100)#窗口大小，减小窗口大小可能有助于捕捉更精细的特征，而增大窗口大小则可能有助于捕捉更全局的特征。
    parser.add_argument('--patch_size', type=str, default='[10]')#
    parser.add_argument('--patch_seq', type=str, default='[5]')#
    parser.add_argument('--seq_size', type=int, default=5)#
    parser.add_argument('--num_epochs', type=int, default=1)#训练轮次，防止过拟合
    parser.add_argument('--batch_size', type=int, default=128)#批量大小，较小使模型更快但不稳定。较大更加稳定但较慢
    parser.add_argument('--input_c', type=int, default=55)#输入通道数
    parser.add_argument('--d_model', type=int, default=128)#模型维度，增加或减少这个值可能会改变模型的容量和复杂度
    parser.add_argument('--anormly_ratio', type=float, default=0.82)#异常检测的比例
    parser.add_argument('-sw_max_mean', type=int, default=0, help='0:mean , 1:max')
    parser.add_argument('-sw_loss', type=int, default=0, help='0:mse  , 1:mae')
    parser.add_argument('-p_seq', type=float, default=0, help='  点级别和子序列比例  ')

    parser.add_argument('-min_size', type=int, default=256)
    parser.add_argument('--output_c', type=int, default=1)
    parser.add_argument('--dataset', type=str, default=f'{dataset_name}')
    parser.add_argument('--data_path', type=str, default=f'{dataset_name}')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Default
    parser.add_argument('--index', type=int, default=137)

    config = parser.parse_args()

    # Convert JSON string to list of ints
    config.patch_size = [int(x) for x in json.loads(config.patch_size)]
    config.patch_seq = [int(x) for x in json.loads(config.patch_seq)]

    config.output_c = config.input_c
    config.min_size = config.batch_size

    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]
    log_file = f'{config.dataset}_log_file.log'
    sys.stdout = Logger(filename=log_file, add_flag=True, stream=sys.stdout)
    sys.stderr = Logger(filename=log_file, add_flag=True, stream=sys.stderr)
    main(config)