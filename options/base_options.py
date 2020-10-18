import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True)
        self.parser.add_argument('--batchSize', type=int, default=1)
        self.parser.add_argument('--loadSize', type=int, default=288)
        self.parser.add_argument('--fineSize', type=int, default=256)
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--output_nc', type=int, default=3)
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--which_model_netD', type=str, default='basic')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256')
        self.parser.add_argument('--n_layers_D', type=int, default=3)
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='-1 for CPU')
        self.parser.add_argument('--name', type=str, default='testing')
        self.parser.add_argument('--dataset_mode', type=str, default='aligned')
        self.parser.add_argument('--model', type=str, default='cGAN')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=16, type=int)
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--norm', type=str, default='instance')
        self.parser.add_argument('--serial_batches', action='store_true')
        self.parser.add_argument('--display_winsize', type=int, default=256,)
        self.parser.add_argument('--no_dropout', action='store_true')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"))
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop')
        self.parser.add_argument('--no_flip', action='store_true')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
