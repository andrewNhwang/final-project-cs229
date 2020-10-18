from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"))
        self.parser.add_argument('--results_dir', type=str, default='./results/')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--which_epoch', type=str, default='latest')
        self.parser.add_argument('--geometry', type=str, default='rot')
        self.parser.set_defaults(dataset_mode='single')
        self.isTrain = False
