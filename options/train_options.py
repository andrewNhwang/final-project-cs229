from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0)
        self.parser.add_argument('--update_html_freq', type=int, default=1000)
        self.parser.add_argument('--print_freq', type=int, default=100)
        self.parser.add_argument('--save_latest_freq', type=int, default=1000)
        self.parser.add_argument('--save_epoch_freq', type=int, default=5)
        self.parser.add_argument('--continue_train', action='store_true')
        self.parser.add_argument('--epoch_count', type=int, default=1)
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest')
        self.parser.add_argument('--niter', type=int, default=100)
        self.parser.add_argument('--niter_decay', type=int, default=100)
        self.parser.add_argument('--iter_num', type=int, default=0)
        self.parser.add_argument('--max_iter_num', type=int, default=1000)
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--no_lsgan', action='store_true')
        self.parser.add_argument('--lambda_AB', type=float, default=10.0)
        self.parser.add_argument('--lambda_A', type=float, default=10.0)
        self.parser.add_argument('--lambda_B', type=float, default=10.0)
        self.parser.add_argument('--pool_size', type=int, default=50)
        self.parser.add_argument('--no_html', action='store_true')
        self.parser.add_argument('--lr_policy', type=str, default='lambda')
        self.parser.add_argument('--geometry', type=str, default='rot')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50)
        self.parser.add_argument('--identity', type=float, default=0.5)
        self.parser.add_argument('--lambda_gc', type=float, default=2.0)
        self.parser.add_argument('--lambda_G', type=float, default=1.0)

        self.isTrain = True
