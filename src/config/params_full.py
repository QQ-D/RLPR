import os
from config.params_nusc import ModelParams as NuscParams
from config.params_mulran import ModelParams as MulranParams
from config.params_boreas import ModelParams as BoreasParams
from config.params_snail import ModelParams as SnailParams
from config.params_hercules import ModelParams as HerculesParams


class ModelParamsFull:
    def __init__(self):
        self.l_enable = True
        self.c_enable = False
        self.r_enable = True

        self.nusc_param = NuscParams()
        self.mulran_param = MulranParams()
        self.boreas_param = BoreasParams()
        self.snail_param = SnailParams()
        self.hercules_param = HerculesParams()

        self.checkpoint_path = ""
        self.training_root = os.environ.get("RLPR_TRAINING_ROOT", "./outputs/experiment")

        if self.training_root:
            for subdir in ("weights", "logs", "cache"):
                path = os.path.join(self.training_root, subdir)
                os.makedirs(path, exist_ok=True)
            self.weights_root = os.path.join(self.training_root, "weights")
            self.logs_root = os.path.join(self.training_root, "logs")
            self.cache_root = os.path.join(self.training_root, "cache")

        self.pretrain_r_path = os.environ.get("RLPR_PRETRAIN_R", "")
        self.pretrain_l_path = os.environ.get("RLPR_PRETRAIN_L", "")
        self.snail_crossmodal_path = os.environ.get("RLPR_SNAIL_CROSS_CHECKPOINT", "")

        self.pos_num = 1
        self.neg_num = 10
        self.resume_checkpoint = True
        self.resume_epoch = 1
        self.epochs = 5
        self.learning_rate = 0.00005
        self.loss_margin = 0.5
        self.num_workers = 1
        self.batch_size = 8
        self.num_test_workers = 1
        self.test_batch_size = 1
        self.output_dim = 512
