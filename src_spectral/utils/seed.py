import os
import random

import numpy as np
import torch


def set_global_seed(seed):
    if os.environ.get('RLPR_DETERMINISTIC', '0') == '1':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Strict reproducibility requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before launch')
        torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
