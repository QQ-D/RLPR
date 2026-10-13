import os

from config.params_full import ModelParamsFull
from dataset.Boreas.boreas_dataloader import make_dataloader as boreas_dataloader
from dataset.Hercules.hercules_dataloader import make_dataloader as hercules_dataloader
from dataset.MulRan.mulran_dataloader import make_dataloader as mulran_dataloader
from dataset.NuScenes.nuscenes_dataloader import make_dataloader as nusc_dataloader
from dataset.SnailRadar.snail_dataloader import make_dataloader as snail_dataloader
from modules.LCRFusionGCNExpLHFFC_225 import LCRFusionGCNExpL as LidarPolar
from modules.LCRFusionGCNExpLHFFCN_225 import LCRFusionGCNExpL as LidarCart
from modules.LCRFusionGCNExpRHFFC import LCRFusionGCNExpR as RadarPolar
from modules.LCRFusionGCNExpRHFFCN import LCRFusionGCNExpR as RadarCart
from modules.LR_cross_hffc_225_dis import LRCross as CrossPolar
from modules.LR_cross_hffcn_225_dis import LRCross as CrossCart

CROSS_MODAL = {
    "nusc": dict(model=CrossPolar, dataloader=nusc_dataloader, epochs=3),
    "mulran": dict(model=CrossPolar, dataloader=mulran_dataloader, epochs=1),
    "snail": dict(model=CrossCart, dataloader=snail_dataloader, epochs=4),
    "hercules": dict(
        model=CrossCart,
        dataloader=hercules_dataloader,
        epochs=1,
        hercules_mode=True,
    ),
}

UNIMODAL = {
    ("nusc", "l"): dict(model=LidarPolar, dataloader=nusc_dataloader, cache_name="nusc", hard_mining_from_epoch=0),
    ("nusc", "r"): dict(model=RadarPolar, dataloader=nusc_dataloader, cache_name="nusc", hard_mining_from_epoch=0),
    ("mulran", "l"): dict(model=LidarPolar, dataloader=mulran_dataloader, cache_name="mulran", hard_mining_from_epoch=0),
    ("mulran", "r"): dict(model=RadarPolar, dataloader=mulran_dataloader, cache_name="mulran", hard_mining_from_epoch=0),
    ("snail", "l"): dict(model=LidarCart, dataloader=snail_dataloader, cache_name="snail", hard_mining_from_epoch=3),
    ("snail", "r"): dict(model=RadarCart, dataloader=snail_dataloader, cache_name="snail", hard_mining_from_epoch=3),
}

TEST = {
    "nusc": dict(model=CrossPolar, dataloader=nusc_dataloader),
    "mulran": dict(model=CrossPolar, dataloader=mulran_dataloader),
    "snail": dict(model=CrossCart, dataloader=snail_dataloader),
    "hercules": dict(model=CrossCart, dataloader=hercules_dataloader),
    "boreas": dict(model=CrossPolar, dataloader=boreas_dataloader),
}

UNIMODAL_EPOCHS = 5


def build_params(epochs=None):
    params = ModelParamsFull()
    if epochs is not None:
        params.epochs = epochs
    elif os.environ.get("RLPR_EPOCHS"):
        params.epochs = int(os.environ["RLPR_EPOCHS"])
    if os.environ.get("RLPR_RESUME_EPOCH"):
        params.resume_epoch = int(os.environ["RLPR_RESUME_EPOCH"])
    if os.environ.get("RLPR_RESUME", "1") == "0":
        params.resume_checkpoint = False
    return params


def build_test_params():
    params = ModelParamsFull()
    if os.environ.get("RLPR_RESUME_EPOCH"):
        params.resume_epoch = int(os.environ["RLPR_RESUME_EPOCH"])
    if os.environ.get("RLPR_CHECKPOINT_PATH"):
        params.checkpoint_path = os.environ["RLPR_CHECKPOINT_PATH"]
    if os.environ.get("RLPR_TEST_ZERO_SHOT", "0") == "1":
        params.resume_checkpoint = False
    return params
