from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.lrmrec import model


# Refine data path depend your own need
dataset.dataset="referit"
dataset.ann_path["referit"] = "data/anns/refclef/refclef.json"
dataset.image_path["referit"] = "data/images/refclef"
dataset.mask_path["referit"] = "data/masks/refclef"

# Refine training cfg
train.epochs=35
train.output_dir = "./output/referit_627_tsrm notin bb_ms_assf_3sf"
train.batch_size = 16
train.save_period = 20
train.log_period = 10
train.evaluation.eval_batch_size = 16
train.sync_bn.enabled = False
train.auto_resume.enabled = False

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.freeze_backbone = True
# model.visual_backbone.freeze_backbone = False
model.visual_backbone.pretrained_weight_path="data/weights/pretrained_weights/cspdarknet_coco.pth"
