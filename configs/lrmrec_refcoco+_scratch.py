from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.lrmrec import model


# Refine data path depend your own need
dataset.dataset="refcoco+"
dataset.ann_path["refcoco+"] = "data/anns/refcoco+/refcoco+.json"
dataset.image_path["refcoco+"] = "data/images/train2014"
dataset.mask_path["refcoco+"] = "data/masks/refcoco+"

# Refine training cfg
train.epochs=35
train.output_dir = "./output/ref+_530_tsrm notin bb_notasff_ms2"
train.batch_size = 32
train.save_period = 20
train.log_period = 10
train.evaluation.eval_batch_size = 32
train.sync_bn.enabled = False
train.auto_resume.enabled = False

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.freeze_backbone = True
# model.visual_backbone.freeze_backbone = False
model.visual_backbone.pretrained_weight_path="data/weights/pretrained_weights/cspdarknet_coco.pth"
