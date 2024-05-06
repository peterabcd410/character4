from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.simrec import model


# Refine data path depend your own need
dataset.ann_path["refcoco"] = "data/anns/refcoco/refcoco.json"
dataset.image_path["refcoco"] = "data/images/train2014"
dataset.mask_path["refcoco"] = "data/masks/refcoco"

# Refine training cfg
train.epochs=35
train.output_dir = "./output/ref_20240409"
train.batch_size = 16
train.save_period = 10
train.log_period = 10
train.evaluation.eval_batch_size = 16
train.sync_bn.enabled = False
train.auto_resume.enabled = False

# Refine optim
optim.lr = train.base_lr

# Refine model cfg
model.visual_backbone.pretrained = True
model.visual_backbone.freeze_backbone = False
model.visual_backbone.pretrained_weight_path="data/weights/pretrained_weights/cspdarknet_coco.pth"
