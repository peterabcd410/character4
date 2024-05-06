from simrec.config import LazyCall
from .common.dataset import dataset
from .common.train import train
from .common.optim import optim
from .common.models.lrmrec import model


# Refine data path depend your own need
dataset.dataset="refcocog"
dataset.ann_path["refcocog"] = "data/anns/refcocog/refcocog.json"
dataset.image_path["refcocog"] = "data/images/train2014"
dataset.mask_path["refcocog"] = "data/masks/refcocog"

# Refine training cfg
train.epochs=35
train.output_dir = "./output/refg_629_tsrm in bb_asff_ms2"
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
