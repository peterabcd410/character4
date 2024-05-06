import os
import time
import datetime
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import sys
sys.path.append('/media/new702/OS/rec/SimREC/')

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
os.environ['MASTER_ADDR']='127.0.0.2'
os.environ['MASTER_PORT']='23457'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from simrec.config import LazyConfig, instantiate
from simrec.datasets.dataloader import build_train_loader, build_test_loader
from simrec.scheduler.build import build_lr_scheduler
from simrec.utils.model_ema import EMA
from simrec.utils.logger import create_logger
from simrec.utils.env import seed_everything
from simrec.utils.metric import AverageMeter
from simrec.utils.distributed import reduce_meters, is_main_process, cleanup_distributed
from simrec.utils.checkpoint import save_checkpoint, load_checkpoint, auto_resume_helper

from gpueater.gpu_eater import occupy_gpus_mem

from tools.eval_engine import validate

torch.cuda.current_device()

def train_one_epoch(cfg, model, optimizer, scheduler, data_loader,scalar, writer, epoch, rank, model_without_ddp,best_det_acc,ema=None):

    print("epoch",epoch,"start","best_det_acc:",best_det_acc)
    # model.train()
    data_loader.sampler.set_epoch(epoch)
    
    # build validation dataset and dataloader
    cfg.dataset.split = "val"
    val_set = instantiate(cfg.dataset)
    val_loader = build_test_loader(
        cfg, 
        val_set,
        shuffle=False,
        drop_last=False,
    )

    save_ids = np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None

    num_iters = len(data_loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    losses_det = AverageMeter('LossDet', ':.4f')
    losses_seg = AverageMeter('LossSeg', ':.4f')
    meters = [batch_time, data_time, losses, losses_det, losses_seg]
    meters_dict = {meter.name: meter for meter in meters}
    
    start = time.time()
    end = time.time()

    val_num = 0

    flag = False#是否已经验证三次
    for idx, (ref_iter, image_iter, mask_iter, box_iter, gt_box_iter, mask_id, info_iter) in enumerate(data_loader):
        model.train()
        data_time.update(time.time() - end)
        # print(mask_iter.shape)#16,1,416,416
        ref_iter = ref_iter.cuda(non_blocking=True)
        image_iter = image_iter.cuda(non_blocking=True)
        mask_iter = mask_iter.cuda(non_blocking=True)
        box_iter = box_iter.cuda(non_blocking=True)

        if cfg.train.multi_scale_training:
            img_scales = cfg.train.multi_scale_training.img_scales
            h, w = img_scales[np.random.randint(0, len(img_scales))]
            image_iter = F.interpolate(image_iter, (h, w))
            mask_iter = F.interpolate(mask_iter, (h, w))

        if scalar is not None:
            with torch.cuda.amp.autocast():
                loss, loss_det, loss_seg = model(image_iter,ref_iter,det_label=box_iter,seg_label=mask_iter)
        else:
            loss, loss_det, loss_seg = model(image_iter, ref_iter, det_label=box_iter,seg_label=mask_iter)

        optimizer.zero_grad()
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            scalar.update()
        else:
            loss.backward()
            if cfg.train.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.train.clip_grad_norm
                )
            optimizer.step()
        scheduler.step()
        
        if ema is not None:
            ema.update_params()
        
        losses.update(loss.item(), image_iter.size(0))
        losses_det.update(loss_det.item(), image_iter.size(0))
        losses_seg.update(loss_seg.item(), image_iter.size(0))

        reduce_meters(meters_dict, rank, cfg)
        if is_main_process():

            global_step = epoch * num_iters + idx
            writer.add_scalar("loss/train", losses.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_det/train", losses_det.avg_reduce, global_step=global_step)
            writer.add_scalar("loss_seg/train", losses_seg.avg_reduce, global_step=global_step)

            
                
        if idx % cfg.train.log_period == 0 or idx==len(data_loader):
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_iters - idx)
            logger.info(
                f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{num_iters}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.7f}  '
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                f'Det Loss {losses_det.val:.4f} ({losses_det.avg:.4f})  '
                f'Seg Loss {losses_seg.val:.4f} ({losses_seg.avg:.4f})  '
                f'Mem {memory_used:.0f}MB')
        
        batch_time.update(time.time() - end)
        end = time.time()

        eval_steps = num_iters // 5
        
        if  epoch >= 20:
            
            #15轮后val3次
            if  idx!=0:
                if idx % eval_steps == 0 or idx==num_iters-1:
                    val_num=val_num+1
                    print("这是val",val_num)
                    box_ap, mask_ap = validate(cfg, model, val_loader, writer, epoch, val_set.ix_to_token, logger, dist.get_rank(), save_ids=save_ids, ema=ema)
            
                    # save checkpoints
                    # if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
                    logger.info(f"saving checkpoints......")
                    if ema is not None:
                        ema.apply_shadow()
                    # periodically save checkpoint
                    # save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
                    # save best checkpoint
                    if box_ap > best_det_acc:
                        best_det_acc = box_ap
                        save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, det_best=True)
                        print("epoch:",epoch,"  eval_step:",idx // eval_steps,"  best_det_acc:",best_det_acc)
                    # if mask_ap > best_seg_acc:
                    #     save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, seg_best=True)
                    if ema is not None:
                        ema.restore()
                    flag = True
                    logger.info(f"checkpoints saved !!!")

            if epoch % cfg.train.save_period == 0:
                save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger,save_at=True)

    if  flag is not True:
        box_ap, mask_ap = validate(cfg, model, val_loader, writer, epoch, val_set.ix_to_token, logger, dist.get_rank(), save_ids=save_ids, ema=ema)
        # save checkpoints
        # if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
        logger.info(f"saving checkpoints......")
        if ema is not None:
            ema.apply_shadow()
        # periodically save checkpoint
        # save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
        # save best checkpoint
        if box_ap > best_det_acc:
            best_det_acc = box_ap
            save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, det_best=True)
        # if mask_ap > best_seg_acc:
        #     save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, seg_best=True)
        if ema is not None:
            ema.restore()
        logger.info(f"checkpoints saved !!!")

    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return best_det_acc



def main(cfg):
    global best_det_acc, best_seg_acc
    best_det_acc, best_seg_acc = 0., 0.

    # build training dataset and dataloader
    cfg.dataset.split = "train"
    train_set = instantiate(cfg.dataset)
    train_loader = build_train_loader(
        cfg, 
        train_set, 
        shuffle=True,
        drop_last=True
    )
    print(train_loader)
    # # build validation dataset and dataloader
    # cfg.dataset.split = "val"
    # val_set = instantiate(cfg.dataset)
    # val_loader = build_test_loader(
    #     cfg, 
    #     val_set,
    #     shuffle=False,
    #     drop_last=False,
    # )

    # build model
    cfg.model.language_encoder.pretrained_emb = train_set.pretrained_emb
    cfg.model.language_encoder.token_size = train_set.token_size
    model = instantiate(cfg.model)

    # build optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optim.params = params
    optimizer = instantiate(cfg.optim)

    # model ema
    ema=None

    torch.cuda.set_device(dist.get_rank())
    if cfg.train.sync_bn.enabled:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted model to use Synchronized BatchNorm.")
    model = DistributedDataParallel(model.cuda(), device_ids=[dist.get_rank()], find_unused_parameters=True)
    model_without_ddp = model.module

    if is_main_process():
        total_params = sum([param.nelement() for param in model.parameters()])
        trainable_params = sum([param.nelement() for param in model.parameters() if param.requires_grad])
        logger.info(str(model))
        logger.info("Number of all params: %.2fM" % (total_params / 1e6))
        logger.info("Number of trainable params: %.2fM" % (trainable_params / 1e6))

    scheduler = build_lr_scheduler(cfg, optimizer, len(train_loader))

    start_epoch = 0

    if cfg.train.auto_resume.enabled:
        resume_file = auto_resume_helper(cfg.train.output_dir)
        if resume_file:
            if cfg.train.resume_path:
                logger.warning(f"auto-resume changing resume file from {cfg.train.resume_path} to {resume_file}")
            cfg.train.resume_path = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.train.output_dir}, ignoring auto resume')

    if cfg.train.resume_path:
        start_epoch = load_checkpoint(cfg, model_without_ddp, optimizer, scheduler, logger)

    if os.path.isfile(cfg.train.vl_pretrain_weight):
        checkpoint = torch.load(cfg.train.vl_pretrain_weight, map_location=lambda storage, loc: storage.cuda())
        logger.warning("loading pretrained weight for finetuning, ignoring resume training, reset start epoch to 0")
        msg = model_without_ddp.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info(msg)
        start_epoch = 0
        logger.info("==> loaded checkpoint from {}\n".format(cfg.train.vl_pretrain_weight) +
                    "==> epoch: {} lr: {} ".format(checkpoint['epoch'], checkpoint['lr']))

    if cfg.train.amp.enabled:
        assert torch.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.train.output_dir)
    else:
        writer = None

    # save_ids = np.random.randint(1, len(val_loader) * cfg.train.batch_size, 100) if cfg.train.log_image else None

    for epoch in range(start_epoch, cfg.train.epochs):
        if cfg.train.ema.enabled and ema is None:
            ema = EMA(model, cfg.train.ema.alpha, cfg.train.ema.buffer_ema)
        best_det_acc = train_one_epoch(cfg, model, optimizer, scheduler, train_loader,scalar, writer, epoch, dist.get_rank(), model_without_ddp,best_det_acc,ema,)
        # box_ap, mask_ap = validate(cfg, model, val_loader, writer, epoch, val_set.ix_to_token, logger, dist.get_rank(), save_ids=save_ids, ema=ema)
        
        # # save checkpoints
        # if epoch % cfg.train.save_period == 0 or epoch == (cfg.train.epochs - 1):
        #     logger.info(f"saving checkpoints......")
        #     if is_main_process():
        #         if ema is not None:
        #             ema.apply_shadow()
        #         # periodically save checkpoint
        #         # save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger)
        #         # save best checkpoint
        #         if box_ap > best_det_acc:
        #             best_det_acc = box_ap
        #             save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, det_best=True)
        #         # if mask_ap > best_seg_acc:
        #         #     save_checkpoint(cfg, epoch, model_without_ddp, optimizer, scheduler, logger, seg_best=True)
        #         if ema is not None:
        #             ema.restore()
        #     logger.info(f"checkpoints saved !!!")

    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimREC")
    parser.add_argument('--config', type=str, required=True, default='/rec/SimREC/configs/simrec_refcoco+_scratch.py')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--local_rank", type=int, default=2, help='local rank for DistributedDataParallel')
    parser.add_argument("--gpus", help='lvisible gpus',type=str)
    args = parser.parse_args()
    cfg = LazyConfig.load(args.config)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    # Environments setting
    seed_everything(cfg.train.seed)

    # Distributed setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # occupy_gpus_mem()
    # print(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=cfg.train.ddp.backend, 
        init_method=cfg.train.ddp.init_method, 
        world_size=world_size, 
        rank=rank
    )
    torch.distributed.barrier()

    # Path setting
    output_dir = cfg.train.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=cfg.train.output_dir, dist_rank=dist.get_rank())

    if is_main_process():
        path = os.path.join(cfg.train.output_dir, "config.yaml")
        LazyConfig.save(cfg, path)

    main(cfg)