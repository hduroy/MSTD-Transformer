from __future__ import print_function
import datetime
import os
import time
import sys
import numpy as np
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import utils

from scheduler import WarmupMultiStepLR

from datasets.msr import MSRAction3D


from models.CLR_Model import ContrastiveLearningModel


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def train_one_epoch(
    model,
    criterion,
    criterion2,
    optimizer,
    lr_scheduler,
    data_loader,
    device,
    epoch,
    print_freq,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "clips/s", utils.SmoothedValue(window_size=10, fmt="{value:.3f}")
    )

    header = "Epoch: [{}]".format(epoch)
    for clip, target, _ in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()

        clip, target = clip.to(device), target.to(device)

        output = model(clip)

        loss1 = criterion(output, target)

        loss = loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = clip.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["clips/s"].update(batch_size / (time.time() - start_time))
        lr_scheduler.step()
        sys.stdout.flush()


def evaluate(model, criterion, criterion2, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    video_prob = {}
    video_label = {}
    with torch.no_grad():
        for clip, target, video_idx in metric_logger.log_every(
            data_loader, 100, header
        ):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(clip)

            loss1 = criterion(output, target)

            loss = loss1

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            prob = F.softmax(input=output, dim=1)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = clip.shape[0]
            target = target.cpu().numpy()
            video_idx = video_idx.cpu().numpy()
            prob = prob.cpu().numpy()
            for i in range(0, batch_size):
                idx = video_idx[i]
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = target[i]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        " * Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5
        )
    )

    # video level prediction
    video_pred = {k: np.argmax(v) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
    total_acc = np.mean(pred_correct)

    class_count = [0] * data_loader.dataset.num_classes
    class_correct = [0] * data_loader.dataset.num_classes

    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += v == label
    class_acc = [c / float(s) for c, s in zip(class_correct, class_count)]

    print(" * Video Acc@1 %f" % total_acc)
    print(" * Class Acc@1 %s" % str(class_acc))

    return total_acc


def main(args):

    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    # Data loading code
    print("Loading data")

    st = time.time()

    dataset = MSRAction3D(
        root=args.data_path,
        frames_per_clip=args.clip_len,
        step_between_clips=1,
        num_points=args.num_points,
        train=True,
    )

    dataset_test = MSRAction3D(
        root=args.data_path,
        frames_per_clip=args.clip_len,
        step_between_clips=1,
        num_points=args.num_points,
        train=False,
    )

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
  
    model = ContrastiveLearningModel(
        radius=args.radius,
        nsamples=args.nsamples,
        spatial_stride=args.spatial_stride,
        temporal_kernel_size=args.temporal_kernel_size,
        temporal_stride=args.temporal_stride,
        en_emb_dim=args.dim,
        en_depth=args.depth,
        en_heads=args.heads,
        en_head_dim=args.dim_head,
        en_mlp_dim=args.mlp_dim,
        num_classes=dataset.num_classes,
        mcm_ratio=args.mcm_ratio,
        dropout1=args.dropout1,
        dropout_cls=args.dropout2,
        pretraining=False,
        vis=False,
        vcm=True,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    print("===> Loading checkpoint for finetune '{}'".format(args.finetune))
    checkpoint = torch.load(args.finetune, map_location="cpu")
    state_dict = checkpoint["model"]

    #     for k in list(state_dict.keys()):
    # print(k)
    # if not k.startswith(('tube_embedding','encoder_pos_embed','encoder_transformer','encoder_norm','encoder_temp_Transformer')): #,'module.encoder_norm'
    #         if not k.startswith(('tube_embedding','encoder_transformer','encoder_temp_Transformer')):
    #             del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    print("missing_keys", log.missing_keys)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=1., gamma=2.)
    # criterion2 = Loss.FocalLoss(alpha=1., gamma=2.)
    criterion2 = Loss.KLloss()

    lr = args.lr
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # convert scheduler to be per iteration, not per epoch, for warmup that lasts
    # between different epochs
    warmup_iters = args.lr_warmup_epochs * len(data_loader)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=args.lr_gamma,
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    print("Start training")
    start_time = time.time()
    acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model,
            criterion,
            criterion2,
            optimizer,
            lr_scheduler,
            data_loader,
            device,
            epoch,
            args.print_freq,
        )

        acc = max(
            acc, evaluate(model, criterion, criterion2, data_loader_test, device=device)
        )

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "model_{}.pth".format(epoch))
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print("Accuracy {}".format(acc))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MSTD-Transformer Model Training")

    parser.add_argument(
        "--data-path", default="./data/processed_data", type=str, help="dataset"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--model", default="MSTDTransformer", type=str, help="model")
    # input
    parser.add_argument(
        "--clip-len",
        default=24,
        type=int,
        metavar="N",
        help="number of frames per clip",
    )
    parser.add_argument(
        "--num-points",
        default=2048,
        type=int,
        metavar="N",
        help="number of points per frame",
    )
    # P4D
    parser.add_argument(
        "--radius", default=0.3, type=float, help="radius for the ball query"
    )
    parser.add_argument(
        "--nsamples",
        default=32,
        type=int,
        help="number of neighbors for the ball query",
    )
    parser.add_argument(
        "--spatial-stride", default=32, type=int, help="spatial subsampling rate"
    )
    parser.add_argument(
        "--temporal-kernel-size", default=3, type=int, help="temporal kernel size"
    )
    parser.add_argument(
        "--temporal-stride", default=2, type=int, help="temporal stride"
    )
    # transformer
    parser.add_argument("--dim", default=80, type=int, help="transformer dim")
    parser.add_argument("--depth", default=5, type=int, help="transformer depth")
    parser.add_argument("--heads", default=2, type=int, help="transformer head")
    parser.add_argument(
        "--dim-head", default=40, type=int, help="transformer dim for each head"
    )
    parser.add_argument("--mlp-dim", default=160, type=int, help="transformer mlp dim")
    parser.add_argument(
        "--dropout1", default=0.0, type=float, help="transformer dropout"
    )
    # output
    parser.add_argument(
        "--dropout2", default=0.0, type=float, help="classifier dropout"
    )
    # training
    parser.add_argument("-b", "--batch-size", default=25, type=int)
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=10,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-milestones",
        nargs="+",
        default=[20, 30],
        type=int,
        help="decrease lr on milestones",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-warmup-epochs", default=10, type=int, help="number of warmup epochs"
    )
    # output
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir",
        default="./output_ssl_10%MSR",
        type=str,
        help="path where to save",
    )
    parser.add_argument(
        "--finetune",
        default="./log_finetune_10%MSR/MSR3D/model.pth",
        help="finetune from checkpoint",
    )
    # resume
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )


    parser.add_argument(
        "--mcm-ratio", default=0.4, type=float, metavar="N", help="dynamic mask  ratio"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
