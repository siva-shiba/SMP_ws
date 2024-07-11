"""deeplab v3+の学習プログラム.

PASCAL VOC2012形式のデータを学習する想定.

引数"data_root" : **/VOCdevkit/VOC2012/... というフォルダ構成を想定している ※torchvisionのVOCSegmentationを使うため

参考 : https://nujust.hatenablog.com/entry/2023/10/11/210700
"""
import argparse
import random
import pathlib
import datetime
import logging
import numpy as np

import segmentation_models_pytorch as smp
import sklearn.metrics
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", default="datasets/voc_origin", type=str, help="datasetのrootディレ(形式はpascal voc2012形式を想定)")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--backbone", type=str, default="tu-xception41")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--random_seed", type=int, default=1)

    return parser.parse_args()


def fix_seed(seed: int):
    """乱数テーブルの設定.
    
    Args:
        seed (int): 乱数シード
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(name, logfile):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)10s %(levelname)8s %(message)s")

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)

    filehander = logging.FileHandler(logfile)
    filehander.setFormatter(formatter)
    logger.addHandler(filehander)
    return logger


def pred_per_epoch(model, device, dataloader, criterion, num_classes, optimizer=None, scheduler=None):
    total_loss = 0.0
    preds_list = []
    labels_list = []

    with torch.set_grad_enabled(model.training):
        for images, labels in dataloader:
            images: torch.Tensor = images.to(device, dtype=torch.float32)
            labels: torch.Tensor = labels.to(device, dtype=torch.long)

            output = model(images)
            loss = criterion(output, labels)

            if model.training and not optimizer is None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred = output.max(dim=1)[1]
            total_loss += loss.item() * np.prod((images.shape[0], images.shape[2], images.shape[3]))
            labels_list.extend(labels.flatten().cpu().tolist())
            preds_list.extend(pred.flatten().detach().cpu().tolist())

        if model.training and not scheduler is None:
            scheduler.step()

    preds_list = np.array(preds_list).flatten()
    labels_list = np.array(labels_list).flatten()
    cm = sklearn.metrics.confusion_matrix(labels_list, preds_list, labels=np.arange(num_classes))
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    scores = {"loss": total_loss / len(preds_list), "miou": np.nanmean(iou)}
    return scores


class CenterCropTransforms:
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def __call__(self, image, label):
        image = TF.resize(image, [self.crop_size], transforms.InterpolationMode.BILINEAR)
        label = TF.resize(label, [self.crop_size], transforms.InterpolationMode.NEAREST)

        image = TF.center_crop(image, [self.crop_size, self.crop_size])
        label = TF.center_crop(label, [self.crop_size, self.crop_size])

        image = TF.to_tensor(image)
        label = torch.from_numpy(np.array(label, dtype=np.uint8))

        image = TF.normalize(image, self.normalize_mean, self.normalize_std)
        return image, label


def main(args):
    fix_seed(args.random_seed)
    num_classes = 21

    output_path = pathlib.Path("outputs", f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{args.name}")
    output_path.mkdir(parents=True)
    logger = setup_logger(__name__, output_path.joinpath("run.log"))

    # dataset load
    train_dataset = VOCSegmentation(root=args.data_root, image_set='train', 
                                    transforms=CenterCropTransforms(args.crop_size))
    val_dataset = VOCSegmentation(root=args.data_root, image_set="val", 
                                  transforms=CenterCropTransforms(args.crop_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True)
    logger.info("VOC Dataset, Train set: %d, Val set: %d", len(train_dataset), len(val_dataset))

    # モデルの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s from %s devices", torch.cuda.current_device(), torch.cuda.device_count())
    else:
        device = torch.device("cpu")
    model = smp.DeepLabV3Plus(
        encoder_name=args.backbone,
        encoder_output_stride=args.output_stride,
        classes=num_classes)
    model.to(device)

    # 損失関数とオプティマイザーの設定
    optimizer_params = [
        dict(params=model.encoder.parameters(), lr=0.1 * args.lr),
        dict(params=model.decoder.parameters(), lr=args.lr),
        dict(params=model.segmentation_head.parameters(), lr=args.lr),
    ]
    optimizer = optim.SGD(params=optimizer_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 訓練ループ

    best_score = 0.0
    for epoch in range(1, 1 + args.epochs):
        model.train()
        train_scores = pred_per_epoch(
            model=model,
            device=device,
            dataloader=train_loader,
            criterion=criterion,
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        logger.info("Epoch %3d, Train, loss %5f, mIoU %5f", epoch, train_scores["loss"], train_scores["miou"])

        model.eval()
        val_scores = pred_per_epoch(
            model=model,
            device=device,
            dataloader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
        )
        logger.info("Epoch %3d, Val  , loss %5f, mIoU %5f", epoch, val_scores["loss"], val_scores["miou"])

        if val_scores["miou"] >= best_score:
            torch.save(model.state_dict(), output_path.joinpath("model.pth"))
            best_score = val_scores["miou"]


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
