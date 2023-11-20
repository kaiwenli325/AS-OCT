import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data_loading import BasicDataset
from loss import dice_loss, LandmarksMSELoss
from model import multitask_network


def train_net(net, device):
    # Set your data path
    train_data_path = ['', '', '']
    val_data_path = ['', '', '']
    train_dataset = BasicDataset(train_data_path, mode='train')
    val_dataset = BasicDataset(val_data_path, mode='val')

    loader_args = dict(batch_size=8, num_workers=12, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=True, drop_last=True, **loader_args)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = LandmarksMSELoss(False)
    criterion_seg = nn.CrossEntropyLoss()

    # 5. Begin training
    for epoch in tqdm(range(150), position=0, desc="Epoch", unit='img', leave=True, colour='green', ncols=100):
        net.train()
        for batch in tqdm(train_loader, position=1, desc="Batch", unit='img', leave=False, colour='red', ncols=100):
            images = batch['image']
            true_masks = batch['mask']
            true_heatmap = batch['heatmap']
            landmarks_vis = batch['landmarks_vis']

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            true_heatmap = true_heatmap.to(device=device, non_blocking=True)
            landmarks_vis = landmarks_vis.to(device=device, non_blocking=True)

            heatmap_pred, masks_pred = net(images)
            lm_loss_temp = criterion(heatmap_pred, true_heatmap, landmarks_vis) \
                   + dice_loss(heatmap_pred.float(), true_heatmap, multiclass=True)
            seg_loss_temp = dice_loss(F.softmax(masks_pred, dim=1).float(),
                             F.one_hot(true_masks, 6).permute(0, 3, 1, 2).float(),
                             multiclass=True) \
                             + criterion_seg(masks_pred, true_masks)

            loss = lm_loss_temp + seg_loss_temp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #  Evaluation round
        net.eval()
        num_val_batches = len(val_loader)
        val_seg_loss = 0
        val_lm_loss = 0

        for batch in val_loader:
            val_image = batch['image']
            val_true_masks = batch['mask']
            val_true_heatmap = batch['heatmap']
            val_landmarks_vis = batch['landmarks_vis']

            val_image = val_image.to(device=device, dtype=torch.float32)
            val_true_masks = val_true_masks.to(device=device, dtype=torch.long)
            val_true_heatmap = val_true_heatmap.to(device=device, non_blocking=True)
            val_landmarks_vis = val_landmarks_vis.to(device=device, non_blocking=True)

            with torch.no_grad():
                val_landmarks_pred, val_masks_pred = net(val_image)
                val_lm_loss += criterion(val_landmarks_pred, val_true_heatmap, val_landmarks_vis) \
                          + dice_loss(val_landmarks_pred.float(), val_true_heatmap, multiclass=True)
                val_seg_loss += dice_loss(F.softmax(val_masks_pred, dim=1).float(),
                                        F.one_hot(val_true_masks, 6).permute(0, 3, 1, 2).float(),
                                        multiclass=True) \
                                 + criterion_seg(val_masks_pred, val_true_masks)

        net.train()

        val_loss = (val_seg_loss + val_lm_loss) / num_val_batches

        scheduler.step()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = multitask_network(n_channels=3, n_seg=6, n_landmark=2)

    net = nn.DataParallel(net).to(device=device)

    train_net(net=net, device=device)


