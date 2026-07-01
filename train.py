#!/usr/bin/env python3
import os
import sys
import json
import glob
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
from scipy.ndimage import zoom
import cc3d
from tqdm import tqdm
import pandas as pd
import warnings
from datetime import datetime
import shutil
from torch.utils.data import Dataset, DataLoader
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

warnings.filterwarnings('ignore')


DEFAULT_CONFIG = {
    "_comment": "Brain hemorrhage segmentation training config",
    "_label_mapping": {
        "1": "IVH",
        "2": "SAH",
        "3": "Ventricle"
    },

    "data_dir": "/root/autodl-tmp/patients",
    "output_dir": "/root/autodl-tmp/training_output",
    "feature_csv_path": "features_train_val.csv",

    "batch_size": 2,
    "num_workers": 16,
    "train_split": 0.8,
    "split_seed": 42,
    "accumulation_steps": 4,
    "normalize": True,
    "augment": True,
    "cache_data": False,
    "target_size": (182, 218, 182),

    "num_classes": 3,
    "base_channels": 32,

    "num_epochs": 300,
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "loss_type": "tversky_focal",
    "tversky_alpha": 0.5,
    "tversky_beta": 0.5,
    "tversky_weight": 0.5,
    "focal_weight": 0.5,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,

    "class_weights": [1, 1, 1],

    "warmup_epochs": 20,
    "cosine_eta_min": 1e-6,

    "early_stopping_patience": 30,

    "save_interval": 5,
    "resume_from": None,
    "run_mode": "train",
    "save_val_predictions": False,
    "val_predictions_dir": None,
    "val_manifest_name": "val_predictions_manifest.csv",
    "val_report_name": "val_eval_detailed.xlsx",
    "export_threshold": 0.5,
    "export_min_size": 10,
    "small_hemorrhage_ml": 0.5,
    "voxel_volume_mm3": 1.0,

    "device": "cuda"
}


VAL_ONLY_RUNTIME_OVERRIDE_KEYS = {
    "run_mode", "resume_from", "data_dir", "output_dir", "num_workers", "device",
    "save_val_predictions", "val_predictions_dir", "val_manifest_name",
    "val_report_name", "export_threshold", "export_min_size", "cache_data",
    "batch_size", "normalize", "target_size", "small_hemorrhage_ml",
    "voxel_volume_mm3",
}


def merge_checkpoint_config_for_val_only(runtime_config, checkpoint_config):
    merged = dict(checkpoint_config or {})
    for key, value in runtime_config.items():
        if key in VAL_ONLY_RUNTIME_OVERRIDE_KEYS or key not in merged:
            merged[key] = value
    return merged


def load_checkpoint_config(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        return {}
    config = checkpoint.get('config', {})
    return config if isinstance(config, dict) else {}


def build_effective_config(config):
    runtime_config = dict(config)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for path_key in ("data_dir", "output_dir", "feature_csv_path", "resume_from", "val_predictions_dir"):
        path_value = runtime_config.get(path_key)
        if isinstance(path_value, str) and path_value and not os.path.isabs(path_value):
            runtime_config[path_key] = os.path.join(script_dir, path_value)

    if runtime_config.get('run_mode', 'train') != 'val_only':
        return runtime_config

    checkpoint_path = runtime_config.get('resume_from')
    if not checkpoint_path:
        raise ValueError("resume_from must be set to best.pth in val_only mode")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_config = load_checkpoint_config(checkpoint_path)
    if not checkpoint_config:
        print("Warning: no config found in checkpoint; using the current script config")
        return runtime_config

    print(f"val_only mode: loaded training config from checkpoint: {checkpoint_path}")
    return merge_checkpoint_config_for_val_only(runtime_config, checkpoint_config)


def convert_probabilities_to_label_map(probs, threshold=0.5, min_size=10):
    seg = (probs > threshold).astype(np.uint8)

    if min_size > 0:
        for c in range(seg.shape[0]):
            if seg[c].sum() == 0:
                continue
            seg[c] = cc3d.dust(
                seg[c].astype(np.uint8),
                threshold=min_size,
                connectivity=26,
                in_place=False
            ).astype(np.uint8)

    masked_probs = probs * seg
    label_map = masked_probs.argmax(axis=0).astype(np.uint8) + 1
    label_map[masked_probs.max(axis=0) <= 0] = 0
    return label_map


def volume_ml_to_voxels(volume_ml, voxel_volume_mm3=1.0):
    return int(round(volume_ml * 1000.0 / voxel_volume_mm3))


class BrainHemorrhageDataset(Dataset):
    """Dataset for paired image and segmentation NIfTI files."""

    def __init__(self, data_dir, num_classes=3, normalize=True, augment=False,
                 cache_data=False, target_size=(182, 218, 182), **kwargs):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.normalize = normalize
        self.augment = augment
        self.cache_data = cache_data
        self.target_size = target_size

        self.data_pairs = []
        all_niigz = sorted(glob.glob(os.path.join(data_dir, '*.nii.gz')))
        all_niigz = [f for f in all_niigz if not f.endswith('_seg.nii.gz')]
        print(f"Found {len(all_niigz)} image .nii.gz files")

        for nii_path in all_niigz:
            basename = os.path.basename(nii_path)
            name_no_ext = basename.replace('.nii.gz', '')
            seg_path = os.path.join(data_dir, name_no_ext + '_seg.nii.gz')

            if os.path.exists(seg_path):
                self.data_pairs.append({
                    'image': nii_path,
                    'label': seg_path,
                    'basename': name_no_ext,
                    'patient_id': self._extract_patient_id(name_no_ext),
                    'timepoint': self._extract_timepoint(name_no_ext),
                })

        print(f"Found {len(self.data_pairs)} complete image-label pairs")

        self.cached_data = {}
        if cache_data:
            print("Caching data in memory...")
            for idx, pair in enumerate(self.data_pairs):
                if idx % 50 == 0:
                    print(f"  Cached {idx}/{len(self.data_pairs)}")
                self.cached_data[idx] = self._load_data(idx)

    def _load_data(self, idx):
        pair = self.data_pairs[idx]

        image = sitk.ReadImage(pair['image'])
        image_array = sitk.GetArrayFromImage(image)
        original_shape = image_array.shape

        label_img = sitk.ReadImage(pair['label'])
        label_array = sitk.GetArrayFromImage(label_img)

        if len(label_array.shape) == 4:
            label_array = label_array[..., 0]

        if self.target_size is not None and image_array.shape != self.target_size:
            zoom_factors = [t / s for t, s in zip(self.target_size, image_array.shape)]
            image_array = zoom(image_array, zoom_factors, order=1)
            label_array = zoom(label_array, zoom_factors, order=0)

        labels = []
        for class_id in range(1, self.num_classes + 1):
            labels.append((label_array == class_id).astype(np.float32))
        label_array = np.stack(labels, axis=0)

        min_cc_size = 10
        for c in range(self.num_classes):
            if label_array[c].sum() == 0:
                continue
            label_array[c] = cc3d.dust(
                label_array[c].astype(np.uint8),
                threshold=min_cc_size, connectivity=26,
                in_place=False
            ).astype(np.float32)

        if self.normalize:
            image_array = np.clip(image_array, 0, 100).astype(np.float32) / 100.0
        else:
            image_array = image_array.astype(np.float32)

        return {
            'image': image_array,
            'label': label_array.astype(np.float32),
            'basename': pair['basename'],
            'patient_id': pair['patient_id'],
            'timepoint': pair['timepoint'],
            'shape': image_array.shape,
            'original_shape': original_shape,
            'image_path': pair['image'],
            'label_path': pair['label'],
        }

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if self.cache_data:
            data = self.cached_data[idx]
        else:
            data = self._load_data(idx)

        image = data['image']
        label = data['label']

        if self.augment:
            image, label = self._augment(image, label)

        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label).float()

        return {
            'image': image_tensor,
            'label': label_tensor,
            'basename': data['basename'],
            'patient_id': data['patient_id'],
            'timepoint': data['timepoint'],
            'image_path': data['image_path'],
            'label_path': data['label_path'],
            'original_shape': data['original_shape'],
        }

    @staticmethod
    def _extract_patient_id(name_no_ext):
        return name_no_ext[:-2] if name_no_ext.endswith('-1') else name_no_ext

    @staticmethod
    def _extract_timepoint(name_no_ext):
        return 'post' if name_no_ext.endswith('-1') else 'pre'

    def _augment(self, image, label):
        data_dict = {
            'data': image[np.newaxis, np.newaxis],
            'seg': label[np.newaxis]
        }

        mirror_tr = MirrorTransform(axes=(0, 1, 2))
        data_dict = mirror_tr(**data_dict)

        spatial_tr = SpatialTransform(
            patch_size=image.shape,
            do_elastic_deform=True,
            alpha=(0., 200.),
            sigma=(9., 13.),
            do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_z=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            do_scale=True,
            scale=(0.85, 1.25),
            border_mode_data='constant',
            border_mode_seg='constant',
            order_seg=0,
            random_crop=False,
            p_el_per_sample=0.2,
            p_rot_per_sample=0.2,
            p_scale_per_sample=0.2
        )
        data_dict = spatial_tr(**data_dict)

        noise_tr = GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15)
        data_dict = noise_tr(**data_dict)

        brightness_tr = BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)
        data_dict = brightness_tr(**data_dict)

        gamma_tr = GammaTransform(gamma_range=(0.7, 1.5), p_per_sample=0.15)
        data_dict = gamma_tr(**data_dict)

        image = data_dict['data'][0, 0]
        label = data_dict['seg'][0]

        return image, label


def load_patient_binary_labels(feature_csv_path):
    if not feature_csv_path:
        raise ValueError("feature_csv_path is required for mRS-stratified sampling")
    if not os.path.exists(feature_csv_path):
        raise FileNotFoundError(f"Feature table not found: {feature_csv_path}")

    df = pd.read_csv(feature_csv_path, encoding='utf-8-sig')
    required_cols = {'patient_id', 'mRS'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing_cols)}")

    label_map = {}
    for _, row in df[['patient_id', 'mRS']].dropna().iterrows():
        patient_id = str(row['patient_id']).strip()
        mrs = int(float(row['mRS']))
        label_map[patient_id] = 0 if mrs <= 2 else 1
    return label_map


def stratified_split_patient_ids(patient_ids, binary_labels, train_split=0.8, seed=42):
    if not patient_ids:
        raise ValueError("No patients are available for splitting")

    class_to_ids = defaultdict(list)
    for pid in patient_ids:
        class_to_ids[int(binary_labels[pid])].append(pid)

    rng = np.random.default_rng(seed)
    train_ids, val_ids = [], []
    for class_label, ids in sorted(class_to_ids.items()):
        ids = list(ids)
        rng.shuffle(ids)

        class_train_count = int(round(len(ids) * train_split))
        class_train_count = max(1, min(class_train_count, len(ids) - 1))
        train_ids.extend(ids[:class_train_count])
        val_ids.extend(ids[class_train_count:])

        print(
            f"Stratum {class_label}: total={len(ids)} "
            f"train={class_train_count} val={len(ids) - class_train_count}"
        )

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)
    return train_ids, val_ids


def create_data_loaders(data_dir, batch_size=1, num_workers=4,
                        train_split=0.8, normalize=True, augment=True,
                        cache_data=False, feature_csv_path=None,
                        split_seed=42, target_size=(182, 218, 182), **kwargs):
    train_dataset_full = BrainHemorrhageDataset(
        data_dir=data_dir,
        normalize=normalize,
        augment=augment,
        cache_data=cache_data,
        target_size=target_size,
    )

    val_dataset_full = BrainHemorrhageDataset(
        data_dir=data_dir,
        normalize=normalize,
        augment=False,
        cache_data=cache_data,
        target_size=target_size,
    )

    patient_label_map = load_patient_binary_labels(feature_csv_path)

    patient_to_indices = defaultdict(dict)
    for idx, pair in enumerate(train_dataset_full.data_pairs):
        patient_to_indices[pair['patient_id']][pair['timepoint']] = idx

    complete_patient_ids = []
    incomplete_patient_ids = []
    missing_label_patient_ids = []
    for patient_id, timepoint_map in sorted(patient_to_indices.items()):
        has_pre = 'pre' in timepoint_map
        has_post = 'post' in timepoint_map
        if not (has_pre and has_post):
            incomplete_patient_ids.append(patient_id)
            continue
        if patient_id not in patient_label_map:
            missing_label_patient_ids.append(patient_id)
            continue
        complete_patient_ids.append(patient_id)

    if incomplete_patient_ids:
        print(f"Warning: skipped {len(incomplete_patient_ids)} patients without both pre and post scans")
        print(f"  Examples: {incomplete_patient_ids[:10]}")
    if missing_label_patient_ids:
        print(f"Warning: skipped {len(missing_label_patient_ids)} patients without mRS labels")
        print(f"  Examples: {missing_label_patient_ids[:10]}")

    train_patient_ids, val_patient_ids = stratified_split_patient_ids(
        complete_patient_ids,
        patient_label_map,
        train_split=train_split,
        seed=split_seed,
    )

    train_indices = []
    for pid in train_patient_ids:
        train_indices.extend([
            patient_to_indices[pid]['pre'],
            patient_to_indices[pid]['post'],
        ])

    val_indices = []
    for pid in val_patient_ids:
        val_indices.extend([
            patient_to_indices[pid]['pre'],
            patient_to_indices[pid]['post'],
        ])

    def summarize_binary(ids, title):
        labels = [patient_label_map[pid] for pid in ids]
        good_count = int(sum(1 for x in labels if x == 0))
        poor_count = int(sum(1 for x in labels if x == 1))
        print(
            f"{title}: patients={len(ids)} images={len(ids) * 2} "
            f"(mRS 0-2: {good_count}, mRS 3-6: {poor_count})"
        )

    print(f"Complete paired patients with labels: {len(complete_patient_ids)}")
    summarize_binary(train_patient_ids, "Training set")
    summarize_binary(val_patient_ids, "Validation set")

    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn   = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            ConvBlock3D(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels + in_channels // 2, out_channels)
        else:
            self.up   = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, (diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2,
                        diff_d // 2, diff_d - diff_d // 2))
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net with optional deep supervision."""

    def __init__(self, in_channels=1, num_classes=3, base_channels=64,
                 bilinear=True, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        c = base_channels

        self.inc   = DoubleConv3D(in_channels, c)
        self.down1 = Down3D(c,      c * 2)
        self.down2 = Down3D(c * 2,  c * 4)
        self.down3 = Down3D(c * 4,  c * 8)
        self.down4 = Down3D(c * 8,  c * 16)

        self.up1 = Up3D(c * 16, c * 8,  bilinear)
        self.up2 = Up3D(c * 8,  c * 4,  bilinear)
        self.up3 = Up3D(c * 4,  c * 2,  bilinear)
        self.up4 = Up3D(c * 2,  c,      bilinear)
        self.outc = OutConv3D(c, num_classes)

        if self.deep_supervision:
            self.ds1 = OutConv3D(c * 8, num_classes)
            self.ds2 = OutConv3D(c * 4, num_classes)
            self.ds3 = OutConv3D(c * 2, num_classes)

    def forward(self, x):
        input_shape = x.shape[2:]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        logits = self.outc(d1)

        if self.training and self.deep_supervision:
            ds1 = F.interpolate(self.ds1(d4), size=input_shape, mode='trilinear', align_corners=True)
            ds2 = F.interpolate(self.ds2(d3), size=input_shape, mode='trilinear', align_corners=True)
            ds3 = F.interpolate(self.ds3(d2), size=input_shape, mode='trilinear', align_corners=True)
            return [logits, ds1, ds2, ds3]

        return logits


def create_model(in_channels=1, num_classes=3, base_channels=32,
                 device='cuda', deep_supervision=True):
    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        bilinear=True,
        deep_supervision=deep_supervision
    )
    return model.to(device)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, weight=None):
        super().__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        if self.weight is not None:
            dice_loss = dice_loss * self.weight.to(pred.device).unsqueeze(0)
        return dice_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, target):
        p = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss

        if self.weight is not None:
            focal_loss = focal_loss * self.weight.to(pred.device).view(1, -1, 1, 1, 1)

        return focal_loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.weight = weight

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        tp = (pred * target).sum(dim=(2, 3, 4))
        fp = (pred * (1 - target)).sum(dim=(2, 3, 4))
        fn = ((1 - pred) * target).sum(dim=(2, 3, 4))
        loss = 1.0 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.weight is not None:
            loss = loss * self.weight.to(pred.device).unsqueeze(0)

        return loss.mean()


class DeepSupervisionLoss(nn.Module):
    def __init__(self, base_loss, weights=None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [1.0, 0.5, 0.25, 0.125]

    def forward(self, preds, target):
        if isinstance(preds, list):
            total = sum(
                (self.weights[i] if i < len(self.weights) else self.weights[-1]) * self.base_loss(p, target)
                for i, p in enumerate(preds)
            )
            return total / sum(self.weights[:len(preds)])
        return self.base_loss(preds, target)


def _build_loss(config, class_weights):
    loss_type = config.get('loss_type', 'tversky_focal')

    if loss_type == 'tversky_focal':
        tversky = TverskyLoss(
            alpha=config.get('tversky_alpha', 0.3),
            beta=config.get('tversky_beta', 0.7),
            weight=class_weights
        )
        focal = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            weight=class_weights
        )
        tw = config.get('tversky_weight', 0.5)
        fw = config.get('focal_weight', 0.5)

        class _TverskyFocal(nn.Module):
            def forward(self, pred, target):
                return tw * tversky(pred, target) + fw * focal(pred, target)

        return _TverskyFocal()

    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=config.get('tversky_alpha', 0.3),
            beta=config.get('tversky_beta', 0.7),
            weight=class_weights
        )
    elif loss_type == 'dice':
        return DiceLoss(weight=class_weights)
    else:
        return nn.BCEWithLogitsLoss(reduction='none')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.run_mode = config.get('run_mode', 'train')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.output_dir     = config.get('output_dir', 'training_output')
        self.log_dir        = os.path.join(self.output_dir, 'logs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        for d in [self.output_dir, self.log_dir, self.checkpoint_dir]:
            os.makedirs(d, exist_ok=True)

        self._save_config()
        self.writer = SummaryWriter(self.log_dir)

        print("\nLoading data...")
        self.train_loader, self.val_loader = create_data_loaders(
            data_dir    = config['data_dir'],
            batch_size  = config.get('batch_size', 1),
            num_workers = config.get('num_workers', 4),
            train_split = config.get('train_split', 0.8),
            normalize   = config.get('normalize', True),
            augment     = config.get('augment', True),
            cache_data  = config.get('cache_data', False),
            feature_csv_path = config.get('feature_csv_path'),
            split_seed = config.get('split_seed', 42),
            target_size = tuple(config.get('target_size', (182, 218, 182))),
        )
        print(f"Training set size: {len(self.train_loader.dataset)}")
        print(f"Validation set size: {len(self.val_loader.dataset)}")

        print("\nCreating model...")
        self.model = create_model(
            in_channels   = 1,
            num_classes   = config.get('num_classes', 3),
            base_channels = config.get('base_channels', 32),
            device        = self.device
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr           = config.get('learning_rate', 1e-3),
            weight_decay = config.get('weight_decay', 1e-5)
        )

        _warmup   = config.get('warmup_epochs', 20)
        _total    = config.get('num_epochs', 300)
        _eta_min  = config.get('cosine_eta_min', 1e-6)
        _lr_max   = config.get('learning_rate', 1e-3)

        def _lr_lambda(epoch):
            if epoch < _warmup:
                return (epoch + 1) / max(_warmup, 1)
            else:
                t = (epoch - _warmup) / max(1, _total - _warmup)
                cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
                return _eta_min / _lr_max + (1.0 - _eta_min / _lr_max) * cos_val

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        class_weights = torch.tensor(
            config.get('class_weights', [1, 1, 1]), dtype=torch.float32
        ).to(self.device)
        print(f"\nClass weights:")
        print(f"  IVH (LabelValue=1):       {class_weights[0]:.1f}")
        print(f"  SAH (LabelValue=2):       {class_weights[1]:.1f}")
        print(f"  Ventricle (LabelValue=3): {class_weights[2]:.1f}")

        base_loss = _build_loss(config, class_weights)
        self.criterion = DeepSupervisionLoss(base_loss)

        self.accumulation_steps = config.get('accumulation_steps', 4)
        print(f"Gradient accumulation steps: {self.accumulation_steps}")

        self.scaler = torch.amp.GradScaler('cuda')

        self.start_epoch  = 0
        self.best_val_dice = 0.0
        self.best_epoch    = 0
        self.class_names   = ['IVH', 'SAH', 'Ventricle']
        self.val_predictions_dir = config.get('val_predictions_dir') or os.path.join(
            self.output_dir, 'val_predictions'
        )
        self.val_manifest_path = os.path.join(
            self.output_dir, config.get('val_manifest_name', 'val_predictions_manifest.csv')
        )
        self.val_report_path = os.path.join(
            self.output_dir, config.get('val_report_name', 'val_eval_detailed.xlsx')
        )

        self.csv_log_path = os.path.join(self.output_dir, 'training_log.csv')
        if self.run_mode == 'train':
            with open(self.csv_log_path, 'w') as f:
                f.write('epoch,train_loss,val_loss,lr,val_mean_dice,'
                        'val_IVH,val_SAH,val_Vent,is_best\n')

        resume_from = config.get('resume_from')
        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)

    def _save_config(self):
        path = os.path.join(self.output_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"Config saved: {path}")

    def _save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict(),
            'best_val_dice':        self.best_val_dice,
            'config':               self.config
        }
        latest = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(ckpt, latest)

        if is_best:
            shutil.copy(latest, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"Best model saved (epoch {epoch})")

        if epoch % self.config.get('save_interval', 5) == 0:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, f'epoch_{epoch:03d}.pth'))

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        try:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception:
            print("  Warning: incompatible scheduler state_dict; skipped")
        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.start_epoch   = ckpt['epoch'] + 1
        self.best_val_dice = ckpt.get('best_val_dice', 0.0)
        print(f"Checkpoint loaded (resuming from epoch {self.start_epoch})")

    @staticmethod
    def _remove_small_cc(pred_np, min_cc_size=10):
        return cc3d.dust(
            pred_np.astype(np.uint8), threshold=min_cc_size,
            connectivity=26, in_place=False
        ).astype(np.float32)

    def _evaluate_full(self, pred, target, basename=None):
        SMALL_THRESH = volume_ml_to_voxels(
            self.config.get('small_hemorrhage_ml', 0.5),
            self.config.get('voxel_volume_mm3', 1.0)
        )

        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        num_classes = self.config.get('num_classes', 3)
        batch_size  = pred_bin.shape[0]

        dice_large   = [[] for _ in range(num_classes)]
        fpr_bg       = [[] for _ in range(num_classes)]
        recall_small = [[] for _ in range(num_classes)]
        records      = []

        for b in range(batch_size):
            name = (basename[b] if isinstance(basename, (list, tuple)) else basename) if basename else f"sample[{b}]"

            for c in range(num_classes):
                target_np  = target[b, c].cpu().numpy()
                target_sum = target_np.sum()

                pred_np  = self._remove_small_cc(pred_bin[b, c].cpu().numpy(), min_cc_size=10)
                pred_sum = pred_np.sum()

                cls_name = self.class_names[c]

                if target_sum == 0:
                    has_fp = 1.0 if pred_sum > 0 else 0.0
                    fpr_bg[c].append(has_fp)
                    group = 'GT=0'
                    if has_fp and pred_sum > SMALL_THRESH:
                        print(f"  [Severe false positive] {name} | {cls_name}: Pred={int(pred_sum)} vox")

                    records.append({
                        'PatientID': name, 'Class': cls_name, 'Group': group,
                        'GTVoxels': 0, 'PredVoxels': int(pred_sum),
                        'FPR': has_fp, 'Recall': None, 'Dice': None
                    })

                elif target_sum <= SMALL_THRESH:
                    inter = (pred_np * target_np).sum()
                    detected = 1.0 if inter > 0 else 0.0
                    recall_small[c].append(detected)
                    group = 'GT_small'

                    records.append({
                        'PatientID': name, 'Class': cls_name, 'Group': group,
                        'GTVoxels': int(target_sum), 'PredVoxels': int(pred_sum),
                        'FPR': None, 'Recall': detected, 'Dice': None
                    })

                else:
                    inter = (pred_np * target_np).sum()
                    dice = 2.0 * inter / (pred_sum + target_sum + 1e-6) if (pred_sum + target_sum) > 0 else 0.0
                    dice_large[c].append(dice)
                    group = 'GT_large'

                    if pred_sum == 0:
                        print(f"  [Severe false negative] {name} | {cls_name}: GT={int(target_sum)} vox, Pred=0")

                    records.append({
                        'PatientID': name, 'Class': cls_name, 'Group': group,
                        'GTVoxels': int(target_sum), 'PredVoxels': int(pred_sum),
                        'FPR': None, 'Recall': None, 'Dice': round(dice, 4)
                    })

        metrics = {}
        for c, name in enumerate(self.class_names):
            metrics[f'{name}_dice']   = float(np.mean(dice_large[c]))   if dice_large[c]   else float('nan')
            metrics[f'{name}_fpr']    = float(np.mean(fpr_bg[c]))       if fpr_bg[c]       else float('nan')
            metrics[f'{name}_recall'] = float(np.mean(recall_small[c])) if recall_small[c] else float('nan')

        dice_scores = [metrics[f'{n}_dice'] for n in self.class_names]

        return dice_scores, metrics, records

    def _save_validation_predictions(self, outputs, batch):
        os.makedirs(self.val_predictions_dir, exist_ok=True)

        manifest_rows = []
        probs_batch = torch.sigmoid(outputs).float().detach().cpu().numpy()
        threshold = self.config.get('export_threshold', 0.5)
        min_size = self.config.get('export_min_size', 10)

        for b in range(probs_batch.shape[0]):
            image_path = batch['image_path'][b]
            label_path = batch['label_path'][b]
            basename = batch['basename'][b]

            original_image = sitk.ReadImage(image_path)
            original_shape = sitk.GetArrayFromImage(original_image).shape
            zoom_back = [s / t for s, t in zip(original_shape, probs_batch[b].shape[1:])]
            probs_original = np.stack([
                zoom(probs_batch[b, c], zoom_back, order=1)
                for c in range(probs_batch.shape[1])
            ], axis=0).astype(np.float32)

            label_map = convert_probabilities_to_label_map(
                probs_original,
                threshold=threshold,
                min_size=min_size
            )

            pred_path = os.path.join(self.val_predictions_dir, f'{basename}_seg.nii.gz')
            pred_img = sitk.GetImageFromArray(label_map)
            pred_img.CopyInformation(original_image)
            sitk.WriteImage(pred_img, pred_path)

            manifest_rows.append({
                'split': 'val',
                'PatientID': basename,
                'image_path': image_path,
                'gt_path': label_path,
                'pred_path': pred_path,
                'original_shape': str(tuple(original_shape)),
            })

        return manifest_rows

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss  = 0.0
        num_batches = 0

        num_epochs = self.config.get('num_epochs', 100)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            with torch.amp.autocast('cuda'):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels) / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss  += loss.item() * self.accumulation_steps
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.4f}'})

        avg_loss = total_loss / num_batches
        self.writer.add_scalar('Loss/train', avg_loss, epoch)

        return avg_loss

    def _validate(self, epoch, save_predictions=False):
        self.model.eval()
        total_loss  = 0.0
        num_batches = 0
        num_classes = self.config.get('num_classes', 3)
        all_records = []
        manifest_rows = []

        dice_acc    = [[] for _ in range(num_classes)]
        fpr_acc     = [[] for _ in range(num_classes)]
        recall_acc  = [[] for _ in range(num_classes)]

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss    = self.criterion(outputs, labels)

                total_loss  += loss.item()
                num_batches += 1

                dice_scores, metrics, records = self._evaluate_full(
                    outputs, labels, basename=batch['basename'])
                for r in records:
                    r['Epoch'] = epoch + 1
                all_records.extend(records)

                if save_predictions:
                    manifest_rows.extend(self._save_validation_predictions(outputs, batch))

                for c, name in enumerate(self.class_names):
                    d = metrics[f'{name}_dice']
                    if not np.isnan(d):
                        dice_acc[c].append(d)
                    f = metrics[f'{name}_fpr']
                    if not np.isnan(f):
                        fpr_acc[c].append(f)
                    rc = metrics[f'{name}_recall']
                    if not np.isnan(rc):
                        recall_acc[c].append(rc)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'IVH':  f'{dice_scores[0]:.3f}' if not np.isnan(dice_scores[0]) else '-',
                    'SAH':  f'{dice_scores[1]:.3f}' if not np.isnan(dice_scores[1]) else '-',
                    'Vent': f'{dice_scores[2]:.3f}' if not np.isnan(dice_scores[2]) else '-',
                })

        avg_loss = total_loss / num_batches

        val_metrics = {}
        dice_avgs = []
        for c, name in enumerate(self.class_names):
            d = float(np.mean(dice_acc[c])) if dice_acc[c] else float('nan')
            dice_avgs.append(d)
            val_metrics[f'{name}_dice']   = d
            val_metrics[f'{name}_fpr']    = float(np.mean(fpr_acc[c]))    if fpr_acc[c]    else float('nan')
            val_metrics[f'{name}_recall'] = float(np.mean(recall_acc[c])) if recall_acc[c] else float('nan')

        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        for c, name in enumerate(self.class_names):
            self.writer.add_scalar(f'Dice/val_{name}', val_metrics[f'{name}_dice'], epoch)
            self.writer.add_scalar(f'FPR/val_{name}', val_metrics[f'{name}_fpr'], epoch)
            self.writer.add_scalar(f'Recall/val_{name}', val_metrics[f'{name}_recall'], epoch)

        return avg_loss, dice_avgs, val_metrics, all_records, manifest_rows

    def train(self):
        print("\n" + "="*60)
        print("Start training")
        print("="*60)

        num_epochs = self.config.get('num_epochs', 100)
        patience   = self.config.get('early_stopping_patience', 30)

        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self._train_epoch(epoch)
            val_loss, val_dice, val_metrics, val_records, _ = self._validate(epoch)

            if val_records:
                df = pd.DataFrame(val_records)
                cols = ['Epoch', 'PatientID', 'Class', 'Group', 'GTVoxels', 'PredVoxels',
                        'FPR', 'Recall', 'Dice']
                df = df[[c for c in cols if c in df.columns]]
                df.to_excel(os.path.join(self.output_dir, f'eval_report_epoch{epoch+1}.xlsx'), index=False)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_rate', lr, epoch)

            valid_val_dice = [d for d in val_dice if not np.isnan(d)]
            val_mean_dice = float(np.mean(valid_val_dice)) if valid_val_dice else 0.0

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  LR: {lr:.6f}")
            print("  GT>0.5ml Dice")
            for c, name in enumerate(self.class_names):
                d = val_metrics.get(f'{name}_dice', float('nan'))
                print(f"    {name}: val_Dice={d:.4f}")
            print(f"    Mean Dice: {val_mean_dice:.4f}")
            print("  GT=0 FPR")
            for c, name in enumerate(self.class_names):
                f = val_metrics.get(f'{name}_fpr', float('nan'))
                print(f"    {name}: FPR={f:.4f}" if not np.isnan(f) else f"    {name}: FPR=-")
            print("  GT in (0,0.5ml] Recall")
            for c, name in enumerate(self.class_names):
                rc = val_metrics.get(f'{name}_recall', float('nan'))
                print(f"    {name}: Recall={rc:.4f}" if not np.isnan(rc) else f"    {name}: Recall=-")

            is_best = val_mean_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_mean_dice
                self.best_epoch    = epoch

            self._save_checkpoint(epoch, is_best=is_best)

            with open(self.csv_log_path, 'a') as f:
                f.write(f'{epoch+1},{train_loss:.6f},{val_loss:.6f},{lr:.8f},{val_mean_dice:.6f},'
                        f'{val_dice[0]:.6f},{val_dice[1]:.6f},{val_dice[2]:.6f},'
                        f'{"Y" if is_best else "N"}\n')

            if epoch - self.best_epoch >= patience:
                print(f"\nEarly stopping: mean Dice did not improve for {patience} epochs")
                break

        print("\n" + "="*60)
        print(f"Training complete. Best model: epoch {self.best_epoch+1}, Mean Dice: {self.best_val_dice:.4f}")
        print("="*60)
        self.writer.close()

    def validate_only(self):
        if not self.config.get('resume_from'):
            raise ValueError("resume_from must be set to best.pth in val_only mode")

        eval_epoch = max(self.start_epoch - 1, 0)
        print("\n" + "=" * 60)
        print("Start validation-only mode")
        print("=" * 60)
        print(f"Validation checkpoint: {self.config['resume_from']}")
        print(f"Validation prediction export: {'enabled' if self.config.get('save_val_predictions', False) else 'disabled'}")
        if self.config.get('save_val_predictions', False):
            print(f"Prediction output dir: {self.val_predictions_dir}")

        val_loss, val_dice, val_metrics, val_records, manifest_rows = self._validate(
            eval_epoch,
            save_predictions=self.config.get('save_val_predictions', False)
        )

        if val_records:
            df = pd.DataFrame(val_records)
            cols = ['Epoch', 'PatientID', 'Class', 'Group', 'GTVoxels', 'PredVoxels', 'FPR', 'Recall', 'Dice']
            df = df[[c for c in cols if c in df.columns]]
            df.to_excel(self.val_report_path, index=False)
            print(f"Detailed validation report saved: {self.val_report_path}")

        if manifest_rows:
            pd.DataFrame(manifest_rows).to_csv(self.val_manifest_path, index=False)
            print(f"Validation prediction manifest saved: {self.val_manifest_path}")

        valid_val_dice = [d for d in val_dice if not np.isnan(d)]
        val_mean_dice = float(np.mean(valid_val_dice)) if valid_val_dice else 0.0

        print(f"\nValidation loss: {val_loss:.4f}")
        print("GT>0.5ml Dice")
        for name in self.class_names:
            d = val_metrics.get(f'{name}_dice', float('nan'))
            print(f"  {name}: val_Dice={d:.4f}" if not np.isnan(d) else f"  {name}: val_Dice=-")
        print(f"  Mean Dice: {val_mean_dice:.4f}")
        print("GT=0 FPR")
        for name in self.class_names:
            f = val_metrics.get(f'{name}_fpr', float('nan'))
            print(f"  {name}: FPR={f:.4f}" if not np.isnan(f) else f"  {name}: FPR=-")
        print("GT in (0,0.5ml] Recall")
        for name in self.class_names:
            rc = val_metrics.get(f'{name}_recall', float('nan'))
            print(f"  {name}: Recall={rc:.4f}" if not np.isnan(rc) else f"  {name}: Recall=-")

        self.writer.close()


if __name__ == "__main__":
    config = DEFAULT_CONFIG.copy()

    # Set to "train" for model training or "val_only" for checkpoint evaluation.
    config["run_mode"]             = "train"

    # Paths can be absolute or relative to this script.
    config["data_dir"]             = "patients"
    config["output_dir"]           = "training_output"
    config["feature_csv_path"]     = "features_train_val.csv"

    # Core training settings.
    config["batch_size"]           = 2
    config["split_seed"]           = 42
    config["target_size"]          = (182, 218, 182)
    config["num_epochs"]           = 300
    config["warmup_epochs"]        = 20

    # Use a best.pth path here when resuming or running validation only.
    config["resume_from"]          = None

    # Export validation predictions for offline metric calculation.
    config["save_val_predictions"] = True
    config["val_predictions_dir"]  = None
    config["val_manifest_name"]    = "val_predictions_manifest.csv"
    config["val_report_name"]      = "val_eval_detailed.xlsx"
    config["export_threshold"]     = 0.5
    config["export_min_size"]      = 10
    config["small_hemorrhage_ml"]  = 0.5
    config["voxel_volume_mm3"]     = 1.0

    # Resolve relative paths and optionally merge checkpoint training settings.
    config = build_effective_config(config)

    trainer = Trainer(config)
    if config.get("run_mode", "train") == "val_only":
        trainer.validate_only()
    else:
        trainer.train()
