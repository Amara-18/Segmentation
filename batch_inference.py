#!/usr/bin/env python3
"""Batch inference for NIfTI segmentation outputs."""
import os
import time
import glob
import json
from contextlib import nullcontext
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm
from datetime import datetime

from train import create_model, convert_probabilities_to_label_map


def build_prediction_filename(basename, suffix=None):
    if suffix:
        return f"{basename}_{suffix}_seg.nii.gz"
    return f"{basename}_seg.nii.gz"


def build_output_path(image_path, output_dir=None, output_name_suffix=None):
    basename = os.path.basename(image_path).replace('.nii.gz', '')
    output_filename = build_prediction_filename(basename, output_name_suffix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, output_filename)
    return os.path.join(os.path.dirname(image_path), output_filename)


class BatchOutputInference:
    """Whole-volume batch inference."""

    def __init__(self, model_path, device='cuda', num_classes=3,
                 base_channels=32, target_size=(182, 218, 182)):
        print("="*80)
        print("Initializing batch inference")
        print("="*80)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = self.device.type == 'cuda'

        checkpoint = torch.load(model_path, map_location=self.device)
        checkpoint_config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}

        self.num_classes = checkpoint_config.get('num_classes', num_classes)
        self.target_size = tuple(checkpoint_config.get('target_size', target_size))
        self.base_channels = checkpoint_config.get('base_channels', base_channels)
        self.default_threshold = checkpoint_config.get('export_threshold', 0.5)
        self.default_min_size = checkpoint_config.get('export_min_size', 10)

        self.model = create_model(
            in_channels      = 1,
            num_classes      = self.num_classes,
            base_channels    = self.base_channels,
            device           = self.device,
            deep_supervision = True
        )
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"Model loaded: {model_path}")
        print(f"Inference size: {self.target_size}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Base channels: {self.base_channels}")
        print(f"Default threshold: {self.default_threshold}")
        print(f"Default minimum connected component size: {self.default_min_size}")
        print(f"Device: {self.device}")
        print("="*80 + "\n")

    def infer_single_file(self, image_path, threshold=None,
                          postprocess=True, min_size=None, output_dir=None,
                          output_name_suffix=None):
        """Run inference for one .nii.gz file."""
        basename = os.path.basename(image_path).replace('.nii.gz', '')
        if threshold is None:
            threshold = self.default_threshold
        if min_size is None:
            min_size = self.default_min_size
        output_path = build_output_path(
            image_path=image_path,
            output_dir=output_dir,
            output_name_suffix=output_name_suffix,
        )

        if os.path.exists(output_path):
            print(f"  Skip existing output: {basename}")
            return output_path, False

        print(f"  Processing: {basename}")

        original_image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(original_image).astype(np.float32)
        if image_array.ndim == 4 and 1 in image_array.shape:
            image_array = np.squeeze(image_array)
        if image_array.ndim != 3:
            raise ValueError(f"Only 3D volumes are supported, got shape={image_array.shape}")
        original_shape = image_array.shape

        zoom_in = [t / s for t, s in zip(self.target_size, original_shape)]
        img_resize = zoom(image_array, zoom_in, order=1)

        img_norm = np.clip(img_resize, 0, 100) / 100.0

        x = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)

        amp_context = torch.amp.autocast(device_type='cuda') if self.use_amp else nullcontext()
        with torch.no_grad():
            with amp_context:
                output = self.model(x)

        probs = torch.sigmoid(output).float().squeeze(0).cpu().numpy()

        zoom_back = [s / t for s, t in zip(original_shape, self.target_size)]
        probs_original = np.stack([
            zoom(probs[c], zoom_back, order=1)
            for c in range(self.num_classes)
        ], axis=0).astype(np.float32)

        effective_min_size = min_size if postprocess else 0
        label_map = convert_probabilities_to_label_map(
            probs_original,
            threshold=threshold,
            min_size=effective_min_size
        )

        seg_img = sitk.GetImageFromArray(label_map)
        seg_img.CopyInformation(original_image)
        sitk.WriteImage(seg_img, output_path)

        class_names = ['IVH', 'SAH', 'Ventricle']
        print(f"    Saved: {os.path.basename(output_path)}")
        for c in range(self.num_classes):
            cnt = int((label_map == c + 1).sum())
            if cnt > 0:
                print(f"      {class_names[c]}: {cnt} voxels")

        return output_path, True

    def infer_directory(self, dir_path, threshold=None,
                        postprocess=True, min_size=None, output_dir=None,
                        output_name_suffix=None, report_name_suffix=None):
        """Run inference for all images in one directory."""
        dir_name = os.path.basename(dir_path)
        print("\n" + "="*80)
        print(f"Processing directory: {dir_name}")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print("="*80)

        if output_name_suffix:
            print(f"  Output filename pattern: *_{output_name_suffix}_seg.nii.gz")
        else:
            print("  Output filename pattern: *_seg.nii.gz")

        all_files = sorted(glob.glob(os.path.join(dir_path, '*.nii.gz')))
        nii_files = [
            f for f in all_files
            if not (f.endswith('_seg.nii.gz') or f.endswith('_segmentation.nii.gz'))
        ]
        print(f"Found {len(nii_files)} image files")

        start_time = time.time()
        results = {
            'directory':  dir_path,
            'output_dir': output_dir or dir_path,
            'dir_name':   dir_name,
            'total_files': len(nii_files),
            'success': 0, 'skipped': 0,
            'failed': [], 'output_files': [],
            'start_time': datetime.now().isoformat(),
        }

        for image_path in tqdm(nii_files, desc=f"Inference {dir_name}"):
            try:
                output_path, created = self.infer_single_file(
                    image_path  = image_path,
                    threshold   = threshold,
                    postprocess = postprocess,
                    min_size    = min_size,
                    output_dir  = output_dir,
                    output_name_suffix = output_name_suffix,
                )
                if os.path.exists(output_path):
                    if created:
                        results['success'] += 1
                    else:
                        results['skipped'] += 1
                    results['output_files'].append(output_path)
            except Exception as e:
                print(f"  Failed: {os.path.basename(image_path)} - {e}")
                results['failed'].append({'file': image_path, 'error': str(e)})

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
        results.update({
            'end_time':         datetime.now().isoformat(),
            'processing_time':  f"{h}h {m}m {s}s",
            'elapsed_seconds':  elapsed,
        })

        report_base_dir = output_dir or dir_path
        os.makedirs(report_base_dir, exist_ok=True)
        report_suffix = f'_{report_name_suffix}' if report_name_suffix else ''
        report_path = os.path.join(
            report_base_dir,
            f'{dir_name}{report_suffix}_inference_report.json'
        )
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nDirectory {dir_name} complete: created={results['success']}, "
              f"skipped={results['skipped']}, failed={len(results['failed'])}, "
              f"time={results['processing_time']}")
        return results

    def infer_all_directories(self, base_dirs, threshold=None,
                              postprocess=True, min_size=None,
                              output_dir=None, summary_path=None,
                              output_name_suffix=None, report_name_suffix=None):
        """Run inference for multiple directories."""
        print("\n" + "="*80)
        print(f"Batch inference for {len(base_dirs)} directories")
        if output_dir:
            print(f"Shared output directory: {output_dir}")
        print("="*80)

        overall_start = time.time()
        summary = {
            'total_directories': len(base_dirs),
            'total_files': 0, 'total_success': 0,
            'total_skipped': 0, 'total_failed': 0,
            'per_directory': {},
            'start_time': datetime.now().isoformat(),
        }

        for idx, dir_path in enumerate(base_dirs):
            if not os.path.exists(dir_path):
                print(f"\nWarning: directory does not exist, skipped: {dir_path}")
                continue

            print(f"\n[{idx+1}/{len(base_dirs)}]")
            try:
                dir_output = output_dir
                if output_dir and len(base_dirs) > 1:
                    dir_output = os.path.join(output_dir, os.path.basename(dir_path))
                res = self.infer_directory(
                    dir_path,
                    threshold,
                    postprocess,
                    min_size,
                    dir_output,
                    output_name_suffix=output_name_suffix,
                    report_name_suffix=report_name_suffix,
                )
                summary['per_directory'][os.path.basename(dir_path)] = res
                summary['total_files']   += res['total_files']
                summary['total_success'] += res['success']
                summary['total_skipped'] += res['skipped']
                summary['total_failed']  += len(res['failed'])
            except Exception as e:
                print(f"Directory failed: {dir_path} - {e}")

        total_elapsed = time.time() - overall_start
        h, m, s = int(total_elapsed // 3600), int((total_elapsed % 3600) // 60), int(total_elapsed % 60)
        summary.update({
            'end_time':              datetime.now().isoformat(),
            'total_processing_time': f"{h}h {m}m {s}s",
            'total_elapsed_seconds': total_elapsed,
        })

        if summary_path:
            summary_dir = os.path.dirname(summary_path)
            if summary_dir:
                os.makedirs(summary_dir, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary report: {summary_path}")

        print("\n" + "="*80)
        print(f"All done: files={summary['total_files']}, "
              f"created={summary['total_success']}, skipped={summary['total_skipped']}, "
              f"failed={summary['total_failed']}, time={summary['total_processing_time']}")
        print("="*80)
        return summary

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Model and inference settings.
    DEVICE        = 'cuda'
    BASE_CHANNELS = 32
    TARGET_SIZE   = (182, 218, 182)
    THRESHOLD     = 0.5
    POSTPROCESS   = False
    MIN_SIZE      = 0

    # Input directories containing .nii.gz images.
    OUTPUT_DIRS = [
        os.path.join(SCRIPT_DIR, 'input_images'),
    ]

    # Model checkpoints to run. Keep train.py in the same directory.
    EXPERT_MODELS = [
        {
            'name': 'ExpertA',
            'model_path': os.path.join(SCRIPT_DIR, 'expertA.pth'),
            'output_suffix': 'ExpertA',
        },
        {
            'name': 'ExpertB',
            'model_path': os.path.join(SCRIPT_DIR, 'expertB.pth'),
            'output_suffix': 'ExpertB',
        },
    ]

    # Set to None to save outputs next to each input image.
    OUTPUT_SEG_DIR = None

    # Directory for summary JSON files.
    SUMMARY_DIR = SCRIPT_DIR

    for expert in EXPERT_MODELS:
        print("\n" + "#" * 80)
        print(f"Start inference: {expert['name']}")
        print("#" * 80)

        inferencer = BatchOutputInference(
            model_path=expert['model_path'],
            device=DEVICE,
            num_classes=3,
            base_channels=BASE_CHANNELS,
            target_size=TARGET_SIZE,
        )

        summary_path = os.path.join(
            SUMMARY_DIR,
            f"real_annotations_{expert['output_suffix']}_summary.json"
        )
        inferencer.infer_all_directories(
            base_dirs=OUTPUT_DIRS,
            threshold=THRESHOLD,
            postprocess=POSTPROCESS,
            min_size=MIN_SIZE,
            output_dir=OUTPUT_SEG_DIR,
            summary_path=summary_path,
            output_name_suffix=expert['output_suffix'],
            report_name_suffix=expert['output_suffix'],
        )
