#!/usr/bin/env python3
"""
Radiomics feature extraction from paired preoperative and postoperative CT images.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from pathlib import Path
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
import functools
import copy
warnings.filterwarnings('ignore')

EXCLUDED_SAH_TERRITORY_PREFIXES = (
    'pre_SAH_ACA_',
    'post_SAH_ACA_',
    'pre_SAH_MCA_',
    'post_SAH_MCA_',
    'pre_SAH_PCA_',
    'post_SAH_PCA_',
    'pre_SAH_Brainstem_',
    'post_SAH_Brainstem_',
    'pre_SAH_Cerebellum_',
    'post_SAH_Cerebellum_',
    'pre_SAH_Cistern_',
    'post_SAH_Cistern_',
)
PROTECTED_CLINICAL_KEYWORDS = ('CS-DE',)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.glcm').setLevel(logging.ERROR)
logging.getLogger('radiomics.imageoperations').setLevel(logging.ERROR)
logging.getLogger('radiomics.featureextractor').setLevel(logging.ERROR)


def is_protected_clinical_column(column_name):
    return any(keyword in column_name for keyword in PROTECTED_CLINICAL_KEYWORDS)


def filter_excluded_feature_columns(feature_dict):
    filtered = {}
    for key, value in feature_dict.items():
        if any(key.startswith(prefix) for prefix in EXCLUDED_SAH_TERRITORY_PREFIXES):
            continue
        filtered[key] = value
    return filtered


def merge_features_with_existing(old_df, new_df):
    if 'patient_id' not in new_df.columns:
        raise ValueError("New feature table must contain the patient_id column")

    if old_df is None or old_df.empty:
        return pd.DataFrame([filter_excluded_feature_columns(row) for row in new_df.to_dict('records')])

    if 'patient_id' not in old_df.columns:
        raise ValueError("Existing feature table must contain the patient_id column")

    if old_df['patient_id'].astype(str).duplicated().any():
        raise ValueError("Existing feature table has duplicated patient_id values and cannot be merged uniquely")

    if new_df['patient_id'].astype(str).duplicated().any():
        raise ValueError("New feature table has duplicated patient_id values and cannot be merged uniquely")

    old_df = pd.DataFrame([filter_excluded_feature_columns(row) for row in old_df.to_dict('records')])
    new_df = pd.DataFrame([filter_excluded_feature_columns(row) for row in new_df.to_dict('records')])
    old_df['patient_id'] = old_df['patient_id'].astype(str)
    new_df['patient_id'] = new_df['patient_id'].astype(str)

    old_indexed = old_df.set_index('patient_id')
    new_indexed = new_df.set_index('patient_id')

    result = old_indexed.copy()
    for column in new_indexed.columns:
        if is_protected_clinical_column(column) and column in result.columns:
            continue
        result[column] = new_indexed[column]

    for column in old_indexed.columns:
        if column not in result.columns:
            result[column] = old_indexed[column]

    ordered_columns = list(old_indexed.columns)
    for column in new_indexed.columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    result = result.reset_index()
    ordered_columns = ['patient_id'] + [c for c in ordered_columns if c in result.columns]
    return result.loc[:, ordered_columns]


def _worker_process_patient(args):
    """
    Worker function for multiprocessing.

    Args:
        args: (extractor_params, patient_info) tuple

    Returns:
        dict: patient feature dictionary or None
    """
    extractor_params, patient_info = args

    try:
        from radiomics import featureextractor
        import SimpleITK as sitk

        if extractor_params['config_path'] and os.path.exists(extractor_params['config_path']):
            temp_extractor = featureextractor.RadiomicsFeatureExtractor(extractor_params['config_path'])
        else:
            temp_extractor = featureextractor.RadiomicsFeatureExtractor()
            temp_extractor.settings['binWidth'] = 25
            temp_extractor.settings['resampledPixelSpacing'] = None
            temp_extractor.settings['interpolator'] = sitk.sitkBSpline
            temp_extractor.settings['normalize'] = True
            temp_extractor.settings['normalizeScale'] = 100
            temp_extractor.enableAllFeatures()

        class TempExtractor:
            def __init__(self, extractor, label_names, extract_by_label):
                self.extractor = extractor
                self.label_names = label_names
                self.extract_by_label = extract_by_label

        temp_obj = TempExtractor(temp_extractor, extractor_params['label_names'], extractor_params['extract_by_label'])

        temp_obj.calculate_basic_stats = lambda *args, **kwargs: RadiomicsExtractor.calculate_basic_stats(temp_obj, *args, **kwargs)
        temp_obj.extract_radiomics_features = lambda *args, **kwargs: RadiomicsExtractor.extract_radiomics_features(temp_obj, *args, **kwargs)
        temp_obj._calculate_multi_label_features = lambda *args, **kwargs: RadiomicsExtractor._calculate_multi_label_features(temp_obj, *args, **kwargs)
        temp_obj._calculate_delta_radiomics = lambda *args, **kwargs: RadiomicsExtractor._calculate_delta_radiomics(temp_obj, *args, **kwargs)
        temp_obj._calculate_clinical_features = lambda *args, **kwargs: RadiomicsExtractor._calculate_clinical_features(temp_obj, *args, **kwargs)
        temp_obj._extract_total_ventricular_features = lambda *args, **kwargs: RadiomicsExtractor._extract_total_ventricular_features(temp_obj, *args, **kwargs)
        temp_obj._extract_merged_ventricular_features = lambda *args, **kwargs: RadiomicsExtractor._extract_merged_ventricular_features(temp_obj, *args, **kwargs)
        temp_obj._calculate_reference_non_sah_features = lambda *args, **kwargs: RadiomicsExtractor._calculate_reference_non_sah_features(temp_obj, *args, **kwargs)
        temp_obj._compute_mask_stats_from_arrays = lambda *args, **kwargs: RadiomicsExtractor._compute_mask_stats_from_arrays(*args, **kwargs)

        return RadiomicsExtractor.process_single_patient(temp_obj, patient_info)

    except Exception as e:
        patient_id = patient_info.get('patient_id', 'unknown')
        print(f"Worker failed for patient {patient_id} with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


class RadiomicsExtractor:
    """Radiomics feature extractor"""

    def __init__(self, data_root, output_dir, config_path=None, extract_by_label=True):
        """
        Initialize the feature extractor

        Args:
            data_root: data root directory
            output_dir: output directory
            config_path: PyRadiomics config path
            extract_by_label: whether to extract features per label
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extract_by_label = extract_by_label
        self.config_path = config_path

        self.label_names = {
            1: 'IVH',
            2: 'SAH',
            3: 'Ventricle'
        }

        if config_path and os.path.exists(config_path):
            self.extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
        else:
            self.extractor = featureextractor.RadiomicsFeatureExtractor()
            self._configure_extractor()

        logger.info(f"Feature extractor initialized")
        logger.info(f"Data root: {self.data_root}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Extract by label: {self.extract_by_label}")

    def _configure_extractor(self):
        """Configure feature extractor parameters"""
        self.extractor.settings['binWidth'] = 25
        self.extractor.settings['resampledPixelSpacing'] = None
        self.extractor.settings['interpolator'] = sitk.sitkBSpline
        self.extractor.settings['normalize'] = True
        self.extractor.settings['normalizeScale'] = 100

        self.extractor.enableAllFeatures()

        logger.info("Using default configuration")

    @staticmethod
    def _compute_mask_stats_from_arrays(image_array, mask_array, spacing):
        voxel_count = int(np.sum(mask_array))
        voxel_vol_cm3 = float(spacing[0] * spacing[1] * spacing[2] * 0.001)
        stats = {
            'volume': voxel_count * voxel_vol_cm3,
            'mean_hu': 0.0,
            'std_hu': 0.0,
            'entropy': 0.0,
        }
        if voxel_count <= 0 or image_array is None:
            return stats

        hu_vals = image_array[mask_array].astype(np.float32)
        stats['mean_hu'] = float(np.mean(hu_vals))
        stats['std_hu'] = float(np.std(hu_vals))
        hist, _ = np.histogram(hu_vals, bins=32)
        hist = hist[hist > 0].astype(np.float32)
        if hist.size > 0:
            prob = hist / hist.sum()
            stats['entropy'] = float(-np.sum(prob * np.log2(prob + 1e-10)))
        return stats

    def calculate_basic_stats(self, image_path, mask_path, label_value=None):
        """
        Calculate basic volume and HU statistics.

        Args:
            image_path: image path
            mask_path: segmentation mask path
            label_value: label value; None uses all nonzero voxels

        Returns:
            dict: basic statistics
        """
        try:
            image = sitk.ReadImage(str(image_path))
            mask = sitk.ReadImage(str(mask_path))

            image_array = sitk.GetArrayFromImage(image)
            mask_array = sitk.GetArrayFromImage(mask)

            if label_value is not None:
                roi_mask = (mask_array == label_value)
            else:
                roi_mask = (mask_array > 0)

            spacing = image.GetSpacing()
            voxel_volume = spacing[0] * spacing[1] * spacing[2]

            roi_voxels = image_array[roi_mask]

            if len(roi_voxels) == 0:
                logger.warning(f"Empty mask: {mask_path}, label={label_value}")
                return None

            stats = {
                'volume_mm3': len(roi_voxels) * voxel_volume,
                'volume_cm3': len(roi_voxels) * voxel_volume / 1000,
                'voxel_count': len(roi_voxels),
                'mean_hu': float(np.mean(roi_voxels)),
                'std_hu': float(np.std(roi_voxels)),
                'min_hu': float(np.min(roi_voxels)),
                'max_hu': float(np.max(roi_voxels)),
                'median_hu': float(np.median(roi_voxels)),
                'q25_hu': float(np.percentile(roi_voxels, 25)),
                'q75_hu': float(np.percentile(roi_voxels, 75))
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate basic statistics for {image_path}: {str(e)}")
            return None

    def extract_radiomics_features(self, image_path, mask_path, label_value=None):
        """
        Extract radiomics features.

        Args:
            image_path: image path
            mask_path: segmentation mask path
            label_value: label value; None uses all nonzero voxels

        Returns:
            dict: radiomics features
        """
        try:
            image = sitk.ReadImage(str(image_path))
            image_array = sitk.GetArrayFromImage(image)
            mask = sitk.ReadImage(str(mask_path))
            mask_array = sitk.GetArrayFromImage(mask)

            if label_value is not None:
                binary_mask_array = (mask_array == label_value).astype(np.uint8)
            else:
                binary_mask_array = (mask_array > 0).astype(np.uint8)

            if int(binary_mask_array.sum()) == 0:
                logger.warning(f"Radiomics ROI is empty, skipped: {mask_path}, label={label_value}")
                return None

            roi_values = image_array[binary_mask_array > 0]
            if roi_values.size == 0:
                logger.warning(f"Radiomics ROI voxels are empty, skipped: {mask_path}, label={label_value}")
                return None
            if not np.isfinite(roi_values).all():
                logger.warning(f"Radiomics ROI contains non-finite values, skipped: {mask_path}, label={label_value}")
                return None
            if float(np.max(roi_values)) == float(np.min(roi_values)):
                logger.warning(f"Radiomics ROI has constant intensity, skipped: {mask_path}, label={label_value}")
                return None

            binary_mask = sitk.GetImageFromArray(binary_mask_array)
            binary_mask.CopyInformation(mask)

            import tempfile
            temp_mask_path = None
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
                temp_mask_path = tmp.name
            sitk.WriteImage(binary_mask, temp_mask_path)

            try:
                features = self.extractor.execute(str(image_path), temp_mask_path)
            finally:
                if temp_mask_path and os.path.exists(temp_mask_path):
                    os.remove(temp_mask_path)

            feature_dict = {}
            for key, value in features.items():
                if 'diagnostics' not in key:
                    try:
                        feature_dict[key] = float(value)
                    except (ValueError, TypeError):
                        feature_dict[key] = str(value)

            return feature_dict

        except Exception as e:
            logger.error(f"Extract radiomics features.Failed {image_path}, label={label_value}: {str(e)}")
            return None

    def _scan_folder_for_patients(self, folder, folder_name):
        """
        Scan one folder for patient files.

        Args:
            folder: folder path
            folder_name: folder name for records

        Returns:
            list: patient records in the folder
        """
        patient_data = []

        files = list(folder.glob("*.nii.gz"))

        if len(files) == 0:
            return patient_data

        file_dict = {}
        for file in files:
            filename = file.name
            if '_segmentation' in filename:
                patient_id = filename.replace('_segmentation.nii.gz', '')
                if patient_id not in file_dict:
                    file_dict[patient_id] = {}
                file_dict[patient_id]['seg'] = file
            elif filename.endswith('_seg.nii.gz'):
                patient_id = filename.replace('_seg.nii.gz', '')
                if patient_id not in file_dict:
                    file_dict[patient_id] = {}
                file_dict[patient_id]['seg'] = file
            else:
                patient_id = filename.replace('.nii.gz', '')
                if patient_id not in file_dict:
                    file_dict[patient_id] = {}
                file_dict[patient_id]['img'] = file

        processed_ids = set()
        for patient_id in file_dict.keys():
            if patient_id in processed_ids:
                continue

            if patient_id.endswith('-1'):
                base_id = patient_id[:-2]
                post_id = patient_id
                pre_id = base_id
            else:
                pre_id = patient_id
                post_id = f"{patient_id}-1"
                base_id = patient_id

            has_pre = pre_id in file_dict and 'img' in file_dict[pre_id] and 'seg' in file_dict[pre_id]
            has_post = post_id in file_dict and 'img' in file_dict[post_id] and 'seg' in file_dict[post_id]

            if has_pre and has_post:
                patient_info = {
                    'patient_id': base_id,
                    'subfolder': folder_name,
                    'pre_image': file_dict[pre_id]['img'],
                    'pre_seg': file_dict[pre_id]['seg'],
                    'post_image': file_dict[post_id]['img'],
                    'post_seg': file_dict[post_id]['seg']
                }
                patient_data.append(patient_info)
                processed_ids.add(pre_id)
                processed_ids.add(post_id)
                logger.debug(f"Found pair: {base_id}")

        return patient_data

    def find_patient_pairs(self):
        """
        Find all pre/post image pairs. Supports direct files or subfolders.

        Returns:
            list: patient records
        """
        patient_data = []

        direct_files = list(self.data_root.glob("*.nii.gz"))

        if len(direct_files) > 0:
            logger.info(f"Detected direct-file mode: {self.data_root} contains patient files directly")
            folder_name = self.data_root.name
            patient_data = self._scan_folder_for_patients(self.data_root, folder_name)
            logger.info(f"Found {len(patient_data)} pre/post image pairs in {folder_name}")
        else:
            logger.info(f"Detected subfolder mode: scanning {self.data_root} subfolders")
            subfolders = [f for f in self.data_root.iterdir() if f.is_dir()]

            if len(subfolders) == 0:
                logger.warning(f"No subfolders or patient files found under {self.data_root}")
                return patient_data

            for subfolder in subfolders:
                logger.info(f"Scanning folder: {subfolder.name}")
                folder_patients = self._scan_folder_for_patients(subfolder, subfolder.name)
                patient_data.extend(folder_patients)

        logger.info(f"Found {len(patient_data)} pre/post image pairs")
        return patient_data

    def process_single_patient(self, patient_info):
        """
        Process one patient for multiprocessing.

        Args:
            patient_info: patient metadata dictionary

        Returns:
            dict: all extracted features, or None on failure
        """
        patient_id = patient_info['patient_id']
        subfolder = patient_info['subfolder']

        try:
            result = {
                'patient_id': patient_id,
                'subfolder': subfolder
            }

            if self.extract_by_label:
                for label_value, label_name in self.label_names.items():
                    pre_stats = self.calculate_basic_stats(
                        patient_info['pre_image'],
                        patient_info['pre_seg'],
                        label_value=label_value
                    )

                    pre_radiomics = None
                    if pre_stats is not None:
                        pre_radiomics = self.extract_radiomics_features(
                            patient_info['pre_image'],
                            patient_info['pre_seg'],
                            label_value=label_value
                        )

                    if pre_stats:
                        for key, value in pre_stats.items():
                            result[f'pre_{label_name}_{key}'] = value

                    if pre_radiomics:
                        for key, value in pre_radiomics.items():
                            result[f'pre_{label_name}_{key}'] = value

                    post_stats = self.calculate_basic_stats(
                        patient_info['post_image'],
                        patient_info['post_seg'],
                        label_value=label_value
                    )

                    post_radiomics = None
                    if post_stats is not None:
                        post_radiomics = self.extract_radiomics_features(
                            patient_info['post_image'],
                            patient_info['post_seg'],
                            label_value=label_value
                        )

                    if post_stats:
                        for key, value in post_stats.items():
                            result[f'post_{label_name}_{key}'] = value

                    if post_radiomics:
                        for key, value in post_radiomics.items():
                            result[f'post_{label_name}_{key}'] = value

                    if pre_stats or post_stats:
                        pre_vol_mm3 = pre_stats['volume_mm3'] if pre_stats else 0
                        post_vol_mm3 = post_stats['volume_mm3'] if post_stats else 0
                        pre_vol_cm3 = pre_stats['volume_cm3'] if pre_stats else 0
                        post_vol_cm3 = post_stats['volume_cm3'] if post_stats else 0

                        result[f'{label_name}_volume_change_mm3'] = post_vol_mm3 - pre_vol_mm3
                        result[f'{label_name}_volume_change_cm3'] = post_vol_cm3 - pre_vol_cm3

                        if pre_vol_cm3 > 0:
                            result[f'{label_name}_volume_change_percent'] = (post_vol_cm3 - pre_vol_cm3) / pre_vol_cm3 * 100
                            result[f'{label_name}_clearance_rate'] = (pre_vol_cm3 - post_vol_cm3) / pre_vol_cm3 * 100
                        elif post_vol_cm3 > 0:
                            result[f'{label_name}_volume_change_percent'] = float('inf')
                            result[f'{label_name}_clearance_rate'] = -float('inf')

                        if pre_stats and post_stats:
                            result[f'{label_name}_mean_hu_change'] = post_stats['mean_hu'] - pre_stats['mean_hu']

                result = self._extract_total_ventricular_features(result, patient_info)

                result = self._calculate_multi_label_features(result)
                result = self._calculate_delta_radiomics(result)
                result = self._calculate_clinical_features(result)
                result = self._calculate_reference_non_sah_features(result, patient_info)

            else:
                pre_stats = self.calculate_basic_stats(
                    patient_info['pre_image'],
                    patient_info['pre_seg']
                )
                pre_radiomics = self.extract_radiomics_features(
                    patient_info['pre_image'],
                    patient_info['pre_seg']
                )

                if pre_stats:
                    for key, value in pre_stats.items():
                        result[f'pre_{key}'] = value

                if pre_radiomics:
                    for key, value in pre_radiomics.items():
                        result[f'pre_{key}'] = value

                post_stats = self.calculate_basic_stats(
                    patient_info['post_image'],
                    patient_info['post_seg']
                )
                post_radiomics = self.extract_radiomics_features(
                    patient_info['post_image'],
                    patient_info['post_seg']
                )

                if post_stats:
                    for key, value in post_stats.items():
                        result[f'post_{key}'] = value

                if post_radiomics:
                    for key, value in post_radiomics.items():
                        result[f'post_{key}'] = value

                if pre_stats and post_stats:
                    result['volume_change_mm3'] = post_stats['volume_mm3'] - pre_stats['volume_mm3']
                    result['volume_change_cm3'] = post_stats['volume_cm3'] - pre_stats['volume_cm3']
                    result['volume_change_percent'] = (post_stats['volume_cm3'] - pre_stats['volume_cm3']) / pre_stats['volume_cm3'] * 100
                    result['mean_hu_change'] = post_stats['mean_hu'] - pre_stats['mean_hu']

            return filter_excluded_feature_columns(result)

        except Exception as e:
            logger.error(f"Failed to process patient {patient_id} with error: {str(e)}")
            return None

    def process_all_patients(self, resume=True, n_jobs=None):
        """
        Process all patients and save extracted features.

        Args:
            resume: whether to resume from existing output
            n_jobs: number of worker processes

        Returns:
            pd.DataFrame: feature table
        """
        patient_pairs = self.find_patient_pairs()

        if len(patient_pairs) == 0:
            logger.warning("No patient data found")
            return None

        output_csv = self.output_dir / 'radiomics_features.csv'
        processed_patients = set()
        all_features = []

        if resume and output_csv.exists():
            logger.info(f"Existing output found; resuming...")
            try:
                existing_df = pd.read_csv(output_csv, encoding='utf-8-sig')

                if 'patient_id' in existing_df.columns:
                    processed_patients = set(str(pid) for pid in existing_df['patient_id'].values)
                    all_features = existing_df.to_dict('records')
                    logger.info(f"Already processed {len(processed_patients)} patients; continuing with remaining patients")
                else:
                    logger.warning(f"CSV file has no patient_id column; columns are: {existing_df.columns.tolist()}")
                    processed_patients = set()
                    all_features = []
            except Exception as e:
                logger.warning(f"Failed to read existing results: {str(e)}，restarting")
                import traceback
                traceback.print_exc()
                processed_patients = set()
                all_features = []

        remaining_patients = [p for p in patient_pairs if p['patient_id'] not in processed_patients]
        logger.info(f"Found {len(patient_pairs)} patients, remaining {len(remaining_patients)} to process")

        if len(processed_patients) > 0:
            sample_processed = list(processed_patients)[:3]
            logger.info(f"Processed patient examples: {sample_processed}")
        if len(remaining_patients) > 0:
            sample_remaining = [p['patient_id'] for p in remaining_patients[:3]]
            logger.info(f"Remaining patient examples: {sample_remaining}")

        if len(remaining_patients) == 0:
            logger.info("All patients are already processed; returning existing data")
            df = pd.DataFrame(all_features)
            logger.info(f"Existing feature table shape: {df.shape}")
            return df

        if n_jobs is None:
            n_jobs = max(1, cpu_count() - 1)
        n_jobs = min(n_jobs, len(remaining_patients))

        logger.info(f"Using {n_jobs} worker processes")

        extractor_params = {
            'config_path': self.config_path,
            'label_names': self.label_names,
            'extract_by_label': self.extract_by_label
        }

        save_interval = 10
        batch_size = save_interval

        total_processed = 0
        total_successful = 0

        for batch_start in range(0, len(remaining_patients), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_patients))
            batch_patients = remaining_patients[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(remaining_patients)-1)//batch_size + 1} "
                       f"({batch_start+1}-{batch_end}/{len(remaining_patients)})")

            if n_jobs > 1:
                worker_args = [(extractor_params, patient_info) for patient_info in batch_patients]
                with Pool(processes=n_jobs) as pool:
                    batch_results = list(tqdm(
                        pool.imap(_worker_process_patient, worker_args),
                        total=len(batch_patients),
                        desc=f"Batch {batch_start//batch_size + 1}"
                    ))
            else:
                batch_results = []
                for patient_info in tqdm(batch_patients, desc=f"Batch {batch_start//batch_size + 1}"):
                    result = self.process_single_patient(patient_info)
                    batch_results.append(result)

            successful_batch = [r for r in batch_results if r is not None]
            total_processed += len(batch_patients)
            total_successful += len(successful_batch)

            logger.info(f"Successful in this batch: {len(successful_batch)}/{len(batch_patients)} patients")

            if len(successful_batch) > 0:
                all_features.extend(successful_batch)

                try:
                    df = pd.DataFrame(all_features)
                    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
                    logger.info(f"Progress saved: {len(all_features)} patients -> {output_csv}")
                except Exception as e:
                    logger.error(f"Failed to save CSV: {str(e)}")
                    backup_csv = self.output_dir / f'radiomics_features_backup_{batch_start//batch_size + 1}.csv'
                    try:
                        df.to_csv(backup_csv, index=False, encoding='utf-8-sig')
                        logger.info(f"Saved backup file: {backup_csv}")
                    except:
                        logger.error("Backup save also failed.")
            else:
                logger.warning(f"Batch {batch_start//batch_size + 1} No patients were processed successfully")

        logger.info(f"=" * 60)
        logger.info("All batches complete.")
        logger.info(f"Total processed: {total_processed} patients")
        logger.info(f"Successful extractions: {total_successful} patients")
        logger.info(f"Failed: {total_processed - total_successful} patients")
        logger.info(f"Success rate: {total_successful/total_processed*100:.1f}%")

        if len(all_features) == 0:
            logger.warning("No patients were processed successfully")
            return None

        df = pd.DataFrame(all_features)
        logger.info(f"Final data shape: {df.shape}")
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"Final results saved to: {output_csv}")

        summary_file = self.output_dir / 'feature_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Radiomics feature extraction summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Total patients: {len(df)}\n")
            f.write(f"Total features: {len(df.columns)}\n\n")
            f.write(f"Feature list:\n")
            for col in df.columns:
                f.write(f"  - {col}\n")

        logger.info(f"Summary saved to: {summary_file}")

        return df

    def _extract_total_ventricular_features(self, result, patient_info):
        """
        Extract total ventricular features by merging IVH and Ventricle.

        Args:
            result: existing feature dictionary
            patient_info: patient metadata

        Returns:
            dict: feature dictionary with total ventricular features
        """
        try:
            if 'pre_IVH_volume_cm3' in result or 'pre_Ventricle_volume_cm3' in result:
                pre_total_stats, pre_total_radiomics = self._extract_merged_ventricular_features(
                    patient_info['pre_image'],
                    patient_info['pre_seg']
                )

                if pre_total_stats:
                    for key, value in pre_total_stats.items():
                        result[f'pre_total_ventricular_{key}'] = value

                if pre_total_radiomics:
                    for key, value in pre_total_radiomics.items():
                        result[f'pre_total_ventricular_{key}'] = value

            if 'post_IVH_volume_cm3' in result or 'post_Ventricle_volume_cm3' in result:
                post_total_stats, post_total_radiomics = self._extract_merged_ventricular_features(
                    patient_info['post_image'],
                    patient_info['post_seg']
                )

                if post_total_stats:
                    for key, value in post_total_stats.items():
                        result[f'post_total_ventricular_{key}'] = value

                if post_total_radiomics:
                    for key, value in post_total_radiomics.items():
                        result[f'post_total_ventricular_{key}'] = value

            if 'pre_total_ventricular_volume_cm3' in result and 'post_total_ventricular_volume_cm3' in result:
                pre_vol = result['pre_total_ventricular_volume_cm3']
                post_vol = result['post_total_ventricular_volume_cm3']

                result['total_ventricular_volume_change_cm3'] = post_vol - pre_vol

                if pre_vol > 0:
                    result['total_ventricular_volume_change_percent'] = (post_vol - pre_vol) / pre_vol * 100

                if 'pre_total_ventricular_mean_hu' in result and 'post_total_ventricular_mean_hu' in result:
                    result['ventricular_density_reduction'] = result['pre_total_ventricular_mean_hu'] - result['post_total_ventricular_mean_hu']

        except Exception as e:
            logger.warning(f"Failed to extract total ventricular features: {str(e)}")

        return result

    def _extract_merged_ventricular_features(self, image_path, seg_path):
        """
        Extract merged ventricular features from IVH label 1 and Ventricle label 3.

        Args:
            image_path: image path
            seg_path: segmentation mask path

        Returns:
            tuple: (basic statistics, radiomics features)
        """
        try:
            mask = sitk.ReadImage(str(seg_path))
            mask_array = sitk.GetArrayFromImage(mask)

            merged_mask_array = np.logical_or(mask_array == 1, mask_array == 3).astype(np.uint8)

            if np.sum(merged_mask_array) == 0:
                logger.debug(f"Merged ventricular mask is empty: {seg_path}")
                return None, None

            merged_mask = sitk.GetImageFromArray(merged_mask_array)
            merged_mask.CopyInformation(mask)

            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
                temp_mask_path = tmp.name
            sitk.WriteImage(merged_mask, temp_mask_path)

            stats = self.calculate_basic_stats(image_path, temp_mask_path, label_value=None)

            radiomics = self.extract_radiomics_features(image_path, temp_mask_path, label_value=None)

            os.remove(temp_mask_path)

            return stats, radiomics

        except Exception as e:
            logger.error(f"Failed to extract merged ventricular features: {str(e)}")
            return None, None

    def _calculate_multi_label_features(self, result):
        """
        Calculate multi-label interaction features.

        Args:
            result: feature dictionary with label-specific features

        Returns:
            dict: feature dictionary with interaction features
        """
        try:
            if 'pre_IVH_volume_cm3' in result and 'pre_Ventricle_volume_cm3' in result:
                result['pre_complete_ventricular_volume'] = result['pre_IVH_volume_cm3'] + result['pre_Ventricle_volume_cm3']

            if 'post_IVH_volume_cm3' in result and 'post_Ventricle_volume_cm3' in result:
                result['post_complete_ventricular_volume'] = result['post_IVH_volume_cm3'] + result['post_Ventricle_volume_cm3']

            if 'pre_IVH_volume_cm3' in result and 'pre_complete_ventricular_volume' in result:
                if result['pre_complete_ventricular_volume'] > 0:
                    result['pre_IVH_occupation_ratio'] = result['pre_IVH_volume_cm3'] / result['pre_complete_ventricular_volume']
                else:
                    result['pre_IVH_occupation_ratio'] = 0

            if 'post_IVH_volume_cm3' in result and 'post_complete_ventricular_volume' in result:
                if result['post_complete_ventricular_volume'] > 0:
                    result['post_IVH_occupation_ratio'] = result['post_IVH_volume_cm3'] / result['post_complete_ventricular_volume']
                else:
                    result['post_IVH_occupation_ratio'] = 0

            if 'pre_IVH_volume_cm3' in result or 'pre_SAH_volume_cm3' in result:
                result['pre_total_hemorrhage_volume'] = result.get('pre_IVH_volume_cm3', 0) + result.get('pre_SAH_volume_cm3', 0)

            if 'post_IVH_volume_cm3' in result or 'post_SAH_volume_cm3' in result:
                result['post_total_hemorrhage_volume'] = result.get('post_IVH_volume_cm3', 0) + result.get('post_SAH_volume_cm3', 0)

            if 'pre_IVH_volume_cm3' in result and 'pre_total_hemorrhage_volume' in result:
                if result['pre_total_hemorrhage_volume'] > 0:
                    result['pre_IVH_hemorrhage_ratio'] = result['pre_IVH_volume_cm3'] / result['pre_total_hemorrhage_volume']

            if 'post_IVH_volume_cm3' in result and 'post_total_hemorrhage_volume' in result:
                if result['post_total_hemorrhage_volume'] > 0:
                    result['post_IVH_hemorrhage_ratio'] = result['post_IVH_volume_cm3'] / result['post_total_hemorrhage_volume']

            if 'pre_complete_ventricular_volume' in result and 'post_complete_ventricular_volume' in result:
                if result['pre_complete_ventricular_volume'] > 0:
                    result['total_ventricle_expansion_index'] = (result['post_complete_ventricular_volume'] - result['pre_complete_ventricular_volume']) / result['pre_complete_ventricular_volume']

            if 'pre_total_hemorrhage_volume' in result and 'post_total_hemorrhage_volume' in result:
                if result['pre_total_hemorrhage_volume'] > 0:
                    result['total_hemorrhage_clearance_rate'] = (result['pre_total_hemorrhage_volume'] - result['post_total_hemorrhage_volume']) / result['pre_total_hemorrhage_volume'] * 100

        except Exception as e:
            logger.warning(f"Calculate multi-label interaction features.with error: {str(e)}")

        return result

    def _calculate_delta_radiomics(self, result):
        """
        Calculate delta-radiomics features.

        Args:
            result: feature dictionary with pre/post features

        Returns:
            dict: feature dictionary with delta features
        """
        try:
            feature_types = [
                'volume_cm3', 'mean_hu', 'std_hu', 'median_hu',
                'original_firstorder_Energy', 'original_firstorder_Entropy',
                'original_firstorder_Skewness', 'original_firstorder_Kurtosis',
                'original_shape_Sphericity', 'original_shape_SurfaceVolumeRatio',
                'original_glcm_Contrast', 'original_glcm_Correlation',
                'original_glcm_Energy', 'original_glcm_Homogeneity',
                'original_glrlm_ShortRunEmphasis', 'original_glrlm_LongRunEmphasis',
                'original_glszm_SmallAreaEmphasis', 'original_glszm_LargeAreaEmphasis'
            ]

            for label_name in self.label_names.values():
                for feature_type in feature_types:
                    pre_key = f'pre_{label_name}_{feature_type}'
                    post_key = f'post_{label_name}_{feature_type}'

                    if pre_key in result and post_key in result:
                        pre_val = result[pre_key]
                        post_val = result[post_key]

                        result[f'delta_{label_name}_{feature_type}_abs'] = post_val - pre_val

                        if pre_val != 0:
                            result[f'delta_{label_name}_{feature_type}_pct'] = (post_val - pre_val) / abs(pre_val) * 100

            if 'delta_IVH_volume_cm3_abs' in result and 'delta_Ventricle_original_shape_Sphericity_abs' in result:
                result['IVH_clearance_ventricle_recovery_ratio'] = result.get('delta_IVH_volume_cm3_abs', 0) / (abs(result.get('delta_Ventricle_original_shape_Sphericity_abs', 1)) + 1e-6)

        except Exception as e:
            logger.warning(f"Failed to calculate delta-radiomics features: {str(e)}")

        return result

    def _calculate_clinical_features(self, result):
        """
        Calculate clinically derived features.

        Args:
            result: feature dictionary with base features

        Returns:
            dict: feature dictionary with clinical features
        """
        try:
            for label_name in ['IVH', 'SAH']:
                pre_mean_key = f'pre_{label_name}_mean_hu'
                post_mean_key = f'post_{label_name}_mean_hu'

                if pre_mean_key in result and post_mean_key in result:
                    result[f'{label_name}_density_evolution_rate'] = (result[post_mean_key] - result[pre_mean_key]) / abs(result[pre_mean_key] + 1e-6) * 100

                    pre_std_key = f'pre_{label_name}_std_hu'
                    post_std_key = f'post_{label_name}_std_hu'
                    if pre_std_key in result and post_std_key in result:
                        result[f'{label_name}_heterogeneity_change'] = result[post_std_key] - result[pre_std_key]

            ivh_clearance = result.get('IVH_clearance_rate', 0)
            sah_clearance = result.get('SAH_clearance_rate', 0)
            total_clearance = result.get('total_hemorrhage_clearance_rate', 0)

            result['surgical_efficacy_score'] = (ivh_clearance * 0.5 + sah_clearance * 0.3 + total_clearance * 0.2)

            if 'pre_Ventricle_original_shape_Sphericity' in result and 'post_Ventricle_original_shape_Sphericity' in result:
                result['ventricle_shape_recovery'] = result['post_Ventricle_original_shape_Sphericity'] - result['pre_Ventricle_original_shape_Sphericity']

            if 'pre_IVH_volume_cm3' in result and 'pre_complete_ventricular_volume' in result:
                if result['pre_complete_ventricular_volume'] > 0:
                    result['pre_IVH_burden'] = result['pre_IVH_volume_cm3'] / result['pre_complete_ventricular_volume']

            if 'post_IVH_volume_cm3' in result and 'post_complete_ventricular_volume' in result:
                if result['post_complete_ventricular_volume'] > 0:
                    result['post_IVH_burden'] = result['post_IVH_volume_cm3'] / result['post_complete_ventricular_volume']

            if 'pre_IVH_occupation_ratio' in result and 'post_IVH_occupation_ratio' in result:
                result['IVH_occupation_reduction'] = result['pre_IVH_occupation_ratio'] - result['post_IVH_occupation_ratio']

            for label_name in ['IVH', 'SAH']:
                pre_entropy = f'pre_{label_name}_original_firstorder_Entropy'
                post_entropy = f'post_{label_name}_original_firstorder_Entropy'

                if pre_entropy in result and post_entropy in result:
                    result[f'{label_name}_texture_simplification'] = result[pre_entropy] - result[post_entropy]

            if 'Ventricle_volume_change_cm3' in result and 'IVH_volume_change_cm3' in result:
                ivh_change = abs(result.get('IVH_volume_change_cm3', 0))
                if ivh_change > 0:
                    result['space_occupying_relief_ratio'] = result.get('Ventricle_volume_change_cm3', 0) / ivh_change

        except Exception as e:
            logger.warning(f"Failed to calculate clinical features: {str(e)}")

        return result

    def _calculate_reference_non_sah_features(self, result, patient_info):
        try:
            pre_img = sitk.ReadImage(str(patient_info['pre_image']))
            pre_seg = sitk.ReadImage(str(patient_info['pre_seg']))
            post_img = sitk.ReadImage(str(patient_info['post_image']))
            post_seg = sitk.ReadImage(str(patient_info['post_seg']))

            pre_img_arr = sitk.GetArrayFromImage(pre_img).astype(np.float32)
            pre_seg_arr = sitk.GetArrayFromImage(pre_seg)
            post_img_arr = sitk.GetArrayFromImage(post_img).astype(np.float32)
            post_seg_arr = sitk.GetArrayFromImage(post_seg)
            spacing = pre_img.GetSpacing()

            pre_ivh = self._compute_mask_stats_from_arrays(pre_img_arr, pre_seg_arr == 1, spacing)
            post_ivh = self._compute_mask_stats_from_arrays(post_img_arr, post_seg_arr == 1, spacing)
            pre_sah = self._compute_mask_stats_from_arrays(pre_img_arr, pre_seg_arr == 2, spacing)
            post_sah = self._compute_mask_stats_from_arrays(post_img_arr, post_seg_arr == 2, spacing)
            pre_vent = self._compute_mask_stats_from_arrays(pre_img_arr, pre_seg_arr == 3, spacing)
            post_vent = self._compute_mask_stats_from_arrays(post_img_arr, post_seg_arr == 3, spacing)
            pre_complete = self._compute_mask_stats_from_arrays(pre_img_arr, np.isin(pre_seg_arr, [1, 3]), spacing)
            post_complete = self._compute_mask_stats_from_arrays(post_img_arr, np.isin(post_seg_arr, [1, 3]), spacing)

            for prefix, stats_dict in [
                ('pre_IVH', pre_ivh),
                ('post_IVH', post_ivh),
                ('pre_SAH', pre_sah),
                ('post_SAH', post_sah),
                ('pre_Ventricle', pre_vent),
                ('post_Ventricle', post_vent),
                ('pre_CompleteVent', pre_complete),
                ('post_CompleteVent', post_complete),
            ]:
                for stat_name, value in stats_dict.items():
                    result[f'{prefix}_{stat_name}'] = value

            pre_ivh_vol = pre_ivh['volume']
            post_ivh_vol = post_ivh['volume']
            pre_sah_vol = pre_sah['volume']
            post_sah_vol = post_sah['volume']
            pre_vent_vol = pre_vent['volume']
            post_vent_vol = post_vent['volume']
            pre_complete_vol = pre_complete['volume']
            post_complete_vol = post_complete['volume']
            pre_total_hem = pre_ivh_vol + pre_sah_vol
            post_total_hem = post_ivh_vol + post_sah_vol

            result['IVH_clearance_rate'] = ((pre_ivh_vol - post_ivh_vol) / pre_ivh_vol * 100) if pre_ivh_vol > 0 else 0.0
            result['SAH_clearance_rate'] = ((pre_sah_vol - post_sah_vol) / pre_sah_vol * 100) if pre_sah_vol > 0 else 0.0
            result['Ventricle_volume_change_abs'] = post_vent_vol - pre_vent_vol
            result['Ventricle_volume_change_pct'] = ((post_vent_vol - pre_vent_vol) / pre_vent_vol * 100) if pre_vent_vol > 0 else 0.0
            result['pre_IVH_occupation_ratio'] = pre_ivh_vol / pre_complete_vol if pre_complete_vol > 0 else 0.0
            result['post_IVH_occupation_ratio'] = post_ivh_vol / post_complete_vol if post_complete_vol > 0 else 0.0
            result['pre_total_hemorrhage_volume'] = pre_total_hem
            result['post_total_hemorrhage_volume'] = post_total_hem
            result['total_ventricle_expansion_index'] = ((post_complete_vol - pre_complete_vol) / pre_complete_vol) if pre_complete_vol > 0 else 0.0
            result['total_hemorrhage_clearance_rate'] = ((pre_total_hem - post_total_hem) / pre_total_hem * 100) if pre_total_hem > 0 else 0.0
            result['IVH_clearance_ventricle_recovery_ratio'] = (pre_ivh_vol - post_ivh_vol) / abs(post_vent_vol - pre_vent_vol) if abs(post_vent_vol - pre_vent_vol) > 0 else 0.0
            result['IVH_density_evolution_rate'] = ((post_ivh['mean_hu'] - pre_ivh['mean_hu']) / abs(pre_ivh['mean_hu'])) if abs(pre_ivh['mean_hu']) > 0 else 0.0
            result['SAH_density_evolution_rate'] = ((post_sah['mean_hu'] - pre_sah['mean_hu']) / abs(pre_sah['mean_hu'])) if abs(pre_sah['mean_hu']) > 0 else 0.0
            result['IVH_heterogeneity_change'] = post_ivh['std_hu'] - pre_ivh['std_hu']
            result['SAH_heterogeneity_change'] = post_sah['std_hu'] - pre_sah['std_hu']
            result['ventricular_density_reduction'] = pre_complete['mean_hu'] - post_complete['mean_hu']

            ivh_cr = result['IVH_clearance_rate'] / 100.0
            sah_cr = result['SAH_clearance_rate'] / 100.0
            vent_ei = max(0.0, -result['total_ventricle_expansion_index'])
            result['surgical_efficacy_score'] = 0.5 * ivh_cr + 0.3 * sah_cr + 0.2 * vent_ei
            result['ventricle_shape_recovery'] = (
                -result['total_ventricle_expansion_index'] * 0.7
                + ((post_complete['mean_hu'] - pre_complete['mean_hu']) / (abs(pre_complete['mean_hu']) + 1e-6)) * 0.3
            )
            result['pre_IVH_burden'] = pre_ivh_vol / pre_complete_vol if pre_complete_vol > 0 else 0.0
            result['post_IVH_burden'] = post_ivh_vol / post_complete_vol if post_complete_vol > 0 else 0.0
            result['IVH_occupation_reduction'] = result['pre_IVH_occupation_ratio'] - result['post_IVH_occupation_ratio']
            result['complete_ventricular_volume_change_abs'] = post_complete_vol - pre_complete_vol
            result['complete_ventricular_volume_change_pct'] = ((post_complete_vol - pre_complete_vol) / pre_complete_vol * 100) if pre_complete_vol > 0 else 0.0
            result['IVH_texture_simplification'] = pre_ivh['entropy'] - post_ivh['entropy']
            result['SAH_texture_simplification'] = pre_sah['entropy'] - post_sah['entropy']

            hem_relief = pre_total_hem - post_total_hem
            vent_relief = max(0.0, pre_vent_vol - post_vent_vol)
            result['space_occupying_relief_ratio'] = hem_relief / (vent_relief + 1e-6)

        except Exception as e:
            logger.warning(f"Failed to calculate reference non-SAH territory features: {str(e)}")

        return result


def load_existing_features(csv_path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if 'patient_id' not in df.columns:
        raise ValueError(f"Existing feature file is missing patient_id column: {csv_path}")
    if df['patient_id'].astype(str).duplicated().any():
        raise ValueError(f"Existing feature file has duplicated patient_id values: {csv_path}")
    return df


def run_single_center(center_cfg):
    center_name = center_cfg['name']
    data_root = center_cfg['data_root']
    output_csv = Path(center_cfg['output_csv'])
    temp_output_dir = Path(center_cfg['temp_output_dir'])
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Start center: {center_name}")
    logger.info(f"Data directory: {data_root}")
    logger.info(f"Target CSV: {output_csv}")

    extractor = RadiomicsExtractor(
        data_root=data_root,
        output_dir=temp_output_dir,
        config_path=None,
        extract_by_label=True,
    )

    new_df = extractor.process_all_patients(resume=False, n_jobs=center_cfg.get('n_jobs'))
    if new_df is None or new_df.empty:
        logger.error(f"Center {center_name} did not produce any features")
        return None

    old_df = load_existing_features(output_csv)
    merged_df = merge_features_with_existing(old_df, new_df)
    merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    logger.info(f"Center {center_name} update complete: {output_csv}")
    logger.info(f"Final shape: {merged_df.shape}")
    return merged_df


def main():
    base_dir = Path(".")
    centers = [
        {
            'name': 'efy',
            'data_root': str(base_dir / 'data/output_efy'),
            'output_csv': str(base_dir / 'data/featuresefy.csv'),
            'temp_output_dir': str(base_dir / 'radiomics_output/efy'),
            'n_jobs': None,
        },
        {
            'name': 'th',
            'data_root': str(base_dir / 'data/output_th'),
            'output_csv': str(base_dir / 'data/featuresth.csv'),
            'temp_output_dir': str(base_dir / 'radiomics_output/th'),
            'n_jobs': None,
        },
    ]

    for center_cfg in centers:
        run_single_center(center_cfg)


if __name__ == "__main__":
    main()
