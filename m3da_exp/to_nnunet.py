import os
from pathlib import Path
from typing import Callable

import nibabel as nb
import numpy as np
from deli import save_json
from tqdm.auto import tqdm

from m3da_exp.utils import PathLike
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


NNUNET_RAW_PATH = Path(os.environ['nnUNet_raw'])
NNUNET_PREPROCESSED_PATH = Path(os.environ['nnUNet_preprocessed'])
NNUNET_RESULTS_PATH = Path(os.environ['nnUNet_results'])


def save_nifti(image, affine, path):
    nb.save(nb.Nifti1Image(image, affine=affine), path)


def spacing2affine(spacing: tuple):
    assert len(spacing) == 3, f"Spacing should be a tuple of length 3. {spacing}"
    affine = np.eye(4, dtype=np.float32)
    for i, x in enumerate(spacing):
        affine[i, i] = float(x)
    return affine


def check_existing_dataset(dataset_id: int, overwrite: bool = False, version: str = 'v2') -> str:
    if (not isinstance(dataset_id, int)) or dataset_id < 1:
        raise ValueError(f'dataset_id is not valid. It should be int > 0. '
                         f'But {dataset_id} ({type(dataset_id)}) is given.')

    if version == 'v1':
        dataset_prefix = f'Task{dataset_id:03d}'
    elif version == 'v2':
        dataset_prefix = f'Dataset{dataset_id:03d}'
    else:
        raise ValueError(f'Wrong nnUNet version is provided: {version}')

    for p in (NNUNET_RAW_PATH, NNUNET_PREPROCESSED_PATH, NNUNET_RESULTS_PATH):
        if len(list(p.glob(dataset_prefix + '*'))) > 0 and not overwrite:
            raise FileExistsError(f'{dataset_prefix} already exists...')

    print('>>> Successfully passed dataset id check', flush=True)

    return dataset_prefix


def to_nnunet_raw(
        dataset_name: str,
        split: dict,
        modality_loaders: dict,
        label_loader: Callable,
        labels: dict,
        affine_loader: Callable,
        nnunet_raw_path: PathLike = None,
        overwrite: bool = False,
        save_test_labels: bool = False,
        version: str = 'v2',
        ids_mapping: dict = None,
):
    nnunet_raw_path = NNUNET_RAW_PATH if nnunet_raw_path is None else Path(nnunet_raw_path)
    nnunet_raw_path.mkdir(exist_ok=True)

    dataset_path = nnunet_raw_path / dataset_name
    if dataset_path.exists() and not overwrite:
        raise FileExistsError(f'Trying to overwrite target [{str(dataset_path)}],'
                              'but `overwrite` is set to False.')
    dataset_path.mkdir(exist_ok=True)

    img_tr_path = dataset_path / 'imagesTr'
    img_tr_path.mkdir(exist_ok=True)
    img_ts_path = dataset_path / 'imagesTs'
    img_ts_path.mkdir(exist_ok=True)
    lbl_tr_path = dataset_path / 'labelsTr'
    lbl_tr_path.mkdir(exist_ok=True)

    lbl_ts_path = dataset_path / 'labelsTs'

    for _id in tqdm(split["train"]):
        _id_save = _id if (ids_mapping is None) else ids_mapping[_id]

        affine = affine_loader(_id)

        for i, (m_name, m_loader) in enumerate(modality_loaders.items()):
            save_nifti(m_loader(_id), affine, img_tr_path / f"{_id_save}_{i:04d}.nii.gz")

        save_nifti(label_loader(_id), affine, lbl_tr_path / f"{_id_save}.nii.gz")

    for _id in tqdm(split["test"]):
        _id_save = _id if (ids_mapping is None) else ids_mapping[_id]

        affine = affine_loader(_id)

        for i, (m_name, m_loader) in enumerate(modality_loaders.items()):
            save_nifti(m_loader(_id), affine, img_ts_path / f"{_id_save}_{i:04d}.nii.gz")

        if save_test_labels:
            lbl_ts_path.mkdir(exist_ok=True)
            save_nifti(label_loader(_id), affine, lbl_tr_path / f"{_id_save}.nii.gz")

    if version == 'v1':
        generate_dataset_json(output_file=str(dataset_path / 'dataset.json'),
                              imagesTr_dir=str(img_tr_path),
                              imagesTs_dir=str(img_ts_path),
                              modalities=tuple(modality_loaders.keys()),
                              labels={v: k for k, v in labels.items()},
                              dataset_name=dataset_name)
    elif version == 'v2':
        dataset_config = {
            "channel_names": {f"{i}": m for i, m in enumerate(modality_loaders)},
            "labels": labels,
            "numTraining": len(split["train"]),
            "file_ending": ".nii.gz"
        }
        save_json(dataset_config, dataset_path / 'dataset.json')


def save_split(
    dataset_name: str,
    split: list,
    nnunet_preprocessed_path: PathLike = None,
    plain_save: bool = False,
    remove_val: bool = False,
):
    nnunet_preprocessed_path = NNUNET_PREPROCESSED_PATH if nnunet_preprocessed_path is None\
        else Path(nnunet_preprocessed_path)
    if plain_save:
        split = split
    elif remove_val:
        split = [{"train": s[0], "val": s[2]} for s in split]
    else:
        split = [{"train": s[0], "val": s[1]} for s in split]
    save_json(split, Path(nnunet_preprocessed_path) / dataset_name / 'splits_final.json')
