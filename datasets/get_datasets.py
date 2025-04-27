from sklearn.model_selection import KFold
import pathlib
from .lits import Lits, Stage2Dataset
from ..processing.preprocessing  import extract_liver_mask_binary
from ..processing.postprocessing import keep_largest_connected_component, smooth_mask
import torch
import numpy as np
from ..utils.utils import model_inferer

def get_datasets_lits(source_folder, seed, fold_number = 5, normalizations = "zscores", mode = "all", liver_masks = None):
    """
    Get the datasets for the LiTS dataset.
    The function will return the training and testing datasets based on the fold number.
    The datasets are created using the Lits class from the lits module.
    Arguments:
    source_folder: str, the path to the folder containing the LiTS dataset.
    seed: int, the random seed for the KFold split.
    fold_number: int, the fold number for the KFold split.
    normalizations: str, the normalization method to be used. Default is "zscores".
    mode: str, all | liver | tumor
    """
    base_folder  = pathlib.Path(source_folder).resolve()

    # Get the list of volume the files in the folder
    volume_files = list(base_folder.glob('volume-*.nii')) 

    patients = []
    # Get the list of segmentation files in the folder, and match them with the volume files 
    for vol in volume_files:
        patient_id = vol.stem.split("-")[1]
        seg_file = base_folder / vol.name.replace("volume", "segmentation")
        patients.append({
            "id": patient_id,
            "volume": vol,
            "segmentation": seg_file
        })

    kfold = KFold(5, shuffle=True, random_state=seed)  
    splits = list(kfold.split(patients))

    train_idx, test_idx = splits[fold_number] 

    train = [patients[i] for i in train_idx]
    test = [patients[i] for i in test_idx]

    if liver_masks is not None:
        train_liver_masks = [liver_masks[i] for i in train_idx] 
        test_liver_masks = [liver_masks[i] for i in test_idx]

    if mode == "tumor":
        train_dataset = Stage2Dataset(train, training=True, normalizations=normalizations, transformations=True, liver_masks = train_liver_masks)
        test_dataset = Stage2Dataset(test, training=False, normalizations=normalizations, liver_masks = test_liver_masks)
    else:
        train_dataset = Lits(train, training=True, normalizations=normalizations, transformations=True, mode=mode)
        test_dataset = Lits(test, training=False, benchmarking=True, normalizations=normalizations, mode=mode)

    return train_dataset, test_dataset

def get_full_dataset(source_folder, normalizations = "zscores"):
    base_folder  = pathlib.Path(source_folder).resolve()

    # Get the list of volume the files in the folder
    volume_files = list(base_folder.glob('volume-*.nii')) 

    patients = []
    # Get the list of segmentation files in the folder, and match them with the volume files 
    for vol in volume_files:
        patient_id = vol.stem.split("-")[1]
        seg_file = base_folder / vol.name.replace("volume", "segmentation")
        patients.append({
            "id": patient_id,
            "volume": vol,
            "segmentation": seg_file
        })

    dataset = Lits(patients, training=False, normalizations=normalizations, mode="all")

    return dataset

def get_liver_mask(source_folder, model_stage_1=None, device=None):
    dataset = get_full_dataset(source_folder)
    liver_masks = []

    if model_stage_1 is None:
        return None

    if model_stage_1 is not None:
        model_stage_1.eval()
        model_stage_1.to(device)
    

    for i in range(len(dataset)):
        data = dataset[i]
        image = data["image"].to(device)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            logits = model_inferer(image, model_stage_1)
            liver_mask = extract_liver_mask_binary(logits, threshold=0.5)[0]
            liver_mask = keep_largest_connected_component(liver_mask)
            liver_mask = smooth_mask(liver_mask, kernel_size=3)
        liver_masks.append(liver_mask.float().cpu().numpy())
    return liver_masks