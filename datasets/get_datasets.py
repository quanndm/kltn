from sklearn.model_selection import KFold
import pathlib
from .lits import Lits, Stage2Dataset
from ..processing.preprocessing  import extract_liver_mask_binary, resize_image, get_bbox_liver
from ..processing.postprocessing import keep_largest_connected_component, smooth_mask
import torch
import numpy as np
from ..utils.utils import model_inferer
import torch
import torch.nn.functional as F
import gc

def get_datasets_lits(source_folder, seed, fold_number = 5, normalizations = "zscores", mode = "all", model_stage_1=None, device=None):
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

    bbox_train = get_liver_mask_bbox(train, model_stage_1, device)
    bbox_test = get_liver_mask_bbox(test, model_stage_1, device)

    if mode == "tumor":
        train_dataset = Stage2Dataset(train, training=True, normalizations=normalizations, transformations=True, liver_masks_bbox = bbox_train)
        test_dataset = Stage2Dataset(test, training=False, normalizations=normalizations, liver_masks_bbox = bbox_train)
    else:
        train_dataset = Lits(train, training=True, normalizations=normalizations, transformations=True, mode=mode)
        test_dataset = Lits(test, training=False, benchmarking=True, normalizations=normalizations, mode=mode)

    return train_dataset, test_dataset


def get_liver_mask_bbox(source, model_stage_1=None, device=None):
    dataset = Lits(source, training=False, benchmarking=True, normalizations="zscores", mode="all")
    liver_masks_bbox = []

    if model_stage_1 is None:
        return None

    if model_stage_1 is not None:
        model_stage_1.eval()
        model_stage_1.to(device)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            image = data["image"].to(device)
            root_size = data["root_size"]
            image = image.unsqueeze(0)
            
            logits = model_inferer(image, model_stage_1)
            liver_mask = extract_liver_mask_binary(logits, threshold=0.4)
            _, liver_mask = resize_image(seg=liver_mask, target_size=root_size, device=device)
            bbox_liver = get_bbox_liver(np.squeeze(liver_mask, 0), margin=10)

            liver_masks_bbox.append(bbox_liver) 
            torch.cuda.empty_cache()
            gc.collect()
            del data, image, logits, liver_mask, bbox_liver

    return liver_masks_bbox