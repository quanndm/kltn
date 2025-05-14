from sklearn.model_selection import KFold
import pathlib
from .lits import Lits, Stage2Dataset, Stage2Dataset2D
from .custom_dataset import CustomDatasetLiver
from ..processing.preprocessing  import extract_liver_mask_binary, resize_image, get_bbox_liver, get_liver_roi
import torch
import numpy as np
from ..utils.utils import model_inferer
import torch
import torch.nn.functional as F
import gc
import os

def get_datasets_lits(source_folder, seed, fold_number = 5, normalizations = "zscores", mode = "all", device=None, liver_masks_bbox = None):
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

    bbox_train = None if liver_masks_bbox is None else [liver_masks_bbox[i] for i in train_idx]
    bbox_test = None if liver_masks_bbox is None else [liver_masks_bbox[i] for i in test_idx]

    if mode == "tumor":
        train_dataset = Stage2Dataset(train, training=True, normalizations=normalizations, transformations=True, liver_masks_bbox = bbox_train)
        test_dataset = Stage2Dataset(test, training=False, normalizations=normalizations, liver_masks_bbox = bbox_test)
    else:
        train_dataset = Lits(train, training=True, normalizations=normalizations, transformations=True, mode=mode)
        test_dataset = Lits(test, training=False, benchmarking=True, normalizations=normalizations, mode=mode)

    return train_dataset, test_dataset

def get_full_dataset_lits(source_folder, normalizations = "zscores", mode = "all", device=None):
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

    dataset = Lits(patients, training=False, normalizations=normalizations, mode=mode)

    return dataset





###################################################################
def get_full_datasets(source_folder, normalizations="zscores"):
    """
    Arguments:
    source_folder: str, the path to the folder containing the LiTS + MSD dataset.
    normalizations: str, the normalization method to be used. Default is "zscores".
    """
    base_folder  = pathlib.Path(source_folder).resolve()

    # Get the list of volume the files in the folder
    volume_files = list(base_folder.glob('*volume*.nii*')) 

    patients = []
    # Get the list of segmentation files in the folder, and match them with the volume files 
    for vol in volume_files:
        # lits-volume-1.nii | msd-volume-1.nii.gz 
        filename = vol.name
        if filename.endswith('.nii.gz'):
            name = filename[:-7]  # bỏ '.nii.gz'
        else:
            name = filename.rsplit('.', 1)[0]  # bỏ '.nii'
        parts = name.split('-')
        source = parts[0]  # lấy phần đầu tiên
        patient_id = parts[2]
        seg_file = base_folder / vol.name.replace("volume", "segmentation")
        patients.append({
            "source": source,
            "id": patient_id,
            "volume": vol,
            "segmentation": seg_file
        })

    datasets = CustomDatasetLiver(patients, training=False, normalizations=normalizations)

    return datasets

def get_datasets_stage_1(source_folder, seed, fold_number=5, normalizations="zscores"):
    """
    Get the datasets for the LiTS dataset.
    The function will return the training and testing datasets based on the fold number.
    The datasets are created using the Lits class from the lits module.
    Arguments:
    source_folder: str, the path to the folder containing the LiTS dataset.
    seed: int, the random seed for the KFold split.
    fold_number: int, the fold number for the KFold split.
    normalizations: str, the normalization method to be used. Default is "zscores".
    """
    base_folder  = pathlib.Path(source_folder).resolve()

    # Get the list of volume the files in the folder
    volume_files = list(base_folder.glob('*volume*.nii*')) 

    patients = []
    # Get the list of segmentation files in the folder, and match them with the volume files 
    for vol in volume_files:
        # lits-volume-1.nii.gz | msd-volume-1.nii.gz
        filename = vol.name
        if filename.endswith('.nii.gz'):
            name = filename[:-7]  # bỏ '.nii.gz'
        else:
            name = filename.rsplit('.', 1)[0]  # bỏ '.nii'
        parts = name.split('-')
        source = parts[0]  # lấy phần đầu tiên
        patient_id = parts[2]
        seg_file = base_folder / vol.name.replace("volume", "segmentation")
        patients.append({
            "source": source,
            "id": patient_id,
            "volume": vol,
            "segmentation": seg_file
        })

    kfold = KFold(5, shuffle=True, random_state=seed)  
    splits = list(kfold.split(patients))

    train_idx, test_idx = splits[fold_number] 

    train = [patients[i] for i in train_idx]
    test = [patients[i] for i in test_idx]

    # Custom dataset for stage 1
    train_dataset = CustomDatasetLiver(train, training=True, normalizations=normalizations, transformations=True)
    test_dataset = CustomDatasetLiver(test, training=False, normalizations=normalizations)

    return train_dataset, test_dataset

def get_liver_mask_bbox(source, model_stage_1=None, device=None):
    # dataset = get_full_dataset_lits(source, normalizations="zscores", mode="all", device=device)
    dataset = get_full_datasets(source, normalizations="minmax")
    liver_masks_bbox = []
    patients_id = []
    sources = []
    if model_stage_1 is None:
        return None

    if model_stage_1 is not None:
        model_stage_1.eval()
        model_stage_1.to(device)
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            source = data["source"]
            image = data["image"].to(device)
            root_size = data["root_size"]
            image = image.unsqueeze(0)
            patients_id.append(data["patient_id"])
            
            logits = model_inferer(image, model_stage_1)
            liver_mask = extract_liver_mask_binary(logits, threshold=0.4)
            _, liver_mask = resize_image(seg=liver_mask, target_size_seg=root_size)
            bbox_liver = get_bbox_liver(np.squeeze(liver_mask, 0), margin=12)

            sources.append(source)
            liver_masks_bbox.append(bbox_liver) 
            torch.cuda.empty_cache()
            gc.collect()
            del data, image, logits, liver_mask, bbox_liver

    return sources, patients_id, liver_masks_bbox # (patients_id, bbox_liver)

def convert_to_2D_dataset(source, bbox, slides = 3, stride = 2, save_dir = "/content/2D_dataset"):
    """
    Convert the 3D dataset to 2.5D dataset.
    Arguments:
    source: str, the path to the folder containing the LiTS dataset.
    bbox: the bounding boxes of the liver ROI crops - full datasets. [ {sources: str, patient_id: int, bbox: list[int]},]
    slides: int, the number of slides to be extracted from each volume.
    save_dir: str, the path to the folder where to save the converted dataset.
    type: str, the type of dataset to be converted, lits | all. Default is "lits".
    """
    base_folder  = pathlib.Path(source).resolve()
    volume_files = list(base_folder.glob('*volume*.nii*')) 
    radius = slides // 2
    os.makedirs(save_dir, exist_ok=True)
    stride = stride

    for i in range(len(volume_files)):
        vol = volume_files[i]

        source = vol.stem.split("-")[0]
        patient_id = vol.stem.split("-")[2]
        seg_file = base_folder / vol.name.replace("volume", "segmentation")
        image = Lits.load_nii(vol)
        seg = Lits.load_nii(seg_file)

        # Get the liver ROI from the image and segmentation
        bb = next((item["bbox"] for item in bbox if item["patient_id"] == int(patient_id)) and source == item["source"], None)
        image, seg = get_liver_roi(image, seg, bb)
        
        # Save the image and segmentation as 2D slices
        D, H, W = image.shape
        for z in range(radius, D - radius, stride):
            # Extract image slices and segmentation slices
            image_slice = image[z - radius: z + radius + 1, :, :]  # shape (slides, H, W)
            seg_slice = seg[z - radius: z + radius + 1, :, :] .astype(np.uint8)  # shape (slides, H, W)

            # Save the slices
            np.savez_compressed(f"{save_dir}/{source}_patient_{patient_id}_slice_{z:03d}.npz", image=image_slice, seg=seg_slice, bbox=np.array(bb))
        print(f"Patient {patient_id} processed.")

def get_datasets_2d(source_folder, seed, fold_number=5, normalizations="zscores"):
    """
    Get the datasets for the LiTS dataset.
    The function will return the training and testing datasets based on the fold number.
    The datasets are created using the Lits class from the lits module.
    Arguments:
    source_folder: str, the path to the folder containing the LiTS dataset.
    seed: int, the random seed for the KFold split.
    fold_number: int, the fold number for the KFold split.
    normalizations: str, the normalization method to be used. Default is "zscores".
    """
    base_folder  = pathlib.Path(source_folder).resolve()

    # get npz files
    files = list(base_folder.glob('*patient_*.npz')) 

    patients = []
    # Get the list of segmentation files in the folder, and match them with the volume files 
    # lits_patient_1_slice_001.npz | msd_patient_1_slice_001.npz
    for file in files:
        patient_id = file.stem.split("_")[2]
        slide = file.stem.split("_")[4]
        source = file.stem.split("_")[0]
        patients.append({
            "source": source,
            "id": patient_id,
            "slide": slide,
            "file": file
        })

    kfold = KFold(5, shuffle=True, random_state=seed)  
    splits = list(kfold.split(patients))

    train_idx, test_idx = splits[fold_number] 

    train = [patients[i] for i in train_idx]
    test = [patients[i] for i in test_idx]

    train_dataset = Stage2Dataset2D(train, training=True, normalizations=normalizations, transformations=True)
    test_dataset = Stage2Dataset2D(test, training=False, normalizations=normalizations)
    return train_dataset, test_dataset
