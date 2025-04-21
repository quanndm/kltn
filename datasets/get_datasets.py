from sklearn.model_selection import KFold
import pathlib
from .lits import Lits, Stage2Dataset

def get_datasets_lits(source_folder, seed, fold_number = 5, normalizations = "zscores", mode = "all", model_stage_1=None):
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

    if mode == "tumor":
        train_dataset = Stage2Dataset(train, training=True, normalizations=normalizations, transformations=True, model_stage_1=model_stage_1)
        test_dataset = Stage2Dataset(test, training=False, normalizations=normalizations, model_stage_1=model_stage_1)
    else:
        train_dataset = Lits(train, training=True, normalizations=normalizations, transformations=True, mode=mode)
        test_dataset = Lits(test, training=False, benchmarking=True, normalizations=normalizations, mode=mode)

    return train_dataset, test_dataset
