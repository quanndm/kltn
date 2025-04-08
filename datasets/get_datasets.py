from sklearn.model_selection import KFold
import pathlib
from .lits import Lits

def get_datasets_lits(source_folder, seed, fold_number = 5, normalizations = "zscores"):
    base_folder  = pathlib.Path(source_folder).resolve()

    # Get the list of volume the files in the folder
    volume_files = list(base_folder.glob('volume-*.nii.zip')) 

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

    # apply for dataset
    train_dataset = Lits(train, training=True, normalizations=normalizations, transformations=True)
    test_dataset = Lits(test, training=False, benchmarking=True, normalizations=normalizations)

    return train_dataset, test_dataset