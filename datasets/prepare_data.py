import requests
import libtorrent as lt
import zipfile
import os
import sys
import time
import re
import tarfile
import shutil
import gzip
import nibabel as nib

def download_dataset_LiTS():
    url = "https://academictorrents.com/download/27772adef6f563a1ecc0ae19a528b956e6c803ce.torrent"
    filename = "27772adef6f563a1ecc0ae19a528b956e6c803ce.torrent"
    response = requests.get(url)
    
    with open(filename, 'wb') as file:
        file.write(response.content)

    ses = lt.session({'listen_interfaces': '0.0.0.0:6881'})
    info = lt.torrent_info(filename)

    params = {
        "ti": info,
        "save_path": "."
    }

    h = ses.add_torrent(params)
    s = h.status()

    while (not s.is_seeding):
        s = h.status()

        print('\r%.2f%% complete (down: %.1f kB/s up: %.1f kB/s peers: %d) %s' % (
            s.progress * 100, s.download_rate / 1000, s.upload_rate / 1000,
            s.num_peers, s.state), end=' ')

        alerts = ses.pop_alerts()
        for a in alerts:
            if a.category() & lt.alert.category_t.error_notification:
                print(a)

        sys.stdout.flush()

        time.sleep(1)

    print(h.status().name, 'complete')

def unzip_dataset_LiTS(dir_name):
    extension = ".zip"
    print(f"\nUnzipping files in {dir_name}")
    for item in os.listdir(dir_name): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = f"{dir_name}/{item}" # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_name) # extract file to dir
            zip_ref.close() # close file
            os.remove(file_name) # delete zipped file

    print("Unzipping complete")

def delete_ircad_files(dir_name):
    """
    Delete the ircad files in the directory
    """
    print("delete data from ircad")
    ircad_ids = set(range(27, 49)) 
    pattern = re.compile(r"(volume|segmentation)-(\d+)\.nii\.zip$") 

    for item in os.listdir(dir_name):
        match = pattern.match(item)
        if match:
            patient_id = int(match.group(2))
            if patient_id in ircad_ids:
                file_path = os.path.join(dir_name, item)
                os.remove(file_path)
                print(f"Deleted {file_path}")
                
def compress_nii_to_gz(dir_name):
    """
    Compress the .nii files to .gz files
    """
    print("Compressing .nii files to .gz")
    nii_files = [f for f in os.listdir(dir_name) if f.endswith('.nii')]
    for nii_file in nii_files:
        nii_path = os.path.join(dir_name, nii_file)
        gz_path = os.path.join(dir_name, nii_file + '.gz')
        with open(nii_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(nii_path)

def prepare_dataset_LiTS(dir_name):
    download_dataset_LiTS()
    delete_ircad_files(dir_name)
    unzip_dataset_LiTS(dir_name)
    compress_nii_to_gz(dir_name)
    print("LiTS dataset prepared successfully.")


def download_and_prepare_msd(dir_name):
    """
    Download and prepare the MSD dataset
    """
    file_tar = "Task03_Liver.tar"
    url = f"https://msd-for-monai.s3-us-west-2.amazonaws.com/{file_tar}"
    tar_path = os.path.join(dir_name, file_tar)
    extract_path = "Task03_Liver"
    response = requests.get(url)
    
    with open(filename, 'wb') as file:
        file.write(response.content)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dir_name)

    test_dir = os.path.join(dir_name, extract_path, "imagesTs")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Deleted {test_dir}")

    # remove the tar file
    os.remove(tar_path)
    print("Successfully downloaded and extracted the MSD dataset.")


def merge_lits_and_msd(lits_dir, msd_dir, output_dir):
    """
    Merge the LiTS and MSD datasets
    """
    def copy_and_prefix_files(src_dir, files, prefix, dst_dir, subfolder=None):
        """
        Copy files from src_dir to dst_dir with a prefix added to the filenames.
        Optionally, specify a subfolder within src_dir.
        """
        for file in files:
            src = os.path.join(src_dir, subfolder, file) if subfolder else os.path.join(src_dir, file)
            dst = os.path.join(dst_dir, f"{prefix}-{file}")
            shutil.copy(src, dst)
            os.remove(src)
            print(f"Copied {src} to {dst}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Copy and prefix LiTS files
    lits_files = os.listdir(lits_dir)
    copy_and_prefix_files(lits_dir, lits_files, "lits", output_dir)

    # Copy and prefix MSD image and label files
    msd_image_files = os.listdir(os.path.join(msd_dir, "imagesTr"))
    msd_label_files = os.listdir(os.path.join(msd_dir, "labelsTr"))
    copy_and_prefix_files(msd_dir, msd_image_files, "msd", output_dir, subfolder="imagesTr")
    copy_and_prefix_files(msd_dir, msd_label_files, "msd", output_dir, subfolder="labelsTr")
