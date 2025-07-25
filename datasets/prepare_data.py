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
                

def prepare_dataset_LiTS(dir_name):
    download_dataset_LiTS()
    unzip_dataset_LiTS(dir_name)
    print("LiTS dataset prepared successfully.")


