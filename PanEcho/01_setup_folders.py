"""Set up batch-processing folders for embedding generation."""

# Authors: Platon Lukanenko, Bohan Jiang, William La Cava

import os
import pandas as pd 
from tqdm import tqdm
import shutil
import fire

def main(
    src_dir,
    out_dir='./out_dir',
):
    """Create batch-processing folder structure for embeddings.

    Args:
        src_dir (str): Source directory containing trim folders.
        out_dir (str): Output directory to initialize.
    """
    if os.path.exists(out_dir):
        answer = (
            input(
                f"{out_dir} exists. If you continue, it will be deleted and overwritten. Continue? [y/N]: "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            shutil.rmtree(out_dir)
        else:
            print("Exiting")
            return
    # Set up subfolders
    os.makedirs(out_dir,exist_ok=False)
    os.makedirs(os.path.join(out_dir,'Batches'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'Incomplete'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'Complete'),exist_ok=True)
    os.makedirs(os.path.join(out_dir,'Embeddings'),exist_ok=True)
    
    # set up csvs
    trim_folders = os.listdir(src_dir)

                    
    # generate csvs in /Incomplete/
    os.makedirs(os.path.join(out_dir,'Incomplete'), exist_ok=True)
    for k in tqdm(trim_folders):
        tmp_csv = pd.DataFrame()
        tmp_csv['Folder_Path'] = [os.path.join(src_dir,k)]
        tmp_csv.to_csv(os.path.join(out_dir,'Incomplete',k+'.csv'),index=False)
    

if __name__ == '__main__':
    fire.Fire(main)
