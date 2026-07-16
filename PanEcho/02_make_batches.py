"""Create batch CSVs for embedding processing."""

# Authors: Platon Lukanenko, Bohan Jiang, William La Cava

import os
import pandas as pd 
from tqdm import tqdm

def main(
    batch_size=20,
    out_dir='./out_dir'
    ):
    """Create batch CSVs listing work items.

    Args:
        batch_size (int): Number of trim folders per batch file.
        out_dir (str): Base output directory created by setup step.
    """
    # set up csvs
    trim_folders = os.listdir(os.path.join(out_dir,'Incomplete'))
    trim_folders=[k[:-4] for k in trim_folders]
    
    # generate csvs in /Batches/ - each lists which csv's in /Incomplete/ this batch is responsibel for
    chunks = [trim_folders[i:i + batch_size] for i in range(0, len(trim_folders), batch_size)]
    for idx, chunk in tqdm(enumerate(chunks)):
        tmp_df = pd.DataFrame()
        tmp_df['batch_folder_names'] = chunk # list which files this batch processes
        tmp_df.to_csv(os.path.join(out_dir, 'Batches', f"{idx+1}.csv"),index=False)

import fire
if __name__ == '__main__':
    fire.Fire(main)
