# -*- coding: utf-8 -*-
"""Process batches to generate clip embeddings."""

# Authors: Platon Lukyanenko, William La Cava

import time

import pandas as pd
import os
import shutil
import random

import torch
from tqdm import tqdm
import cv2
import numpy as np
from torchvision import tv_tensors

from torchvision.transforms import v2


from torch.utils.data import Dataset, DataLoader
import h5py

import fire

# https://discuss.pytorch.org/t/speed-up-dataloader-using-the-new-torchvision-transforms-support-for-tensor-batch-computation-gpu/113166
Train_Transforms = torch.nn.Sequential(
    v2.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-15, 15)),
    v2.CenterCrop((224, 224)),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet norm RGB
)

Debug_Transforms_noNorm = torch.nn.Sequential(  # for visualizing
    v2.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=0.5),
    v2.RandomCrop(size=(224, 224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=(-15, 15)),
    v2.CenterCrop((224, 224)),
    # v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet norm RGB
)

Test_Transforms = torch.nn.Sequential(
    v2.CenterCrop((224, 224)),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet norm RGB
)


def pull_clips(file_loc, transform_func, num_clips=16):
    """Sample multiple random clips from a video and stack them.

    Args:
        file_loc (str): Path to a video file.
        transform_func (callable): Transform pipeline for clips.
        num_clips (int): Number of clips to sample.

    Returns:
        torch.Tensor: Stacked clips with shape (num_clips, 1, 3, 16, 224, 224).
    """
    # capture = cv2.VideoCapture(file_loc) # pull the file once, instead of x16...

    a = [pull_clip(file_loc, transform_func) for k in range(num_clips)]
    # time_taken_for_vid = time.time()-start_time
    return torch.vstack(a)  # , time_taken_for_vid


def pull_clip(file_loc, transform_func):
    """Sample a random 16-frame clip and apply transforms.

    Args:
        file_loc (str): Path to a video file.
        transform_func (callable): Transform pipeline for clips.

    Returns:
        torch.Tensor: Clip tensor shaped (1, 3, 16, 224, 224).
    """
    capture = cv2.VideoCapture(file_loc)
    clip_len = 16
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count < clip_len:
        start_idx = 0
    else:
        start_idx = np.random.randint(0, frame_count - clip_len + 1, size=1)[0]

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_idx - 1)

    v = []
    for i in range(clip_len):
        if i < frame_count:
            ret, frame = capture.read()
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            v.append(frame)
        else:
            v.append(frame)  # "last image carried forward"

    v = np.stack(v, axis=0)  # f x h x w x 3
    v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))  # f x 3 x h x w

    # debug_save_video(v.permute((0,2,3,1)).numpy(),'orig')

    # v = convex_hull_correction(v) # no convex hull correction here
    # debug_save_video(v.permute((0,2,3,1)).numpy(),'masked')

    v = v.type(torch.float32) / 255  # convert to 0-1 float32 format # added 5/1/25
    # debug_save_video((Debug_Transforms_noNorm(v)*255).type(torch.uint8).permute((0,2,3,1)).numpy(),'augmented')

    v = transform_func(
        v
    )  # can't do on GPU - not enough memory; also, these are cheap transforms

    v_as_input = v.unsqueeze(0).transpose(1, 2)

    return v_as_input


class MyDataset(Dataset):
    """Dataset of video file paths and transforms."""

    def __init__(self, path_list, transforms):
        """Initialize dataset of video file paths.

        Args:
            path_list (list[str]): Video file paths.
            transforms (callable): Transform pipeline for clips.
        """
        self.path_list = path_list
        self.transforms = transforms

    def __len__(self):
        """Return number of files in the dataset."""
        return len(self.path_list)

    def __getitem__(self, idx):
        """Return stacked clips and file path for an index.

        Args:
            idx (int): Dataset index.

        Returns:
            tuple[torch.Tensor, str]: Clips tensor and file path.
        """
        return pull_clips(self.path_list[idx], self.transforms), self.path_list[idx]

def main(
    batch_num, out_dir="./out_loc", seed=0, train_transforms=False, parallel_count=8
):
    """Process a batch of trim folders into embedding files.

    Args:
        batch_num (int): Batch ID to process.
        out_dir (str): Output directory containing batch files and folders.
        seed (int): RNG seed for reproducibility.
        train_transforms (bool): Use training transforms if True.
        parallel_count (int): Number of parallel workers for the dataloader.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # load PanEcho
    model = torch.hub.load(
        "CarDS-Yale/PanEcho", "PanEcho", force_reload=False, backbone_only=True
    )
    model.to("cuda")
    model.eval()

    # okay, we have a batch number to work with
    print(batch_num, type(batch_num))

    batch_csv_path = os.path.join(out_dir, "Batches", str(batch_num) + ".csv")
    batch = pd.read_csv(batch_csv_path)

    # transform set

    transforms_to_use = Train_Transforms if train_transforms else Test_Transforms

    # okay, inside batch
    for trim_n in batch["batch_folder_names"].astype(str):
        # per folder, if done, skip
        if (trim_n + ".csv") in os.listdir(os.path.join(out_dir, "Complete")):
            print("already processed", (trim_n + ".csv"))
            continue

        # not done -> process
        tmp_df = pd.read_csv(os.path.join(out_dir, "Incomplete", trim_n + ".csv"))
        file_folder_path = tmp_df["Folder_Path"].values[0]  # path of trim folder

        print(file_folder_path)
        file_path_list = [
            os.path.join(file_folder_path, file)
            for file in os.listdir(file_folder_path)
        ]

        print("files found", len(file_path_list))

        start = time.time()
        dataset = MyDataset(file_path_list, transforms_to_use)
        dataloader = DataLoader(
            dataset,
            batch_size=parallel_count,
            num_workers=parallel_count,
            shuffle=False,
        )
        # dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

        clip_list = []
        returned_path_list = []
        embedding_list = []
        for i, (clips, file_path) in enumerate(
            tqdm(dataloader)
        ):  # 64 x 16 x 3 x 16 x 224 x 224
            clip_list.append(clips)
            returned_path_list.append(file_path)

            with torch.no_grad():
                embedding_list = embedding_list + [
                    model(k.to("cuda")).to("cpu") for k in clips
                ]

        clip_list = torch.vstack(clip_list)  # yields [N<256] * 16 * 3 * 16 * 224 * 224
        embedding_list = torch.stack(embedding_list, axis=0)  # N x 16 x 768

        embed_path = os.path.join(out_dir, "Embeddings", trim_n + "_embed.hdf5")
        with h5py.File(embed_path, "w") as f:  # f[keys[0]][keys_2[0]]['emb'][()]
            study_group = f.create_group(trim_n)
            for file_path, emb in zip(file_path_list, embedding_list):
                sub_group = study_group.create_group(file_path)
                sub_group.create_dataset("emb", data=emb)

        time_taken = (time.time() - start) / clip_list.shape[0]
        print("folder embedding time per file", time_taken)

        # move the trim_n.csv to 'complete'
        src_dr = os.path.join(out_dir, "Incomplete", trim_n + ".csv")
        dst_dr = os.path.join(out_dir, "Complete", trim_n + ".csv")
        shutil.move(src_dr, dst_dr)



if __name__ == "__main__":
    fire.Fire(main)
