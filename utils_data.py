import numpy as np
import os
import h5py
import torch
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
from typing import List
from natsort import natsorted
from pathlib import Path


def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    #h5_dir = '../datasets/UBFC_h5'
    h5_dir = "cphys_data/UBFC"
    train_list = []
    val_list = []

    val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]

    for subject in range(1, 50):
        if os.path.isfile(h5_dir+'/%d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/%d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/%d.h5'%(subject))

    return train_list, val_list


def Camvisim_LU_split(k_fold=False, num_folds=5, fold_idx=0):
    # h5_dir = '../datasets/UBFC_h5'
    h5_dir = "cphys_data_nfiltered/camvisim_r2l_mts"
    train_list = []
    val_list = []
    train_folds = []

    if not k_fold:
        val_subject = ["025", "020", "019", "017", "016"]
        for subject in range(1, 26):
            subject_num_3digit = f"{subject:03}"
            subject_file = f"{h5_dir}/S{subject_num_3digit}.h5"
            if os.path.isfile(subject_file):
                if subject_num_3digit in val_subject:
                    val_list.append(subject_file)
                else:
                    train_list.append(subject_file)
        return train_list, val_list

    else:
        h5_list = Path(h5_dir).rglob("*.h5")
        subject_ids = [x.stem for x in h5_list]
        val_folds = crossval_create_folds(subject_ids=subject_ids, num_folds=num_folds)
        for fold in val_folds:
            val_list.append([f"{h5_dir}/{x}.h5" for x in fold if os.path.isfile(f"{h5_dir}/{x}.h5")])
            train_folds.append(natsorted(list(set(subject_ids) - set(fold))))

        for fold in train_folds:
            train_list.append([f"{h5_dir}/{x}.h5" for x in fold if os.path.isfile(f"{h5_dir}/{x}.h5")])

    return train_list[fold_idx], val_list[fold_idx]


def PURE_LU_split(k_fold=False, num_folds=5, fold_idx=0):
    # h5_dir = '../datasets/UBFC_h5'
    h5_dir = "cphys_data/pure"
    train_list = []
    val_list = []
    train_folds = []

    if not k_fold:
        val_subject = ["10", "09", "08"]
        for subject in range(1, 11):
            subject_num_pure = f"{subject:02}"
            subject_files = Path(h5_dir).rglob(f"{subject_num_pure}-*.h5")
            for sub_file in subject_files:
                sub_file = str(sub_file)
                if os.path.isfile(sub_file):
                    if subject_num_pure in val_subject:
                        val_list.append(sub_file)
                    else:
                        train_list.append(sub_file)
        return train_list, val_list

    else:

        h5_list = Path(h5_dir).rglob("*.h5")
        subject_ids = [x.stem.split('-')[0] for x in h5_list]
        subject_ids = natsorted(np.unique(subject_ids))
        val_folds = crossval_create_folds(subject_ids=subject_ids, num_folds=num_folds)
        train_folds = [natsorted(set(subject_ids) - set(x)) for x in val_folds]

        for train_fold, val_fold in zip(train_folds, val_folds):
            train_fold_subject_files = list()
            val_fold_subject_files = list()

            for t_subject_num_pure in train_fold:
                train_subject_files = Path(h5_dir).rglob(f"{t_subject_num_pure}-*.h5")
                train_fold_subject_files.extend(train_subject_files)

            for v_subject_num_pure in val_fold:
                val_subject_files = Path(h5_dir).rglob(f"{v_subject_num_pure}-*.h5")
                val_fold_subject_files.extend(val_subject_files)

            train_list.append([str(x) for x in train_fold_subject_files if os.path.isfile(x)])
            val_list.append([str(x) for x in val_fold_subject_files if os.path.isfile(x)])

    print()
    return train_list[fold_idx], val_list[fold_idx]


def crossval_create_folds(subject_ids, num_folds: int) -> List[np.ndarray]:
    """
    Creates cross-validation folds for a given dataset by splitting unique subject IDs into a specified number of folds.

    Args:
        dataset (PPGDataset): A dataset object that contains subject IDs for each sample.
                              It must have an attribute `all_subject_ids` that is a list or array of subject IDs.
        num_folds (int): The number of folds to create for cross-validation.

    Returns:
        List[np.ndarray]: A list of numpy arrays, where each array contains the subject IDs for one fold.

    """

    unique_subject_ids = np.array(list(natsorted(set(subject_ids))))
    num_patients = len(unique_subject_ids)
    partition_size_separator = num_patients % num_folds
    if partition_size_separator == 0:
        len_partitions = num_patients // num_folds
        folds = [unique_subject_ids[(x*len_partitions):(x+1)*len_partitions] for x in range(num_folds)]
    else:
        len_big_partitions = (num_patients // num_folds) + 1
        len_small_partitions = (num_patients // num_folds)
        fold_sizes = [len_big_partitions if x < partition_size_separator else len_small_partitions for x in range(num_folds)]
        folds = [unique_subject_ids[start:start + size] for start, size in zip([sum(fold_sizes[:i]) for i in range(len(fold_sizes))], fold_sizes)]
    return folds


def check_list_dimension(input_list):
    """
    Checks whether a primitive list is 1D or 2D.

    Parameters:
        input_list (list): The input list to check.

    Returns:
        str: "1D", "2D", or "Invalid".
    """
    if not isinstance(input_list, list):
        return "Invalid"

    # If no elements are lists, it's 1D
    if all(not isinstance(item, list) for item in input_list):
        return "1D"

    # If all elements are lists, and no sub-list contains further lists, it's 2D
    if all(isinstance(item, list) for item in input_list) and \
            all(all(not isinstance(sub_item, list) for sub_item in item) for item in input_list):
        return "2D"

    # Otherwise, it's invalid
    return "Invalid"


class H5Dataset(Dataset):

    def __init__(self, train_list, T, device):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length
        # Added - Neil
        self.device = device

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['imgs'].shape[0]

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            img_seq = torch.from_numpy(img_seq)
            img_seq = img_seq.to(self.device)
        return img_seq
