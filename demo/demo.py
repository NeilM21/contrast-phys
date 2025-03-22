import os
import cv2
import json
import torch
import h5py
import re
import asyncio
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Union, List, Tuple
from PIL import Image

from PhysNetModel import PhysNet
from utils_sig import *
from utils_im import image_frequency_analysis, notch_filter_images, save_images_async
from face_detection import face_detection
from scipy.stats import pearsonr

def read_h5(temp_hdf5_data, source_dataset):
    length = temp_hdf5_data['subject_data/gt_ppg_data'].shape[0]
    full_frame_batch_idx_list = temp_hdf5_data["subject_data/batch_and_frame_idx"][:].astype(str)
    if source_dataset != "pure":
        all_subject_ids = [x.split('_')[0] for x in full_frame_batch_idx_list]
    else:
        all_subclip_subject_ids = [x.split('_')[0] for x in full_frame_batch_idx_list]
        all_subject_ids = [x.split('-')[0] for x in full_frame_batch_idx_list]

    return all_subject_ids


def resample_data(input_signal, target_length):
    """Samples a PPG sequence into specific length."""
    return np.interp(
        np.linspace(1, input_signal.shape[0], target_length), np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal)


def get_gt_data(trimmed_data_dir: str, file_format: str) -> Tuple[dict, dict]:
    """
    Calculate average heart rates (HR) and ground truth photoplethysmography (PPG) data from files in a specified directory.

    This function processes files of various formats (`csv`, `txt`, and `json`) to compute average HR values and optionally extract PPG data.
    The files are expected to reside within the specified `trimmed_data_dir` directory, and their format is determined by the `file_format` parameter.

    Args:
        trimmed_data_dir (str or Path): The directory containing the input files.
        file_format (str): The format of the input files. Supported formats are:
            - "csv": Expects files with columns "HR" (heart rate) and "PPG" (photoplethysmography).
            - "txt": Expects files with numerical data, where the second line contains heart rates.
            - "json": Expects files containing a list of records with "pulseRate" (heart rate) and waveform data.

    Returns:
        tuple:
            - average_hrs_dict (dict): A dictionary where keys are file identifiers (stems or parent stems) and values are the average HR rounded to two decimal places.
            - gt_ppg_dict (dict): A dictionary where keys are file identifiers and values are extracted PPG data (if applicable).

    """
    hr_files = list(Path(trimmed_data_dir).rglob(f"*.{file_format}"))

    if file_format == "json":
        average_hrs_dict = {x.stem[:5]: 0 for x in hr_files}
        gt_ppg_dict = {x.stem[:5]: 0 for x in hr_files}
    else:
        average_hrs_dict = {x.stem[:4]: 0 for x in hr_files}
        gt_ppg_dict = {x.stem[:4]: 0 for x in hr_files}

    #average_hrs_dict = {x.parent.name: 0 for x in hr_files}

    # Calculate mean HR for each file and update the dictionary
    if file_format == "csv":
        average_hrs_dict = {x.stem[:4]: 0 for x in hr_files}
        for hr_file in hr_files:
            mean_hr = pd.read_csv(hr_file)["HR"].mean()
            average_hrs_dict[hr_file.stem[:4]] = np.round(mean_hr, 2)
            gt_ppg_dict[hr_file.stem[:4]] = pd.read_csv(hr_file)["PPG"]

    elif file_format == "txt":
        average_hrs_dict = {x.parent.stem: 0 for x in hr_files}
        gt_ppg_dict = {x.parent.stem: 0 for x in hr_files}

        for hr_file in hr_files:
            with open(hr_file, "r") as ppg_data:
                gt_data = ppg_data.read()
                gt_data = gt_data.split("/n")
                gt_data = [[float(x) for x in y.split()] for y in gt_data]
                bvps = gt_data[0]
                heart_rates = gt_data[1]

            mean_hr = np.mean(heart_rates)
            average_hrs_dict[hr_file.parent.stem] = np.round(mean_hr, 2)
            gt_ppg_dict[hr_file.parent.stem] = bvps

    elif file_format == "json":
        average_hrs_dict = {x.parent.stem: 0 for x in hr_files}
        gt_ppg_dict = {x.parent.stem: 0 for x in hr_files}
        for hr_file in hr_files:
            with open(hr_file, "r") as ppg_data:
                gt_dict = json.load(ppg_data)
                gt_data = [label["Value"]["waveform"] for label in gt_dict["/FullPackage"]]
                #TODO: ask about HR resampling
                heart_rates = [label["Value"]["pulseRate"] for label in gt_dict["/FullPackage"]]
                gt_data = resample_data(np.array(gt_data), len(gt_dict["/Image"]))
                #heart_rates2 = resample_data(heart_rates, len(gt_dict["/Image"]))

            mean_hr = np.mean(heart_rates)
            average_hrs_dict[hr_file.parent.stem] = np.round(mean_hr, 2)
            gt_ppg_dict[hr_file.parent.stem] = gt_data

    return average_hrs_dict, gt_ppg_dict


def configure_mode(mode: str) -> Dict[str, Union[str, List[str], None]]:
    """
    Configures and returns mode-specific settings for video and ground truth directories.

    Args:
        mode (str): The dataset type to be used.

    Returns:
        Dict[str, Union[str, List[str], None]]: A dictionary containing settings for the specified mode, including:
            - "gt_hrs_dir" (str): Path to ground truth heart rate directory.
            - "videos_dir" (str): Path to videos directory.
            - "video_patterns" (Optional[List[str]]): File patterns to search for video files (e.g., "*.mts").
            - "gt_format" (str): Format of ground truth data (e.g., "csv", "txt", "json").
            - "fd_mode" (str): Mode for face detection (e.g., "video" or "image").
            Returns an empty dictionary if the mode is unsupported.
    """

    settings = {
        "camvisim_mts_exp2": {
            "gt_hrs_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps",
            "videos_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps",
            "video_patterns": ["*.mts"],
            "gt_format": "csv",
            "fd_mode": "video",
            "fps": 25,
        },
        "camvisim_mts_original": {
            "gt_hrs_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps",
            "videos_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_original",
            "video_patterns": ["*.mts"],
            "gt_format": "csv",
            "fd_mode": "video",
            "fps": 25,
        },
        "camvisim_mts_mdh_original": {
            "gt_hrs_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps",
            "videos_dir": "C://NIVS Project/data/nivs_mdh/raw",
            "video_patterns": ["*.mts"],
            "gt_format": "csv",
            "fd_mode": "video",
            "fps": 25,
        },
        "camvisim_mts_original_ideal": {
            "gt_hrs_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps",
            "videos_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_original",
            "video_patterns": ["*.mts"],
            "gt_format": "csv",
            "fd_mode": "video_select",
            "fps": 25,
        },
        "camvisim_lab_phone": {
            "gt_hrs_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps",
            "videos_dir": "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_original",
            "video_patterns": ["*.mov", "*.mp4"],
            "gt_format": "csv",
            "fd_mode": "video",
            "fps": 30,
        },
        "ubfc": {
            "gt_hrs_dir": "C://NIVS Project/data/external/UBFC Experiments/UBFC-rPPG/UBFC_DATASET/DATASET_2",
            "videos_dir": "C://NIVS Project/data/external/UBFC Experiments/UBFC-rPPG/UBFC_DATASET/DATASET_2",
            "video_patterns": ["*.avi"],
            "gt_format": "txt",
            "fd_mode": "video",
            "fps": 30,
        },
        "pure":{
            "gt_hrs_dir": "C://NIVS Project/data/external/PURE Dataset/Raw",
            "videos_dir": "C://NIVS Project/data/external/PURE Dataset/Raw",
            "video_patterns": None,  # PURE has directories instead of files
            "gt_format": "json",
            "fd_mode": "image",
            "fps": 30,
        },
    }
    return settings.get(mode, {})


def run_image_analysis(face_list: np.ndarray, subject_id: str, plots_save_dir: str):
    """
    Analyzes an array of video frames by performing frequency analysis using FFT and saves plots.

    This function processes a list of video frames, prepares a sample frame for visualization,
    and calls a frequency analysis function to identify and analyze frequencies in the images.
    The results, including a sample image plot, are saved to the specified directory.

    Args:
        face_list (np.ndarray): An  array of video frames.
        subject_id (str): A unique identifier for the subject, used in plot titles and filenames.
        plots_save_dir (str): The directory where plots and analysis results will be saved.
                              If the directory does not exist, it is created.

    Returns:
        None

    """
    sample_frame = face_list[0]

    if sample_frame.shape == (3, 3000, 128, 128):
        sample_frame = np.transpose(sample_frame, (1, 2, 3, 0))[0]

    if sample_frame.dtype == np.float32:
        sample_frame = (sample_frame*255.0).astype(np.uint8)

    if not os.path.isdir(plots_save_dir):
        os.makedirs(plots_save_dir)

    # plt.show()
    plt.imshow(sample_frame)
    plt.title(f'{subject_id}  frame for analysis', fontweight="bold")
    plt.savefig(f"{plots_save_dir}/sample_{subject_id}.png", dpi=100)
    plt.close()
    image_frequency_analysis(face_list, subject_id=subject_id, save_plots=True, plots_dir=plots_save_dir)


def read_cphys_openface_h5(vid_path: str):
    """
    Reads an H5 file containing video frames and extracts facial images along with the subject ID.

    Args:
        vid_path (str): The path to the H5 file containing video data. The file name is expected
                        to contain a subject identifier in the format "S<number>".

    Returns:
        tuple: A tuple containing:
            - face_list (numpy.ndarray): A 5D array of extracted facial images with shape
              (1, num_frames, height, width, channels), where `num_frames` is the total
              number of frames in the video.
            - subject_id (str): A string representing the subject ID in the format "Subject <number>".

    Raises:
        AttributeError: If the subject identifier (e.g., "S003") is not found in the file path.
    """

    s_search = re.search(r'S(\d+)', vid_path)
    s_number = s_search.group(1)
    if not s_search:
        raise AttributeError(f"Subject ID for video {vid_path} not found")

    subject_id = f"Subject {s_number}"
    file = h5py.File(vid_path, 'r')
    face_list = file['imgs'][:]
    face_list = np.transpose(face_list, (3, 0, 1, 2))
    face_list = np.expand_dims(face_list, axis=0)
    return face_list, subject_id


def normal_demo():
    mode = "camvisim_mts_exp2"
    filter_images = True
    #read_data_mode = "cphys_h5"
    read_data_mode = None
    save_frames = False

    settings = configure_mode(mode)
    gt_hrs_dir = settings.get("gt_hrs_dir")
    videos_dir = settings.get("videos_dir")
    video_patterns = settings.get("video_patterns")
    gt_format = settings.get("gt_format")
    fd_mode = settings.get("fd_mode")
    fps = settings.get("fps")

    # Gather video paths
    if video_patterns:
        videos_path = []
        for pattern in video_patterns:
            videos_path.extend(Path(videos_dir).glob(f"*/*.{pattern.split('.')[-1]}"))
    else:
        videos_path = [f for f in Path(videos_dir).iterdir() if f.is_dir()]

    # Optional filtering
    if mode == "camvisim_lab_phone":
        videos_path = [x for x in videos_path if int((str(x).split("\\")[-1])[1:4]) != 13]

    # Convert to string paths
    videos_path = [str(x) for x in videos_path]

    # Load ground truth heart rates
    original_hrs_dict, gt_ppg_dict = get_gt_data(gt_hrs_dir, file_format=gt_format)

    # Initialize predictions dictionary
    if mode == "pure":
        preds_hr_dict = {x.split('\\')[-1]: 0 for x in videos_path}
    else:
        preds_hr_dict = {x.split('\\')[-2]: 0 for x in videos_path}

    # Read from existing H5 files generated via OpenFace and the directions in train.py
    if read_data_mode == "cphys_h5":
        videos_path = "../cphys_data_nfiltered/CAMVISIM_R2L_MTS"
        videos_path = Path(videos_path).glob("*.h5")

    demo_run_name = "camvisim_mts_trimmed_tempnotch_testing"
    plots_save_dir = f"./plots/{demo_run_name}/new_viz2"
    save_frames_dir = f"{plots_save_dir}/frames"
    hrs_df = pd.DataFrame(columns=["Subject_Name", "camvisim_hr", "gt_contrast_hr", "pred_contrast_hr", "Diff_camvisim", "Diff_cphys", "ppg_rmse", "ppg_pearson"])
    start_times = None
    # Read source videos/images
    for vid_idx, vid_path in enumerate(videos_path):
        vid_path = str(vid_path)
        if read_data_mode != "cphys_h5":
            subject_id = vid_path.split('\\')[-2]
            if mode == "pure":
                subject_id = vid_path.split('\\')[-1]

            face_list, fps2 = face_detection(video_path=vid_path, mode=fd_mode, source=mode, apply_filter=False,
                                            filter_type="notch")

        else:
            face_list, subject_id = read_cphys_openface_h5(vid_path=vid_path)

        if filter_images:
            face_list = notch_filter_images(face_list=face_list, subject_id=subject_id, notch_freq=2.08, fs=25)

        if save_frames:
            # For cases when the array shape is (1, C, T, 128, 128), becomes (T, 128, 128, C)
            if not os.path.isdir(save_frames_dir):
                os.makedirs(save_frames_dir)

            if face_list.ndim > 4:
                face_list = np.transpose(face_list[0], (1, 2, 3, 0))

            all_im_data_pil = [Image.fromarray(x) for x in face_list]
            paths = [f"{save_frames_dir}/{subject_id}_{x}.jpg" for x in range(len(all_im_data_pil))]
            formats = ["JPEG" for _ in range(len(all_im_data_pil))]
            asyncio.run(save_images_async(image_batch=all_im_data_pil, file_paths=paths, file_formats=formats))

        #run_image_analysis(face_list=face_list, subject_id=subject_id, plots_save_dir=plots_save_dir)
        #continue

        print(f'{subject_id}: rPPG estimation')
        #device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            device = torch.device('cpu')
        else:
            device = torch.device('cpu')

        #TODO: CHECK UBFC EVAL ORDER VS GT ORDER
        with torch.no_grad():
            if face_list.ndim == 4:
                face_list = np.transpose(face_list, (3, 0, 1, 2))
                face_list = np.expand_dims(face_list, axis=0)

            face_list = torch.tensor(face_list.astype('float32')).to(device)
            model = PhysNet(S=2).to(device).eval()
            model.load_state_dict(torch.load('./model_weights.pt', map_location=device))
            #model.load_state_dict(torch.load('./ubfctraintest1_6_epoch29.pt', map_location=device))
            #model.load_state_dict(torch.load('../results/ubfc_pretrained_model_on_camvisim_mts_nfiltered_data/1/epoch4.pt', map_location=device))

            rppg = model(face_list)[:, -1, :]
            rppg = rppg[0].detach().cpu().numpy()
            rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

        #if subject_id == "Subject 007":
        #    print()

        # Attenuating 125bpm artefact for MTS files
        if mode == "camvisim" and not filter_images:
            print(f"[POSTPROCESSING] Applying notch filter to prediction rPPG for subject {subject_id}")
            rppg = iirnotch(in_signal=rppg, sample_freq=25.0, notch_freq=2.083, q_factor=10.0)

        gt = butter_bandpass(gt_ppg_dict[list(gt_ppg_dict)[vid_idx]].values, lowcut=0.6, highcut=4, fs=fps)
        fft_pred_hr, pred_psd_y, pred_psd_x = hr_fft(rppg, fs=fps)
        fft_gt_hr, gt_psd_y, gt_psd_x = hr_fft(gt, fs=fps)
        ppg_rmse = np.round(np.sqrt(np.mean(np.square(gt - rppg))), 4)
        ppg_pearson = np.round(np.corrcoef(gt, rppg)[0, 1], 4)

        if not os.path.isdir(plots_save_dir):
            os.makedirs(plots_save_dir)

        preds_hr_dict[subject_id] = np.round(fft_pred_hr, 2)
        original_hr_val = original_hrs_dict[list(original_hrs_dict)[vid_idx]]
        camvisim_hr_diff = np.round(np.abs(fft_pred_hr - original_hr_val), 4)
        cphys_hr_diff = np.round(np.abs(fft_pred_hr - fft_gt_hr), 4)
        hrs_df.loc[len(hrs_df)] = [subject_id, original_hr_val, np.round(fft_gt_hr, 2), np.round(fft_pred_hr, 2),
                                   camvisim_hr_diff, cphys_hr_diff, ppg_rmse, ppg_pearson]
        hrs_df.to_csv(f"{plots_save_dir}/{demo_run_name}.csv")



        gt = normalize(x=gt)
        rppg = normalize(x=rppg)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))
        #TODO: Add overlay of GT PPG and optionally HR_FFT RESULT to compare repoare owner pretrained moel and inhouse

        ax1.plot(np.arange(len(gt))/fps, gt, label='gt')
        ax1.plot(np.arange(len(rppg))/fps, rppg, label='pred')
        ax1.set_xlabel('time (sec)')
        ax1.grid('on')
        ax1.set_title('rPPG waveform')
        ax1.legend(loc="upper left")

        ax2.plot(gt_psd_x, gt_psd_y, label='gt')
        ax2.plot(pred_psd_x, pred_psd_y, label='pred')
        ax2.set_xlabel('heart rate (bpm)')
        ax2.set_xlim([40, 200])
        ax2.grid('on')
        ax2.set_title('PSD')
        ax2.legend(loc="upper left")

        plt.savefig(f'{plots_save_dir}/{subject_id}.png')
        print(f'{subject_id} HR: {np.round(fft_pred_hr, 2)}')
        plt.close()

    print("Done!")


def camvisim_master_h5_demo():
    gt_hrs_dir = "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps"
    h5_path = "C://NIVS Project/src/CAMVISIM_master/preprocessing/h5_data/camvisim_lab_rgb_end_exp2.h5"
    temp_hdf5_data = h5py.File(h5_path, mode="r")
    subj_ids = read_h5(temp_hdf5_data, "camvisim_lab")
    unique_ids = sorted(set(subj_ids))
    original_hrs_dict, gt_ppg_dict = get_gt_data(gt_hrs_dir, file_format='csv')
    average_hrs_dict = {x: 0 for x in unique_ids}
    save_frames = True
    filter_images = True

    # Create a dictionary to store index ranges for each unique subject ID
    subject_index_ranges = {}
    fps = 25
    # Step 2: Find index ranges of each subject ID in subj_ids
    demo_run_name = "camvisim_master_h5_tempnotch_sig_testing"
    plots_save_dir = f"./plots/{demo_run_name}"
    save_frames_dir = f"{plots_save_dir}/frames"

    hrs_df = pd.DataFrame(columns=["Subject_Name", "camvisim_hr", "gt_contrast_hr", "pred_contrast_hr", "Diff_camvisim", "Diff_cphys", "ppg_rmse", "ppg_pearson"])
    preds_hr_dict = {x: 0 for x in unique_ids}
    for subj_id in unique_ids:
        # Find the indexes where the subject ID occurs
        indexes = np.where(np.array(subj_ids) == subj_id)[0]

        # Store the range (start, end) of the occurrence
        subject_index_ranges[subj_id] = (indexes[0], indexes[-1])
        face_list = temp_hdf5_data["subject_data/face_data"][indexes[0]:indexes[-1]+1]
        face_list = np.transpose(face_list, (3, 0, 1, 2))  # (C, T, H, W)
        face_list = np.array(face_list)[np.newaxis]

        if filter_images:
            face_list = notch_filter_images(face_list=face_list, subject_id=subj_id, notch_freq=2.08, fs=25)

        if save_frames:
            # For cases when the array shape is (1, C, T, 128, 128), becomes (T, 128, 128, C)
            if not os.path.isdir(save_frames_dir):
                os.makedirs(save_frames_dir)

            if face_list.ndim > 4:
                face_list = np.transpose(face_list[0], (1, 2, 3, 0))

            all_im_data_pil = [Image.fromarray(x) for x in face_list]
            paths = [f"{save_frames_dir}/{subj_id}_{x}.jpg" for x in range(len(all_im_data_pil))]
            formats = ["JPEG" for _ in range(len(all_im_data_pil))]
            asyncio.run(save_images_async(image_batch=all_im_data_pil, file_paths=paths, file_formats=formats))

        run_image_analysis(face_list=face_list, subject_id=subj_id, plots_save_dir=plots_save_dir)

        print(f'{subj_id}: rPPG estimation')
        device = torch.device('cuda')

        start_time = time.time()
        with torch.no_grad():
            if face_list.ndim == 4:
                face_list = np.transpose(face_list, (3, 0, 1, 2))
                face_list = np.expand_dims(face_list, axis=0)

            face_list = torch.tensor(face_list.astype('float32')).to(device)
            model = PhysNet(S=2).to(device).eval()
            model.load_state_dict(torch.load('./model_weights.pt', map_location=device))
            rppg = model(face_list)[:, -1, :]
            rppg = rppg[0].detach().cpu().numpy()
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Time taken for {subj_id}: {processing_time} seconds")
            rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

        gt = butter_bandpass(gt_ppg_dict[f"{subj_id}"], lowcut=0.6, highcut=4, fs=fps)
        rppg = iirnotch(in_signal=rppg, sample_freq=25.0, notch_freq=2.083, q_factor=10.0)
        fft_pred_hr, psd_y, psd_x = hr_fft(rppg, fs=fps)
        fft_gt_hr, gt_psd_y, gt_psd_x = hr_fft(gt, fs=fps)
        ppg_rmse = np.round(np.sqrt(np.mean(np.square(gt - rppg))), 4)
        ppg_pearson = np.round(np.corrcoef(gt, rppg)[0, 1], 4)

        average_hrs_dict[subj_id] = np.round(fft_pred_hr, 2)
        preds_hr_dict[subj_id] = np.round(fft_pred_hr, 2)
        original_hr_val = original_hrs_dict[subj_id]
        camvisim_hr_diff = np.round(np.abs(fft_pred_hr - original_hr_val), 4)
        cphys_hr_diff = np.round(np.abs(fft_pred_hr - fft_gt_hr), 4)

        hrs_df.loc[len(hrs_df)] = [subj_id, original_hr_val, np.round(fft_gt_hr), np.round(fft_pred_hr, 2),
                                   camvisim_hr_diff, cphys_hr_diff, ppg_rmse, ppg_pearson]
        hrs_df.to_csv(f"{plots_save_dir}/{demo_run_name}.csv")
        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        ax1.plot(np.arange(len(rppg)) / fps, rppg)
        ax1.set_xlabel('time (sec)')
        ax1.grid('on')
        ax1.set_title('rPPG waveform')

        ax2.plot(psd_x, psd_y)
        ax2.set_xlabel('heart rate (bpm)')
        ax2.set_xlim([40, 200])
        ax2.grid('on')
        ax2.set_title('PSD')
        plt.savefig(f'{plots_save_dir}/psd_{subj_id}.png')
        plt.close()
        #plt.savefig(f'./plots/{subj_id}.png')

        print(f'{subj_id} HR: {np.round(fft_pred_hr, 2)}')
    print("Done")

    # Define directories and settings based on the mode


#camvisim_master_h5_demo()
normal_demo()



#get_gt_data("C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps")
"""
 # TODO: remove after "background image testing"
            face_list = np.array([x[0:300, 0:180, :] for x in face_list])
            a = face_list[0]
            plt.imshow(a)
            plt.title(f'{subject_id} background crop for analysis', fontweight="bold")
            #plt.show()
            plt.savefig(f"{plots_save_dir}/{subject_id}_bkg.png", dpi=100)
            #face_list = diff_normalize_data(np.transpose(face_list[0], (1, 2, 3, 0)))
            image_frequency_analysis(face_list, subject_id=subject_id, save_plots=True, plots_dir=plots_save_dir)
            face_list = []
            continue
"""