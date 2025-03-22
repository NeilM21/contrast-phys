from preprocessing import openface_h5
from test import get_lowest_ipr_epoch
from pathlib import Path
import os
import re
import h5py
import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from utils_sig import *
from typing import Tuple, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from natsort import natsorted
from train_model.evaluation.rppg_toolbox import metrics


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
            - gt_hrs_dict (dict): A dictionary where keys are file identifiers (stems or parent stems) and values are the average HR rounded to two decimal places.
            - gt_ppg_dict (dict): A dictionary where keys are file identifiers and values are extracted PPG data (if applicable).

    """
    hr_files = list(Path(trimmed_data_dir).rglob(f"*.{file_format}"))
    gt_hrs_dict = dict()
    gt_ppg_dict = dict()

    if file_format == "json":
        gt_hrs_dict = {x.stem[:5]: 0 for x in hr_files}
        gt_ppg_dict = {x.stem[:5]: 0 for x in hr_files}
        for hr_file in hr_files:
            with open(hr_file, "r") as ppg_data:
                gt_dict = json.load(ppg_data)
                gt_data = [label["Value"]["waveform"] for label in gt_dict["/FullPackage"]]
                #TODO: ask about HR resampling
                heart_rates = [label["Value"]["pulseRate"] for label in gt_dict["/FullPackage"]]
                gt_data = resample_data(np.array(gt_data), len(gt_dict["/Image"]))
                #heart_rates2 = resample_data(heart_rates, len(gt_dict["/Image"]))

            gt_hrs_dict[hr_file.parent.stem] = heart_rates
            gt_ppg_dict[hr_file.parent.stem] = gt_data

    elif file_format == "csv":
        gt_hrs_dict = {x.stem[:4]: 0 for x in hr_files}
        gt_ppg_dict = {x.stem[:4]: 0 for x in hr_files}
        for hr_file in hr_files:
            heart_rates = pd.read_csv(hr_file)["HR"]
            gt_hrs_dict[hr_file.stem[:4]] = heart_rates
            gt_ppg_dict[hr_file.stem[:4]] = pd.read_csv(hr_file)["PPG"]

    elif file_format == "txt":
        gt_hrs_dict = {x.parent.stem: 0 for x in hr_files}
        gt_ppg_dict = {x.parent.stem: 0 for x in hr_files}

        for hr_file in hr_files:
            with open(hr_file, "r") as ppg_data:
                gt_data = ppg_data.read()
                gt_data = gt_data.split("\n")
                gt_data = [[float(x) for x in y.split()] for y in gt_data]
                bvps = gt_data[0]
                heart_rates = gt_data[1]

            gt_hrs_dict[hr_file.parent.stem] = heart_rates
            gt_ppg_dict[hr_file.parent.stem] = bvps

    return gt_hrs_dict, gt_ppg_dict


def generate_landmarks(landmarks_folder: str, video_clip_list: List[Path], source_dataset: Optional[str] = "") -> None:
    """
    Generates facial landmarks for a list of video clips using the OpenFace tool and saves them to specified folders.

    Args:
        landmarks_folder (str): The path to the folder where landmarks will be saved.
        video_clip_list (List[Path]): A list of paths to video clips for which landmarks will be generated.
        source_dataset (Optional[str]): Name of the source dataset, such as "ubfc".
                                        If provided, will structure the landmarks folders accordingly.

    Returns:
        None
    """

    video_clip_list = [str(path).replace("//", '/') for path in video_clip_list]
    video_clip_list = [f'"{str(path)}"' for path in video_clip_list]
    landmarks_folder_original = landmarks_folder

    for idx, vid_path in enumerate(video_clip_list):
        if source_dataset == "ubfc":
            match = re.search(r'subject(/d+)', vid_path)
            subject_number = match.group(1)
            landmarks_folder = f"{landmarks_folder_original}/{subject_number}"
            if not os.path.isdir(landmarks_dir):
                os.makedirs(landmarks_dir)

        if source_dataset == "ubfc":
            match = re.search(r'subject(/d+)', vid_path)
            subject_number = match.group(1)
            landmarks_folder = f"{landmarks_folder_original}/{subject_number}"
            if not os.path.isdir(landmarks_dir):
                os.makedirs(landmarks_dir)

        os.system('.//openface//FeatureExtraction.exe -fdir %s -out_dir %s -2Dfp' % (vid_path, landmarks_folder))


def generate_h5_for_training(video_clip_list: List[Path], dataset: str, landmarks_path: str, h5_save_path: str) -> None:
    """
    Generates .h5 files for training from a list of videos and their corresponding landmarks.

    Args:
        video_clip_list (List[Path]): List of paths to video files.
        dataset (str): Mode of processing, such as "camvisim_r2l_mts" or "ubfc".
        landmarks_path (str): The path where landmarks are stored.
        h5_path (str): The directory path where the generated .h5 files will be saved.

    Returns:
        None
    """

    for v in tqdm(video_clip_list, desc="Processing videos"):
        if dataset == "camvisim_r2l_mts":
            video_name = v.stem
            video_landmark_path = f"{landmarks_path}/{video_name}.csv"
            h5_path = f"{h5_save_path}/{video_name.split('_')[0]}.h5"
            gt_path = str(list(v.parent.glob("*.csv"))[0])
            gt_df = pd.read_csv(gt_path)
            bvps = gt_df["PPG"].values

        elif dataset == "ubfc":
            match = re.search(r'subject(/d+)', str(v))
            subject_number = match.group(1)
            video_landmark_path = f"{landmarks_path}/{subject_number}/vid.csv"
            h5_path = f"{h5_save_path}/{subject_number}.h5"

            gt_path = f"{v.parent}/ground_truth.txt"
            with open(gt_path, "r") as ppg_data:
                gt_data = ppg_data.read()
                gt_data = gt_data.split("/n")
                gt_data = [[float(x) for x in y.split()] for y in gt_data]
                bvps = gt_data[0]

        elif dataset == "pure":
            subject_number = v.stem
            video_landmark_path = f"{landmarks_path}/{subject_number}.csv"
            h5_path = f"{h5_save_path}/{subject_number}.h5"

            gt_path = f"{v.parent}/{subject_number}.json"
            num_images = len(list(Path(gt_path).parent.rglob("*.png")))
            with open(gt_path, "r") as ppg_data:
                gt_data = json.load(ppg_data)
                bvps = np.array([label["Value"]["waveform"] for label in gt_data["/FullPackage"]])
                bvps = resample_data(bvps, target_length=num_images)

        openface_h5(video_path=str(v), landmark_path=video_landmark_path, h5_path=h5_path, bvps=bvps, store_size=128,
                    filter_ims=False, save_frames=True, mode=dataset)


def evaluate_model_cphys_eval(rppg_path: str, dataset: str, dataset_fps: int,source_hrs_dict: Dict[str, float]) -> None:
    """
    Evaluates the performance of an rPPG model by comparing predicted heart rates
    with ground truth heart rates. The function processes rPPG signals, computes
    heart rates using FFT, and saves evaluation results.

    Args:
        rppg_path (str): Path to the directory containing rPPG results.
        dataset (str): Name of the dataset (e.g., "ubfc", "camvisim_r2l_mts").
        dataset_fps (int): Frames per second of the dataset.
        source_hrs_dict (Dict[str, float]): Dictionary mapping subject IDs to
            their original heart rate values.

    Returns:
        None: The function saves results as plots and a CSV file but does not return a value.
    """
    plots_save_dir = f'{rppg_path}/plots/'
    if not os.path.isdir(plots_save_dir):
        os.makedirs(plots_save_dir)

    gt_hr_dict = dict()
    preds_hr_dict = dict()
    rmse_hrs_dict = dict()
    contrast_rmse_dict = dict()
    original_rmse_dict = dict()

    rmse_ppg_dict = dict()
    hrs_df = dict()

    # Only for Testing
    subject_files = list(Path(rppg_path).glob("*.npy"))

    for result_file in subject_files:
        if dataset == "ubfc":
            subject_id = f"subject{str(result_file.stem)}"
        else:
            subject_id = f"{result_file.name.split('.')[0]}"

        gt_hr_dict[subject_id] = list()
        preds_hr_dict[subject_id] = list()

        rppg_dict = np.load(str(result_file), allow_pickle=True)[()]
        gt = rppg_dict['bvp_list'].flatten()
        rppg = rppg_dict['rppg_list'].flatten()

        rmse_ppg_dict[subject_id] = np.sqrt(np.mean(np.square(gt-rppg)))

        gt = butter_bandpass(gt, lowcut=0.6, highcut=4, fs=dataset_fps)
        rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=dataset_fps)

        gt_hr, gt_psd_y, gt_psd_x = hr_fft(gt, fs=dataset_fps)
        rppg_hr, rppg_psd_y, rppg_psd_x = hr_fft(rppg, fs=dataset_fps)

        gt_norm = normalize(x=gt)
        rppg_norm = normalize(x=rppg)

        gt_psd_y_norm = normalize(x=gt_psd_y)
        rppg_psd_y_norm = normalize(x=rppg_psd_y)

        gt_hr_dict[subject_id] = np.round(gt_hr, 2)
        preds_hr_dict[subject_id] = np.round(rppg_hr, 2)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

        ax1.plot(np.arange(len(gt_norm)) / dataset_fps, gt_norm)
        ax1.plot(np.arange(len(rppg_norm)) / dataset_fps, rppg_norm)

        ax1.set_xlabel('time (sec)')
        ax1.grid('on')
        ax1.set_title('rPPG waveform')

        ax2.plot(gt_psd_x, gt_psd_y_norm)
        ax2.plot(rppg_psd_x, rppg_psd_y_norm)
        ax2.set_xlabel('heart rate (bpm)')
        ax2.set_xlim([40, 200])
        ax2.grid('on')
        ax2.set_title('PSD')

        plt.savefig(f'{rppg_path}/plots/{subject_id}.png')

        print(f'{subject_id} GT HR: {np.round(gt_hr, 2)}')
        print(f'{subject_id} Pred HR: {np.round(rppg_hr, 2)}')

    hrs_df["Subject Name"] = preds_hr_dict.keys()
    trimmed_source_hrs_dict = {key: source_hrs_dict[key] for key in gt_hr_dict.keys()}

    df = pd.DataFrame({
        'Subject Name': gt_hr_dict.keys(),
        'original_hr': trimmed_source_hrs_dict.values(),
        'gt_contrast_hr': gt_hr_dict.values(),
        'pred_contrast_hr': preds_hr_dict.values(),
    })

    df["contrast_mae"] = np.round(np.abs(df["gt_contrast_hr"] - df["pred_contrast_hr"]), 4)
    if dataset == "camvisim_r2l_mts":
        df["original_gt_mae"] = np.round(np.abs(df["original_hr"] - df["pred_contrast_hr"]), 4)
    df["ppg_rmse"] = np.round(list(rmse_ppg_dict.values()), 4)

    #df["contrast_rmse"] = contrast_rmse_dict.values()
    #df["original_gt_rmse"] = original_rmse_dict.values()

    #df["rmse_ppg"] = np.round(list(rmse_ppg_dict.values()), 4)
    numeric_means = np.round(df.iloc[:, 1:].mean(), 4)
    new_row = pd.DataFrame([['Average'] + numeric_means.tolist()], columns=df.columns)
    # Append the row and reset the index
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(f"{rppg_path}/results.csv")


def evaluate_model_grouped_eval(rppg_path: str, dataset: str, dataset_fps: int, source_hrs_dict: Dict[str, float]) -> None:
    plots_save_dir = f'{rppg_path}/plots/'
    if not os.path.isdir(plots_save_dir):
        os.makedirs(plots_save_dir)

    gt_hr_dict = dict()
    preds_hr_dict = dict()
    rmse_hrs_dict = dict()
    contrast_rmse_dict = dict()
    original_rmse_dict = dict()

    rmse_ppg_dict = dict()
    hrs_df = dict()

    gt_ppg_dict = dict()
    pred_ppg_dict = dict()


    # Only for Testing
    subject_files = list(Path(rppg_path).glob("*.npy"))

    for result_file in subject_files:
        if dataset == "ubfc":
            subject_id = f"subject{str(result_file.stem)}"
        else:
            subject_id = f"{result_file.name.split('.')[0]}"

        gt_hr_dict[subject_id] = list()
        preds_hr_dict[subject_id] = list()

        rppg_dict = np.load(str(result_file), allow_pickle=True)[()]
        gt = rppg_dict['bvp_list'].flatten()
        rppg = rppg_dict['rppg_list'].flatten()
        gt_ppg_dict[subject_id] = gt
        pred_ppg_dict[subject_id] = rppg

    model_path = result_file.parts[1]

    contrast_phys_eval_config = {
        # Data and model pipeline used up to generation of results
        'source_repository': "contrast-phys",
        "eval_mode": "rppg-toolbox",

        # Required parameters with default values
        'model_name': "Contrast-Phys",
        'model_path': model_path,
        'dataset_name': dataset,

        # Signal processing parameters
        'ppg_fs': dataset_fps,  # Sampling frequency in Hz
        'window_size': 10,  # Window size in seconds

        # Evaluation method parameters
        'fft_eval_method': ["fft", "peak detection"],  # 'peak detection', 'fft', or 'both'
        'peak_detector_mode': 'msptd',  # Peak detection algorithm
        'test_label_format': "raw",  # 'standardise', 'raw', or 'DiffNormalized'

        # Window and padding options
        'use_smaller_window': False,  # Whether to use windowed evaluation
        'use_signal_padding': False,  # Whether to pad signals for peak detection

        # Evaluation metrics and modes
        'eval_metrics': ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR'],  # Metrics to calculate
        'run_mode': "only_test",  # 'train_and_test' or 'only_test'
        'inf_mode': "test_data_inference",  # 'train_data_inference' or 'standard'

        # Cross-validation parameters
        'crossval_mode': crossval_mode,  # Whether in cross-validation mode
        'fold_number': None,  # Cross-validation fold number
        'run_name': model_path,  # Identifier for evaluation run
    }

    metrics.calculate_metrics(predictions=pred_ppg_dict, labels=gt_ppg_dict, acq_hrs=source_hrs_dict,
                              eval_config=contrast_phys_eval_config)

def print_h5(name, obj):
    print(name)


if __name__ == "__main__":
    create_landmarks = False
    create_h5_for_training = False
    train_cphys_model = False
    run_evaluation = True
    #evaluation_type = "contrast_phys"
    evaluation_type = "rppg_toolbox"

    mode = "camvisim_r2l_mts"
    #mode = "pure"
    #mode="ubfc"

    if mode == "camvisim_r2l_mts":
        videos_dir = "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps"
        gt_hrs_dir = "C://NIVS Project/src/CAMVISIM_master/video_trimming/GT Syncing/data_trimmed/exp2_nivs_labsubject_sync_timestamps"
        video_list = list(Path(videos_dir).glob("*/*.mts"))
        gt_format = "csv"
        fps = 25

    elif mode == "ubfc":
        videos_dir = "C://NIVS Project/data/external/UBFC Experiments/UBFC-rPPG/UBFC_DATASET/DATASET_2"
        gt_hrs_dir = "C://NIVS Project/data/external/UBFC Experiments/UBFC-rPPG/UBFC_DATASET/DATASET_2"

        video_list = list(Path(videos_dir).glob("*/*.avi"))
        gt_list = list(Path(videos_dir).glob("*/*.txt"))
        gt_format = "txt"
        fps = 30

    elif mode == "pure":
        gt_hrs_dir = "C://NIVS Project/data/external/PURE Dataset/Raw"
        videos_dir = "C://NIVS Project/data/external/PURE Dataset/Raw"
        gt_format = "json"

        video_list = [p for p in Path(videos_dir).glob("*/*") if p.is_dir()]
        gt_list = list(Path(videos_dir).glob("*/*.json"))
        fps = 30

    landmarks_dir = f"landmarks/{mode}"
    h5_dir = f"cphys_data/{mode}"

    if create_landmarks:
        if not os.path.isdir(landmarks_dir):
            os.makedirs(landmarks_dir)

        generate_landmarks(landmarks_folder=landmarks_dir, video_clip_list=video_list, source_dataset=mode)

    if create_h5_for_training:
        if not os.path.isdir(h5_dir):
            os.makedirs(h5_dir)

        generate_h5_for_training(video_clip_list=video_list, dataset=mode, landmarks_path=landmarks_dir, h5_save_path=h5_dir)

    # TODO: check logic for the below function for PURE and UBFC - currently relevant and reported for CAMVISIM only
    original_hrs_dict, ground_truth_ppg_dict = get_gt_data(gt_hrs_dir, file_format=gt_format)

    mean_original_hrs_dict = {x: np.round(np.mean(original_hrs_dict[x]), 2) for x in original_hrs_dict.keys()}

    # NOTE: run test.py before running this
    if run_evaluation:
        get_lowest_ipr_model = False
        # results_path = f"results/camvisim_r2l_mts/3/5"
        # results_path = "results/ubfc_train_test1/6/2"
        # Cphys2 on UBFC retrain - 14, Cphys2 nfiltered - 15
        # results_path = "results/ubfc_train_test1/6/15"
        # results_path = f"results/camvisim_train_cphys2_proper/1/1"
        # results_path = "results/'camvisim_train_cphys_nfiltered_proper_trainbenchmark_dataloader/15/2"
        # results_path = "results/ubfc_pretrained_model_on_camvisim_mts_nfiltered_data/1/1"
        # results_path = "results/ubfc_pretrained_model_on_camvisim_mts_nfiltered_data_lr_scheduling/1/1"
        # results_path = "results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1/1/2"
        # results_path = "results/ubfc_pretrained_model_camvisimnfilt_freeze_toencoder2/1/3"


        results_path = "results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_crossval/13/folds/best_ipr/2025_01_18_16-04-50"
        #results_path = "results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_test2_crossval_seed72/1/folds/best_ipr/2025_01_19_11-19-22"
        #results_path = "results/ubfc_pretrained_pt_repomodel/5"
        #results_path = "results/ubfc_pretrained_model_camvisimnfilt_retrain_nofreeze/3/folds/best_ipr/2025_01_22_19-58-50"
        #results_path = "results/fresh_model_camvisimnfilt_lrsched/1/folds/best_ipr/2025_01_25_11-02-48"
        #results_path = "results/fresh_model_pure_lrsched_corrected/1/folds/best_ipr/2025_01_30_20-44-53"
        #results_path = "results/ubfc_pretrained_model_pure_freeze_toloop1_crossval/1/folds/best_ipr/2025_02_01_08-56-07"
        #results_path = "results/ubfc_pretrained_model_camvisimnfilt_retrain_nofreeze/3/folds/best_ipr/2025_01_22_19-58-50"
        #results_path = "results/fresh_model_ubfc_lrsched/7/folds/best_ipr/2025_02_02_10-14-59"

        #results_path = "results/ubfc_pretrained_pt_repomodel/7"

        #results_path = "results/fresh_model_ubfc_lrsched/7/folds/best_ipr/2025_02_13_20-06-53"
        if "folds" in Path(results_path).parts:
            crossval_mode = True

        #if get_lowest_ipr_model:
        #    idx, ipr, best_ipr_model_path = get_lowest_ipr_epoch(train_model_path=results_path)
        #    results_path = best_ipr_model_path

        if evaluation_type == "contrast_phys":
            # CPhys Eval uses mean hrs directly for comparison
            evaluate_model_cphys_eval(rppg_path=results_path, dataset=mode, dataset_fps=fps,
                                      source_hrs_dict=mean_original_hrs_dict)
        else:
            evaluate_model_grouped_eval(rppg_path=results_path, dataset=mode, dataset_fps=fps,
                                      source_hrs_dict=original_hrs_dict)