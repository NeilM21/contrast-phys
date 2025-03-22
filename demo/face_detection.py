import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from utils_im import apply_notch_filter_rgb, adaptive_fourier_masking, apply_band_reject_filter, \
    prepare_im_for_band_reject, band_reject_filter2, zscore_normalization, channelwise_minmax_or_clip


def face_detection(video_path, mode, source, apply_filter, filter_type):

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    mtcnn = MTCNN(device=device)
    video_list = []

    if mode == "video":
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        N = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            N += 1

            #N Cap added for CAMVISIM extended testing videos
            if ret == True and N <= 3000:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_list.append(frame)
            else:
                break

            # if N/dataset_fps>60: # only get the first 60s video
            #     break

        cap.release()

    else:
        fps = 30
        video_frames_paths = list(Path(video_path).rglob("*.png"))
        for frame_path in video_frames_paths:
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_list.append(frame)

    face_list = []
    prev_coords = None
    progress_bar = tqdm(total=len(video_list), ascii=True, desc=f"Performing face detection...")
    for t, frame in enumerate(video_list):
        # Adjusted for dynamic detection for frequency analysis testing
        if t == 0:
            boxes, _, = mtcnn.detect(frame) # we only detect face bbox in the first frame, keep it in the following frames.
        if t == 0:
            if boxes is not None:
                prev_coords = boxes
            else:
                boxes = prev_coords

        if prev_coords is not None:
            box_len = np.max([boxes[0, 2] - boxes[0, 0], boxes[0, 3] - boxes[0, 1]])
            box_half_len = np.round(box_len / 2 * 1.1).astype('int')
            box_mid_y = np.round((boxes[0, 3]+boxes[0, 1])/2).astype('int')
            box_mid_x = np.round((boxes[0, 2]+boxes[0, 0])/2).astype('int')
            cropped_face = frame[box_mid_y-box_half_len:box_mid_y+box_half_len, box_mid_x-box_half_len:box_mid_x+box_half_len]
        else:
            cropped_face = frame

        if t % 250 == 0:
            progress_bar.update(250)

        cropped_face = cv2.resize(cropped_face, (128, 128))
        #cropped_face = np.true_divide(cropped_face, 255, dtype=np.float32)
        #plt.imshow(cropped_face)
        #plt.show()
        face_list.append(cropped_face)

    progress_bar.close()
    face_list = np.array(face_list) # (T, H, W, C)
    face_list = np.transpose(face_list, (3, 0, 1, 2)) # (C, T, H, W)
    face_list = np.array(face_list)[np.newaxis]

    return face_list, fps

"""
    elif mode == "video_select":
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if clip_select_start_time > 0:
            frame_seek_counter = 0
            success = cap.grab()

            while success and frame_seek_counter < clip_select_start_time:
                frame_seek_counter += 1
                success = cap.grab()

        ret, frame = cap.retrieve()
        N = 1
        while cap.isOpened():

            #TODO: N Cap added for CAMVISIM extended testing videos
            if ret == True and N <= 3000:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_list.append(frame)
                ret, frame = cap.read()
                N += 1
            else:
                break

            # if N/dataset_fps>60: # only get the first 60s video
            #     break

        cap.release()
        
    # ================================= FROM AFTER FD RESULT =================================================
        if apply_filter:
            if filter_type == "notch":
                cropped_face = apply_notch_filter_rgb(cropped_face, artifact_bpm=125, frame_rate=25, notch_radius=5,
                                                      norm_type="minmax", use_gauss_blur=False, gauss_kernel=(5, 5))

            elif filter_type == "band_reject":
                cropped_face = band_reject_filter2(cropped_face,target_bpm=125, fs=25)

            elif filter_type == "adaptive_fourier":
                cropped_face = adaptive_fourier_masking(cropped_face, artifact_bpm=125, frame_rate=25, notch_radius=5, norm_mode="clip")

            elif filter_type == "zscore_normalisation":
                cropped_face = zscore_normalization(cropped_face)

            elif filter_type == "minmax":
                cropped_face = channelwise_minmax_or_clip(cropped_face, technique="minmax") 
"""
