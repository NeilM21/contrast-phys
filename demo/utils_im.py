import os
import io
import aiofiles
import asyncio

import numpy as np
import cv2
from scipy.fftpack import fft, fft2, ifft2, fftshift, ifftshift, fftfreq
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from scipy.signal import iirnotch, filtfilt


def adaptive_fourier_masking(image, artifact_bpm, frame_rate, notch_radius, norm_mode):
    """
    Adaptive Fourier masking to remove the 125 BPM artifact from an RGB image.
    """
    artifact_frequency_hz = artifact_bpm / 60.0
    filtered_channels = []

    for channel in cv2.split(image):
        rows, cols = channel.shape
        center_row, center_col = rows // 2, cols // 2
        notch_distance = int(artifact_frequency_hz / frame_rate * np.sqrt(rows ** 2 + cols ** 2))

        dft = np.fft.fft2(channel)
        dft_shifted = np.fft.fftshift(dft)
        magnitude_spectrum = np.abs(dft_shifted)

        mask = np.ones((rows, cols), np.float32)
        for (cx, cy) in [(center_row + notch_distance, center_col),
                         (center_row - notch_distance, center_col),
                         (center_row, center_col + notch_distance),
                         (center_row, center_col - notch_distance)]:
            for i in range(rows):
                for j in range(cols):
                    d = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                    mask[i, j] *= 1 / (1 + (d / notch_radius) ** 2)  # Smooth attenuation

        filtered_dft_shifted = dft_shifted * mask
        filtered_dft = np.fft.ifftshift(filtered_dft_shifted)
        filtered_channel = np.fft.ifft2(filtered_dft)
        filtered_channel = np.abs(filtered_channel)
        if norm_mode == "clip":
            filtered_channel = np.clip(channel, 0, 255).astype(np.uint8)
        elif norm_mode == "minmax":
            filtered_channel = cv2.normalize(filtered_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        filtered_channels.append(filtered_channel)

    filtered_image_bgr = cv2.merge(filtered_channels)
    return filtered_image_bgr


def apply_notch_filter_rgb(image, artifact_bpm, frame_rate, notch_radius=5, norm_type=None, use_gauss_blur=False,
                           gauss_kernel=(5, 5)):
    """
    Apply a notch filter to an RGB image to remove artifacts at a specified BPM in the frequency domain.
    Note: This function returns an RGB image even if the input is in BGR format.

    Parameters:
    - image: np.array, Input BGR image.
    - artifact_bpm: float, The frequency in beats per minute (BPM) to filter out.
    - frame_rate: float, The frame rate of the source in frames per second (fps).
    - notch_radius: int, Radius of the notch filter around each center (default is 10).

    Returns:
    - filtered_image_rgb: np.ndarray, The RGB image after applying the notch filter.
    """
    # Convert BPM to frequency in Hz
    artifact_frequency_hz = artifact_bpm / 60.0

    # Initialize a list to store filtered channels
    filtered_channels = []

    # Split the BGR image into separate channels
    for channel in cv2.split(image):

        # Get image dimensions
        rows, cols = channel.shape
        center_row, center_col = rows // 2, cols // 2

        # Calculate notch centers in the Fourier domain
        notch_distance = int(artifact_frequency_hz / frame_rate * np.sqrt(rows ** 2 + cols ** 2))
        notch_centers = [
            (center_row + notch_distance, center_col),
            (center_row - notch_distance, center_col),
            (center_row, center_col + notch_distance),
            (center_row, center_col - notch_distance)
        ]

        # Perform FFT and shift the zero frequency component to the center
        dft = np.fft.fft2(channel)
        dft_shifted = np.fft.fftshift(dft)

        # Create a notch filter mask
        mask = np.ones((rows, cols), np.uint8)
        for (cx, cy) in notch_centers:
            cv2.circle(mask, (cy, cx), notch_radius, 0, -1)

        # Apply the mask to the shifted DFT
        filtered_dft_shifted = dft_shifted * mask

        # Shift back and perform the inverse FFT
        filtered_dft = np.fft.ifftshift(filtered_dft_shifted)
        filtered_channel = np.fft.ifft2(filtered_dft)
        filtered_channel = np.abs(filtered_channel)

        # Normalize the result to be in uint8 format
        if norm_type =="minmax":
            filtered_channel = cv2.normalize(filtered_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif norm_type == "clip":
            filtered_channel = np.clip(filtered_channel, 0, 255).astype(np.uint8)
       #filtered_channel = cv2.normalize(filtered_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Calculate the 1st and 99th percentiles for more robust scaling
        #v_min, v_max = np.percentile(filtered_channel, (1, 99))

        # Scale the values to fit in the 8-bit range
        #filtered_channel = (filtered_channel - v_min) / (v_max - v_min + 1e-8) * 255
        #filtered_channel = np.clip(filtered_channel, 0, 255).astype(np.uint8)

        # Append the filtered channel to the list
        filtered_channels.append(filtered_channel)

    # Merge the filtered channels back into a BGR image
    filtered_image = cv2.merge(filtered_channels)

    if use_gauss_blur:
        filtered_image = cv2.GaussianBlur(filtered_image, gauss_kernel, 0)

    return filtered_image


def channelwise_minmax_or_clip(input_im, technique):
    filtered_channels = list()
    # Split the BGR image into separate channels
    for channel in cv2.split(input_im):
        if technique == "minmax":
            filtered_channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif technique == "clip":
            filtered_channel = np.clip(channel, 0, 255).astype(np.uint8)

        # Append the filtered channel to the list
        filtered_channels.append(filtered_channel)

    # Merge the filtered channels back into a BGR image
    filtered_image = cv2.merge(filtered_channels)
    return filtered_image


def zscore_normalization(input_im):
    # ((x - mu)/ std)
    norm_im = (input_im - np.mean(input_im)) / np.std(input_im)
    return norm_im


def image_frequency_analysis(face_list: List[np.ndarray], subject_id: str, plots_dir: str, save_plots: bool) -> None:
    """
       Performs frequency analysis on a list of face images and generates frequency spectrum plots
       for the red, green, blue (RGB) channels separately and then their combined average. Saves the plots to the specified directory.

       Args:
           face_list (List[np.ndarray]): A list of face image arrays in the format (N, C, H, W),
                                         where N is the number of frames, C is the channel count (3 for RGB),
                                         and H, W are the height and width of the images.
           subject_id (str): Identifier for the subject, used in plot titles and filenames.
           plots_dir (str): Path to the directory where the frequency analysis plots should be saved.
           save_plots (bool): If True, the plots are saved to the specified directory.

       Returns:
           None
       """
    save_dir = f"{plots_dir}/freq_analysis"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if face_list.ndim > 4:
        face_list = np.transpose(face_list[0], (1, 2, 3, 0))

    average_frame_pixels_r = list()
    average_frame_pixels_g = list()
    average_frame_pixels_b = list()

    for face_frame in face_list:
        face_frame = Image.fromarray(face_frame)
        r, g, b = face_frame.split()

        r = np.array(r)
        g = np.array(g)
        b = np.array(b)

        average_frame_pixels_r.append(np.mean(r))
        average_frame_pixels_g.append(np.mean(g))
        average_frame_pixels_b.append(np.mean(b))

    N = len(face_list)

    all_channels = np.array([average_frame_pixels_r, average_frame_pixels_g, average_frame_pixels_b])
    average_pixels_all_channels = np.mean(all_channels, axis=0)

    # Remove DC component
    # average_frame_pixels_r = average_frame_pixels_r - np.mean(average_frame_pixels_r)
    # average_frame_pixels_g = average_frame_pixels_g - np.mean(average_frame_pixels_g)
    # average_frame_pixels_b = average_frame_pixels_b - np.mean(average_frame_pixels_b)

    frequencies_r = abs(fft(average_frame_pixels_r))
    frequencies_g = abs(fft(average_frame_pixels_g))
    frequencies_b = abs(fft(average_frame_pixels_b))
    frequencies_rgb = abs(fft(average_pixels_all_channels))
    fps = 25

    # Filter out non-biological frequencies outside the range [36..240] bpm, i.e. 0.6Hz-4Hz
    low_idx = np.round(0.6 / fps * frequencies_r.shape[0]).astype('int')
    high_idx = np.round(4 / fps * frequencies_r.shape[0]).astype('int')

    frequencies_r[:low_idx] = 0
    frequencies_r[high_idx:] = 0

    frequencies_g[:low_idx] = 0
    frequencies_g[high_idx:] = 0

    frequencies_b[:low_idx] = 0
    frequencies_b[high_idx:] = 0

    frequencies_rgb[:low_idx] = 0
    frequencies_rgb[high_idx:] = 0

    x_hr = np.arange(len(frequencies_r))/len(frequencies_r)*fps*60
    plt.subplot(2, 2, 1)
    plt.plot(x_hr, frequencies_r)
    plt.title("Frequency Spectrum of R Channel Averages")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim([0, 200])
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x_hr, frequencies_g)
    plt.title("Frequency Spectrum of G Channel Averages")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim([0, 200])
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x_hr, frequencies_b)
    plt.title("Frequency Spectrum of B Channel Averages")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim([0, 200])
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x_hr, frequencies_rgb)
    plt.title("Frequency Spectrum of All Channel (RGB) Averages")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.xlim([0, 200])
    plt.grid(True)

    plt.suptitle(f"{subject_id} Frequency Analysis")
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(16, 12)
    #plt.show()
    plt.savefig(f"{save_dir}/{subject_id}.png", dpi=100)
    plt.close()
    #print()


def notch_filter_images(face_list: np.ndarray, subject_id: str, notch_freq: float, fs: int) -> np.ndarray:
    """
    Applies a temporal notch filter to a list of face image frames to remove a specific frequency artifact
    (e.g., from a 125 BPM signal). The notch filter is applied to each pixel and color channel (R, G, B)
    separately over the time axis.

    Args:
        face_list (np.ndarray): A 4D array of face image frames in the format (N, H, W, C),
                                where N is the number of frames, H and W are the height and width
                                of the images, and C is the number of color channels (3 for RGB).
        subject_id (str): Identifier for the subject, used for tracking or logging purposes.
        notch_freq (float): The frequency of the artifact to be removed (in Hz).
        fs (int): The sampling frequency or frame rate of the video (in Hz).

    Returns:
        np.ndarray: A 4D array of the filtered face images, with the same shape as the input.
    """

    if face_list.ndim > 4:
        face_list = np.transpose(face_list[0], (1, 2, 3, 0))

    face_list_filtered = np.zeros(shape=face_list.shape, dtype=np.float32)
    # Get notch filter coefficients
    b_notch, a_notch = iirnotch(w0=notch_freq, Q=10.0, fs=fs)

    print(f"[PREPROCESSING] Applying notch filter (channel-wise) at {notch_freq}Hz to {subject_id}...")
    for channel in range(face_list.shape[3]):
        if channel == 1:
            face_list_filtered[:, :, :, channel] = face_list[:, :, :, channel]
            continue

        for row in range(face_list.shape[1]):
            for col in range(face_list.shape[2]):
                pixels = face_list[:, row, col, channel]
                face_list_filtered[:, row, col, channel] = filtfilt(b_notch, a_notch, pixels)

    face_list_filtered = (face_list_filtered - np.min(face_list_filtered)) / (np.max(face_list_filtered) - np.min(face_list_filtered))
    face_list_filtered = (face_list_filtered*255).astype(np.uint8)
    #plt.figure()
    #plt.title("Original Image", weight="bold")
    #plt.imshow(face_list[0])
    #'plt.figure()
    #plt.title("Filtered Image", weight="bold")
    #plt.imshow(face_list_filtered[0])
    #plt.show()
    return face_list_filtered


# ====================== ASYNC IMAGE SAVING FUNCTIONS ================================
async def async_save_image(image, file_path, ext, quality=75):
    # Save image to an in-memory buffer first
    buffer = io.BytesIO()
    image.save(buffer, format=ext, quality=quality, optimize=True)  # You can change the format (PNG, JPEG, etc.)

    # Use aiofiles to write the image to disk asynchronously
    async with aiofiles.open(file_path, mode='wb') as f:
        await f.write(buffer.getvalue())


async def save_images_async(image_batch, file_paths, file_formats):
    # List to hold async tasks
    tasks = []

    for image, path, ext in zip(image_batch, file_paths, file_formats):
        tasks.append(async_save_image(image, path, ext))

    # Run all tasks concurrently
    await asyncio.gather(*tasks)


# ===================================================== DEPRECATED =====================================================
def apply_band_reject_filter(channel_array, band_radius=2.08, filter_strength=0.2):
    # Perform FFT and shift the zero frequency component to the center
    f_transform = fftshift(fft2(channel_array))

    # Generate mesh grid for the filter
    rows, cols = channel_array.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt((X ** 2) + (Y ** 2))

    # Create the band-reject filter
    band_reject_filter = 1 - np.exp(-((D - band_radius) ** 2) / (2 * filter_strength ** 2))

    # Apply the filter to the frequency-transformed image
    filtered_f_transform = f_transform * band_reject_filter

    # Perform inverse FFT to get the filtered image
    filtered_image = np.abs(ifft2(fftshift(filtered_f_transform)))

    return filtered_image


def prepare_im_for_band_reject(input_im):
    image = Image.fromarray(input_im)
    r, g, b = image.split()

    # Convert each channel to a numpy array
    r_array = np.array(r)
    g_array = np.array(g)
    b_array = np.array(b)

    # Apply the filter to each channel
    filtered_r = apply_band_reject_filter(r_array)
    filtered_g = apply_band_reject_filter(g_array)
    filtered_b = apply_band_reject_filter(b_array)

    # Convert filtered arrays back to PIL Images
    filtered_r_img = Image.fromarray(np.uint8(filtered_r))
    filtered_g_img = Image.fromarray(np.uint8(filtered_g))
    filtered_b_img = Image.fromarray(np.uint8(filtered_b))

    # Merge the channels
    filtered_img_pil = Image.merge("RGB", (filtered_r_img, filtered_g_img, filtered_b_img))
    filtered_image_rgb = np.array(filtered_img_pil)

    return filtered_image_rgb


def band_reject_filter2(image, target_bpm, fs, width=10):
    """
    Apply a band-reject filter to an RGB image frame.

    Parameters:
    - image: Input RGB image as a NumPy array.
    - target_bpm: The frequency in BPM to attenuate.
    - fs: Sampling frequency (frame rate in Hz).
    - width: The width of the rejection band in the frequency domain.

    Returns:
    - filtered_image: The filtered RGB image.
    """

    rows, cols, _ = image.shape
    crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

    # Convert BPM to spatial frequency
    target_frequency = (target_bpm / 60) * fs  # Target frequency in Hz

    # Create a band-reject filter mask
    def create_filter(shape, center, radius, width):
        rows, cols = shape
        crow, ccol = center
        mask = np.ones((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if radius - width / 2 <= distance <= radius + width / 2:
                    mask[i, j] = 0
        return mask

    # Initialize the filtered image in the spatial domain
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Process each channel independently
    for channel in range(3):
        # Transform channel to frequency domain
        f = fft2(image[:, :, channel])
        fshift = fftshift(f)  # Shift zero-frequency to center

        # Create the band-reject filter
        mask = create_filter((rows, cols), (crow, ccol), target_frequency, width)

        # Apply the filter in the frequency domain
        fshift_filtered = fshift * mask

        # Inverse FFT to get back to the spatial domain
        f_ishift = ifftshift(fshift_filtered)
        img_back = ifft2(f_ishift)
        filtered_image[:, :, channel] = np.abs(img_back)

    # Normalize and clip
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image





