import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

# Import the PhysNet model
# Note: This assumes the PhysNet class is defined in a local file
# If it's in a different module, adjust the import accordingly
from PhysNetModel import PhysNet
import h5py
import gc

matplotlib.use('TkAgg')


class GradCAMPlusPlusPhysNet:
    """
    Implementation of GRAD-CAM++ for visualizing CNN decisions in PhysNet (temporal rPPG model).
    Adapted to work with 3D convolutions and temporal data.
    """

    def __init__(self, model, layer_name):
        """
        Initialize GradCAM++ with a PhysNet model and target layer.

        Args:
            model: Pre-trained PhysNet model
            layer_name: Name of the target convolutional layer (e.g., 'start', 'loop1', etc.)
        """
        # Store the model
        self.model = model

        # Set the model to evaluation mode
        self.model.eval()

        # Store the device the model is on
        self.device = next(model.parameters()).device

        # Get the target layer
        self.target_layer = self._get_layer_by_name(model, layer_name)

        # Hook storage
        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _get_layer_by_name(self, model, layer_name):
        """
        Get a layer from the PhysNet model by its name.

        Args:
            model: PhysNet model
            layer_name: Name of the layer to retrieve

        Returns:
            The requested layer
        """
        # For PhysNet, we can directly access the modules
        if hasattr(model, layer_name):
            return getattr(model, layer_name)

        # If the layer is inside a sequential block, we need to navigate through it
        for name, module in model.named_modules():
            if name == layer_name:
                return module

        raise ValueError(f"Layer {layer_name} not found in the model")

    def _forward_hook(self, module, input, output):
        """
        Hook method to store activations during forward pass.
        """
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        """
        Hook method to store gradients during backward pass.
        """
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None, target_time=None):
        """
        Generate GRAD-CAM++ for a specific time point in the input video.

        Args:
            input_tensor: Preprocessed input tensor (batch of video frames)
            target_class: Index of the target class (if None, uses the highest activation)
            target_time: Specific time point to visualize (if None, uses the middle frame)

        Returns:
            Class activation map for the specified time point
        """
        # Ensure input is on the same device as model
        input_tensor = input_tensor.to(self.device)

        # Enable gradient computation
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)  # PhysNet output is (B, M, T) where M is spatial points + 1

        # If target_class is not specified, use the channel with highest activation
        if target_class is None:
            # Sum across temporal dimension
            temporal_sum = torch.sum(output, dim=2)
            # Get the channel with highest activation
            target_class = torch.argmax(temporal_sum).item()

        # If target_time is not specified, use the middle frame
        if target_time is None:
            target_time = output.size(2) // 2

        # Zero out gradients
        self.model.zero_grad()

        # Create a one-hot tensor for the target
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class, target_time] = 1

        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # For 3D convolutions (PhysNet), we need to handle the temporal dimension
        # Get the temporal slice corresponding to the target_time
        if len(activations.shape) == 5:  # (B, C, T, H, W)
            # Map the output time index to the activation time index
            # This mapping might be approximate due to temporal pooling/interpolation
            activation_time = int(target_time * activations.shape[2] / output.shape[2])

            # Get the activation and gradient for the specific time point
            act_slice = activations[:, :, activation_time, :, :]
            grad_slice = gradients[:, :, activation_time, :, :]
        else:
            # If activations don't have a temporal dimension, use as is
            act_slice = activations
            grad_slice = gradients

        # Calculate weights using the gradient information
        weights = torch.mean(grad_slice, dim=(2, 3), keepdim=True)

        # Apply ReLU to the weights
        weights = F.relu(weights)

        # Multiply weights with activations
        cam = torch.sum(weights * act_slice, dim=1, keepdim=True)

        # Apply ReLU to the CAM
        cam = F.relu(cam)

        # Normalize
        cam = cam / (torch.max(cam) + 1e-7)

        # Convert to numpy
        cam = cam[0, 0].detach().cpu().numpy()

        return cam

    def __call__(self, input_tensor, target_class=None, target_time=None):
        """
        Make the class callable.
        """
        return self.generate_cam(input_tensor, target_class, target_time)

    def remove_hooks(self):
        """
        Remove the hooks to free up memory.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()

        # Clear stored data
        self.activations = None
        self.gradients = None


def preprocess_video(video_frames, device='cpu'):
    """
    Preprocess video frames for the PhysNet model.

    Args:
        video_frames: List or numpy array of video frames
        device: Device to put the tensor on

    Returns:
        Preprocessed video tensor
    """
    # Convert to tensor if it's not already
    if not isinstance(video_frames, torch.Tensor):
        # Convert to float and normalize to [0, 1]
        if isinstance(video_frames, list):
            video_frames = np.array(video_frames)

        if video_frames.dtype == np.uint8:
            video_frames = video_frames.astype(np.float32) / 255.0

        # Convert to tensor
        video_tensor = torch.from_numpy(video_frames).float()
    else:
        video_tensor = video_frames

    # PhysNet expects input with shape (B, C, T, H, W)
    # If the input is (T, H, W, C), transpose it
    if video_tensor.dim() == 4 and video_tensor.shape[-1] == 3:  # (T, H, W, C)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

    # Add batch dimension if it's not present
    if video_tensor.dim() == 4:  # (C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # (B, C, T, H, W)

    # Move to device
    video_tensor = video_tensor.to(device)

    return video_tensor


def visualize_gradcam_temporal(video_frames, cam, target_time, alpha=0.6):
    """
    Create a visualization of the GRAD-CAM++ results for a specific time point.

    Args:
        video_frames: Original video frames
        cam: Class activation map
        target_time: Time point to visualize
        alpha: Transparency of the heatmap overlay

    Returns:
        original_frame: Original frame at the target time
        heatmap: GRAD-CAM++ heatmap
        superimposed_frame: Heatmap overlaid on the original frame
    """
    # Get the frame at the target time
    if isinstance(video_frames, torch.Tensor):
        if video_frames.dim() == 5:  # (B, C, T, H, W)
            frame = video_frames[0, :, target_time].permute(1, 2, 0)  # (H, W, C)
        else:  # (C, T, H, W)
            frame = video_frames[:, target_time].permute(1, 2, 0)  # (H, W, C)
        frame = frame.detach().numpy()
    else:
        frame = video_frames[target_time]

    # Convert to uint8 if it's float
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        frame = (frame * 255).astype(np.uint8)

    # Resize the CAM to match frame size
    cam_resized = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    # Apply colormap to the CAM
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure frame is in BGR format for cv2
    if frame.shape[-1] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame

    # Overlay the heatmap on the frame
    superimposed = cv2.addWeighted(frame_bgr, 1 - alpha, heatmap, alpha, 0)

    # Convert back to RGB for matplotlib
    original_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_frame = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    return original_frame, heatmap, superimposed_frame


def visualize_gradcam_temporal_sequence(subj_file, video_frames, model, layer_name, idx_range, target_class=None,
                                        num_frames=5, dataset_name="", loaded_weights=False, model_name =""):
    """
    Visualize GRAD-CAM++ for a sequence of frames from the video.

    Args:
        video_frames: Video frames
        model: PhysNet model
        layer_name: Name of the layer to visualize
        target_class: Target class to visualize (if None, uses the highest activation)
        num_frames: Number of frames to visualize

    Returns:
        None (displays the visualizations)
    """
    # Initialize GRAD-CAM++
    gradcam = GradCAMPlusPlusPhysNet(model, layer_name)

    # Preprocess video
    video_tensor = preprocess_video(video_frames, device=next(model.parameters()).device)

    # Get output for class prediction
    with torch.no_grad():
        output = model(video_tensor)

    # If target_class is not specified, use the channel with highest activation
    if target_class is None:
        temporal_sum = torch.sum(output, dim=2)
        target_class = torch.argmax(temporal_sum).item()
        print(f"Using channel with highest activation: {target_class}")

    # Get time points to visualize
    time_points = np.linspace(0, video_tensor.shape[2] - 1, num_frames, dtype=int)

    # Create a figure for visualization
    fig, axes = plt.subplots(num_frames, 3, figsize=(15, 5 * num_frames))

    for i, t in enumerate(time_points):
        # Generate CAM for this time point
        cam = gradcam(video_tensor, target_class=target_class, target_time=t)

        # Visualize
        original, heatmap, superimposed = visualize_gradcam_temporal(video_tensor, cam, t)
        plt.suptitle(f'{layer_name} GRAD-CAM++ Temporal Progression')

        # Plot
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f'Frame {t}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title(f'GRAD-CAM++ Heatmap (t={t})')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(superimposed)
        axes[i, 2].set_title(f'Overlay (t={t})')
        axes[i, 2].axis('off')

    if loaded_weights:
        save_path = f"gradcam_results/{model_name}/{dataset_name}/pretrained"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
    else:
        save_path = f"gradcam_results/{model_name}/{dataset_name}/random_init"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)


    plt.savefig(f'{save_path}/{subj_file}_gradcam_plus_plus_temporal_{layer_name}.png')
    plt.close(fig)  # Close the figure after saving

    # Remove hooks
    gradcam.remove_hooks()
    del gradcam, cam

def main():
    """
    Main function to demonstrate GRAD-CAM++ with PhysNet.
    """
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize PhysNet model
    model = PhysNet(S=2, in_ch=3).to(device)



    # Sample video frames (replace this with your actual video data)
    # For demonstration, we'll create a random video
    start_idx = 0
    end_idx = 128

    # Define datasets
    datasets = [
        {"name": "CAMVISIM", "path": "../cphys_data_nfiltered/camvisim_r2l_mts"},
        {"name": "UBFC", "path": "../cphys_data/UBFC"}
    ]

    # Define model configurations
    model_configs = [
        {"name": "pretrained", "load_weights": True},
        {"name": "non_pretrained", "load_weights": False}
    ]

    weights_path = 'model_weights/fresh_model_camvisimnfilt_lrsched/epoch5.pt'
    model_name = weights_path.split('/')[-2]

    for dataset in datasets:
        for model_weight_config in model_configs:
            if model_weight_config["load_weights"]:
                # Load weights if available
                try:
                    model.load_state_dict(torch.load(weights_path, map_location=device))
                    print("Model weights loaded successfully")
                except:
                    print("No pre-trained weights found. Using random initialization.")
            else:
                model = PhysNet(S=2, in_ch=3).to(device)
                print("Running with random initialization.")

            h5_dir = dataset["path"]
            all_subject_files = list(Path(h5_dir).rglob("*.h5"))

            for subject_filename in all_subject_files:
                with h5py.File(str(subject_filename)) as subject_file:  # Use with statement
                    video_frames = subject_file['imgs'][start_idx:end_idx].copy()
                    print(f"Video shape: {video_frames.shape}")
                    subj_idx = subject_filename.stem

                # Define the target layer
                # For PhysNet, we can use any of the convolutional layers
                #layer_names = ['loop4', "encoder2", "loop1"]  # Choose a layer to visualize
                layer_names = ['loop4']  # Choose a layer to visualize

                for layer_name in layer_names:
                    # Visualize GRAD-CAM++ for a sequence of frames
                    visualize_gradcam_temporal_sequence(subj_idx, video_frames, model, layer_name,
                                                        idx_range=[start_idx, end_idx], target_class=None, num_frames=5,
                                                        dataset_name=dataset["name"], loaded_weights=model_weight_config["load_weights"],
                                                        model_name=model_name)

                # Force cleanup
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                del video_frames


if __name__ == "__main__":
    main()