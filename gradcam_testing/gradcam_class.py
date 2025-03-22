import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()

        # Hooks to save activations and gradients
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_image):
        # Forward pass
        output = self.model(input_image)

        # Zero gradients and backward pass for the output vector (continuous output)
        self.model.zero_grad()

        # In Grad-CAM++ for regression, backpropagate based on the mean of the output vector
        target = output.mean()  # Assuming output is a vector, take its mean as the target scalar
        target.backward()

        # Gradients and activations for the target layer
        gradients = self.gradients[0]  # shape: (C, H, W)
        activations = self.activations[0]  # shape: (C, H, W)

        # Compute alpha, weight for each channel
        alpha_num = gradients.pow(2)
        alpha_denom = 2 * gradients.pow(2) + (activations * gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-7)

        positive_gradients = F.relu(gradients)

        # Weighted sum of activations and gradients
        weights = (alpha * positive_gradients).sum(dim=(1, 2))
        gradcam_map = F.relu((weights.view(-1, 1, 1) * activations).sum(dim=0))

        # Normalize the Grad-CAM map to [0, 1]
        gradcam_map -= gradcam_map.min()
        gradcam_map /= gradcam_map.max()
        return gradcam_map.cpu().numpy()