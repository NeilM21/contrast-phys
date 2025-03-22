import torch
import cv2
import numpy as np
import h5py
import torchvision.models as models
from gradcam_class import GradCAMPlusPlus
from PhysNetModel import PhysNet


def preprocess_image(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.float32(img) / 255.0
    # Normalize
    img = img - np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0)
    return img


device = torch.device('cpu')

# Load a pre-trained model (or your custom model)
model = PhysNet(S=2).to(device).eval()
model.load_state_dict(torch.load('../results/ubfc_pretrained_model_camvisimnfilt_freeze_toloop1_crossval/13/folds/1/epoch5.pt',
                                 map_location=device))

# Set the target layer (for ResNet, usually the last convolutional layer)
target_layer = model.decoder2[0]
gradcam = GradCAMPlusPlus(model, target_layer)

file = h5py.File('../cphys_data/UBFC/1.h5', 'r')
images = file['imgs'][:]
im_0 = np.transpose(images[0], (2, 0, 1))

# sample input shape to model: torch.Size([1, 3, 750, 128, 128])
# sample input shape to layer: torch.Size([1, 64, 750, 8, 8])
img_batch = images.transpose((3, 0, 1, 2))
img_batch = img_batch[np.newaxis].astype('float32')
img_batch = torch.tensor(img_batch).to(device, non_blocking=True)
heatmap = gradcam(img_batch)

heatmap = gradcam(im_0)

# Resize heatmap to match the original image
heatmap = cv2.resize(heatmap, (128, 128))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)