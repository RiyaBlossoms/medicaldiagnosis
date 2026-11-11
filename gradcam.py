"""
Grad-CAM implementation for EfficientNet-B3 feature maps.
Produces heatmaps highlighting regions driving the prediction.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global Average Pool over gradient spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam


def overlay_heatmap(cam: np.ndarray, original_img: Image.Image,
                     alpha: float = 0.4) -> np.ndarray:
    img_np = np.array(original_img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * img_np + alpha * heatmap)
    return overlay


def get_gradcam_for_model(model) -> GradCAM:
    """Attach Grad-CAM to the last conv block of EfficientNet-B3."""
    target = model.efficientnet.blocks[-1][-1].conv_pwl
    return GradCAM(model, target)
