"""Single-image inference pipeline with Grad-CAM and clinical guidance."""
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from model import HybridMedicalModel
from dataset import get_transforms
from gradcam import get_gradcam_for_model, overlay_heatmap
from clinical_guidance import get_guidance


def run_inference(cfg: Config, image_path: str, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridMedicalModel(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gradcam = get_gradcam_for_model(model)

    # Preprocess image
    original = Image.open(image_path).convert("RGB")
    tfm = get_transforms(cfg, "val")
    tensor = tfm(original).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    # Forward pass
    with torch.enable_grad():
        logits = model(tensor)

    probs = F.softmax(logits, dim=1).squeeze()
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    pred_label = cfg.class_names()[pred_idx]

    # Grad-CAM heatmap
    cam = gradcam.generate(tensor, class_idx=pred_idx)
    overlay = overlay_heatmap(cam, original)

    # Clinical guidance
    guidance = get_guidance(cfg.disease, pred_label, confidence)

    # Print results
    print(f"\n{'='*60}")
    print(f"Disease domain : {cfg.disease}")
    print(f"Prediction     : {pred_label} ({confidence*100:.1f}% confidence)")
    print(f"Urgency        : {guidance.urgency.upper()}")
    print(f"Specialist     : {guidance.specialist}")
    print(f"Recommendations:")
    for r in guidance.recommendations:
        print(f"  • {r}")
    print(f"\nNOTE: {guidance.disclaimer}")
    print(f"{'='*60}\n")

    # Save visualisation
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original.resize((224, 224))); axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet");                axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay);                         axes[2].set_title(f"Overlay: {pred_label}")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_gradcam.png", dpi=150)
    plt.close()

    # Probability bar chart
    names = cfg.class_names()
    vals  = probs.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(names, vals, color=["#534AB7" if i == pred_idx else "#CECBF6"
                                        for i in range(len(names))])
    ax.set_xlim(0, 1); ax.set_xlabel("Probability")
    ax.set_title(f"Class probabilities — {cfg.disease}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_probs.png", dpi=150)
    plt.close()

    print(f"Results saved to {out_dir}")
    return guidance
