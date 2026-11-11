"""Global configuration dataclass."""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    disease: str = "brain_tumor"

    # Image settings
    img_size: int = 224
    num_channels: int = 3

    # Model
    efficientnet_variant: str = "efficientnet_b3"
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    fusion_hidden_dim: int = 512
    dropout: float = 0.3

    # Training
    batch_size: int = 20
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    seed: int = 42

    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"

    # Disease class maps
    class_maps: Dict[str, List[str]] = field(default_factory=lambda: {
        "brain_tumor":  ["glioma", "meningioma", "no_tumor", "pituitary"],
        "pneumonia":    ["normal", "pneumonia"],
        "skin_cancer":  ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "eye_disease":  ["cataract", "diabetic_retinopathy", "glaucoma", "normal"],
    })

    def num_classes(self) -> int:
        return len(self.class_maps[self.disease])

    def class_names(self) -> List[str]:
        return self.class_maps[self.disease]
