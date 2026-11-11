"""
Hybrid Deep Learning Medical Diagnosis System
Entry point — runs training or inference based on CLI args.
"""


import argparse
from train import run_training
from inference import run_inference
from config import Config



def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid DL Medical Diagnosis System"
    )
    parser.add_argument(
        "--mode", choices=["train", "infer"],
        required=True, help="train or run inference"
    )
    parser.add_argument(
        "--disease",
        choices=["brain_tumor", "pneumonia", "skin_cancer", "eye_disease"],
        required=True,
    )
    parser.add_argument("--image", type=str, help="Path to image for inference")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    return parser.parse_args()




if __name__ == "__main__":

    print("PROGRAM STARTED")

    args = parse_args()

    print("ARGS PARSED")

    cfg = Config(disease=args.disease)

    print("CONFIG CREATED")

    if args.mode == "train":
        run_training(cfg)
    else:
        if not args.image or not args.checkpoint:
            raise ValueError("--image and --checkpoint required for inference")
        run_inference(cfg, args.image, args.checkpoint)

 