from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from network import UNet, load_gan_models
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt



@torch.no_grad()
def inference(dataset_path: Path, model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))
    
    generator, _discriminator = load_gan_models(model_path)
    
    generator.to(device)
    generator.eval()

    transform = transforms.Compose([
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
        ])
    
    for i, image_path in enumerate(dataset_path.rglob("*.png")):
        # if i > 10:
        #     break
        
        print(f"Processing {image_path}")
        gray_img = Image.open(image_path)
        
        gray_img = gray_img.convert("L")
        gray_tensor = transform(gray_img)
        gray_tensor = gray_tensor.unsqueeze(0).to(device)
        
        assert gray_tensor.shape == (1, 1, 512, 1024), f"Input shape mismatch: {gray_tensor.shape}"
        
        with torch.no_grad():
            output_tensor = generator(gray_tensor)
            
        output = output_tensor.squeeze().cpu().numpy()
        
        assert output.shape == (3, 512, 1024), f"Output shape mismatch: {output.shape}"

        output_img = np.transpose(output, (1, 2, 0))
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)

        Image.fromarray(output_img).save(f"output_predictions/{image_path.name}")
    

# #### code below should not be changed ############################################################################
def main() -> None:
    parser = ArgumentParser(description="Inference script for a neural network.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    parser.add_argument("model_path", type=Path, help="Path to the model weights")
    args = parser.parse_args()
    inference(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
