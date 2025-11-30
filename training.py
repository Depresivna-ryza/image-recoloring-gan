from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dataset import RecoloringDataset
from network import GANLoss, UNet, Discriminator, GAN, load_gan_models, save_gan_models
from torch import Tensor, nn
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, random_split
from torchview import draw_graph


# sample function for model architecture visualization
# draw_graph function saves an additional file: Graphviz DOT graph file, it's not necessary to delete it
def draw_network_architecture(net: nn.Module, input_sample: Tensor) -> None:
    # saves visualization of model architecture to the model_architecture.png
    draw_graph(
        net,
        input_sample,
        graph_dir="TB",
        save_graph=True,
        filename="model_architecture",
        expand_nested=True,
    )


# sample function for losses visualization
def plot_learning_curves(
    loss_dict: dict[str, list[float]],
) -> None:
    # plt.figure(figsize=(10, 5))
    # plt.title("Train and Evaluation Losses During Training")
    # plt.plot(train_losses, label="train_loss")
    # plt.plot(validation_losses, label="validation_loss")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig("learning_curves.png")
    
    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(loss_dict["Train L1 Loss"], label="Train L1 Loss")
    plt.plot(loss_dict["Train Adveserial Generator Loss"], label="Train Adversarial Generator Loss")
    plt.plot(loss_dict["Train Adveserial Discriminator Loss"], label="Train Adversarial Discriminator Loss")
    plt.plot(loss_dict["Validation L1 Loss"], label="Validation L1 Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


# sample function for training
def fit(
    generator: UNet,
    discriminator: Discriminator,
    batch_size: int,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    g_optimizer: Optimizer,
    d_optimizer: Optimizer,
    device: torch.device,
) -> dict[str, list[float]]:
    
    loss_dict: dict[str, list[float]] = {
        "Train L1 Loss": [],
        "Train Adveserial Generator Loss": [],
        "Train Adveserial Discriminator Loss": [],
        "Validation L1 Loss": [],
    }
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    gan_loss_fn = GANLoss().to(device)
    l1_loss_fn = nn.L1Loss()
    
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        train_l1_losses = []
        train_g_losses = []
        train_d_losses = []

        for grayscale, real_rgb in train_dataloader:
            grayscale = grayscale.to(device)
            real_rgb = real_rgb.to(device)
            
            ### DISCRIMINATOR ###
            with torch.no_grad():
                fake_rgb = generator(grayscale)
            fake_input = torch.cat([grayscale, fake_rgb], dim=1)
            real_input = torch.cat([grayscale, real_rgb], dim=1)

            discriminator.zero_grad()
            pred_fake = discriminator(fake_input.detach())
            pred_real = discriminator(real_input)
            loss_d_fake = gan_loss_fn(pred_fake, False)
            loss_d_real = gan_loss_fn(pred_real, True)
            loss_d = loss_d_fake + loss_d_real
            
            loss_d.backward()
            d_optimizer.step()

            ### GENERATOR ###
            generator.zero_grad()
            fake_rgb = generator(grayscale)
            fake_input = torch.cat([grayscale, fake_rgb], dim=1)
            pred_fake = discriminator(fake_input)
            loss_g_gan = gan_loss_fn(pred_fake, True)
            loss_g_l1 = l1_loss_fn(fake_rgb, real_rgb)
            # loss_g_l1 = l1_loss_fn(fake_rgb, real_rgb) * 100
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()
            g_optimizer.step()
            
            
            ### LOSSES ###
            
            train_d_losses.append(loss_d.item())
            train_g_losses.append(loss_g_gan.item())
            train_l1_losses.append(loss_g_l1.item())
            
            print(f"Epoch {epoch}, batch {len(train_l1_losses)}, G L1 Loss: {loss_g_l1.item():.6f}, D Fake Loss: {loss_d_fake.item():.6f}, D Real Loss: {loss_d_real.item():.6f}, G ADV Loss: {loss_g_gan.item():.6f}")
        
        loss_dict["Train L1 Loss"].append(sum(train_l1_losses) / len(train_l1_losses))
        loss_dict["Train Adveserial Generator Loss"].append(sum(train_g_losses) / len(train_g_losses))
        loss_dict["Train Adveserial Discriminator Loss"].append(sum(train_d_losses) / len(train_d_losses))


        # Validation loss (L1 only)
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for grayscale, real_rgb in val_dataloader:
                grayscale = grayscale.to(device)
                real_rgb = real_rgb.to(device)
                fake_rgb = generator(grayscale)
                val_loss += l1_loss_fn(fake_rgb, real_rgb).item()

        val_loss /= len(val_dataloader)
        loss_dict["Validation L1 Loss"].append(val_loss)

        print(f"Epoch {epoch}, Validation L1 Loss: {val_loss:.6f}, Train L1 Loss: {loss_dict['Train L1 Loss'][-1]:.6f}")
        print(f"Train Adveserial Generator Loss: {loss_dict['Train Adveserial Generator Loss'][-1]:.6f}, Train Adveserial Discriminator Loss: {loss_dict['Train Adveserial Discriminator Loss'][-1]:.6f}")
        torch.save(generator.state_dict(), f"model.pt")
        
        # save
        path = save_gan_models(generator, discriminator)
        print(f"Model saved as {path}")

    print("Training finished!")
    return loss_dict


# declaration for this function should not be changed
def training(dataset_path: Path = Path("data_col_public/"), bootstrap=False) -> None:
    """Performs training on the given dataset.

    Args:
        dataset_path: Path to the dataset.

    Saves:
        - model.pt (trained model)
        - learning_curves.png (learning curves generated during training)
        - model_architecture.png (a scheme of model's architecture)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Computing with {}!".format(device))

    batch_size = 16
    epochs = 300
    train_val_split = 0.90

    # dataset
    full_dataset = RecoloringDataset(dataset_path)

    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    generator_net = UNet()
    discriminator_net = Discriminator()
    
    
    if bootstrap:
        generator_net, discriminator_net = load_gan_models()
        print("Loaded pretrained model weights.")
    else:
        print("Training from scratch.")

    
    input_sample = full_dataset[0][0].unsqueeze(0).to(device)
    
    gan_network = GAN(generator_net, discriminator_net)
    
    draw_network_architecture(gan_network, input_sample)
    print("Model architecture saved as model_architecture.png")

    # optimizer and learning rate
    generator_optimizer = Adam(generator_net.parameters(), lr=1e-4)
    discriminator_optimizer = Adam(discriminator_net.parameters(), lr=1e-4)

    # training
    loss_dict = fit(
        generator_net,
        discriminator_net,
        batch_size,
        epochs,
        train_dataloader,
        val_dataloader,
        generator_optimizer,
        discriminator_optimizer,
        device,
    )

    # save model
    path = save_gan_models(generator_net, discriminator_net)
    print(f"Model saved as {path}")

    
    plot_learning_curves(loss_dict)


##### code below should not be changed ############################################################################


def main() -> None:
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset")
    args = parser.parse_args()
    training(args.dataset_path)


if __name__ == "__main__":
    main()
