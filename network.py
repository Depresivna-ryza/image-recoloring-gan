from pathlib import Path
import torch.nn.functional as F
from torch import Tensor, nn
import torch
import torchvision

class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, leaky_relu: bool) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    
        self.relu: nn.Module
        if leaky_relu:
            self.relu = nn.LeakyReLU(0.02)
        else: 
            self.relu = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], leaky_relu=True) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks[:-1]:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)

        x = self.enc_blocks[-1](x)
        ftrs.append(x)

        return ftrs


class Decoder(nn.Module):
    def __init__(self, enc_chs, dec_chs):
        super().__init__()
        self.enc_chs = enc_chs
        self.dec_chs = dec_chs
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    dec_chs[i], dec_chs[i+1], 2, 2
                    ) for i in range(len(dec_chs)-1)
            ]
        )
        
        self.dec_blocks = nn.ModuleList(
            [Block(
                dec_chs[i+1] + enc_chs[-(i+2)],  # concatenate upsampled + encoder features
                dec_chs[i+1],
                leaky_relu=True
                ) for i in range(len(dec_chs)-1)
             ]
            ) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.dec_chs)-1):
            x = self.upconvs[i](x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    # def __init__(self, enc_chs=(1, 64, 64, 64, 128, 128, 256, 512), dec_chs=(512, 256, 256, 128, 126, 64, 64)):
    def __init__(self, enc_chs=(1, 32, 32, 64, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 64, 32, 32)):
    # def __init__(self, enc_chs=(1, 16, 16, 32, 32, 64, 128), dec_chs=(128, 64, 32, 32, 16, 16)):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(enc_chs, dec_chs)
        self.head1 = nn.Conv2d(dec_chs[-1], 3, kernel_size=1)
        # self.head2 = nn.Conv2d(3+1, 3, kernel_size=1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        head1 = self.head1(out)
        
        # head2 = self.head2(torch.cat([head1, x], dim=1))
        
        return head1

class Discriminator(nn.Module):
    def __init__(self, input_channels=4): # 3 RGB + 1 grayscale
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, stride=1, padding=1),
            
            # nn.Conv2d(256, 512, 4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv2d(1024, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)
    
class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
    
    def get_labels(self, preds: Tensor, target_is_real: bool):
        if target_is_real:
            return torch.full_like(preds, 0.95)
        else:
            return torch.zeros_like(preds)
    
    def __call__(self, preds: Tensor, target_is_real: bool):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
    

class GAN(nn.Module):
    def __init__(self, generator: UNet, discriminator: Discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        assert x.shape == (1, 1, 512, 1024), f"Input shape mismatch: {x.shape}"
        
        rgb = self.generator(x)
        assert rgb.shape == (1, 3, 512, 1024), f"Output shape mismatch: {rgb.shape}"
        
        input_img = torch.cat([x, rgb], dim=1)
        discriminator_output = self.discriminator(input_img)
        return discriminator_output, rgb


def load_gan_models(path: Path = Path("model.pt")) -> tuple[UNet, Discriminator]:
    gan = GAN(UNet(), Discriminator())
    
    state_dict = torch.load(path)
    gan.load_state_dict(state_dict)
    return (gan.generator, gan.discriminator)

def save_gan_models(generator: UNet, discriminator: Discriminator, path: str = "model.pt") -> str:
    gan = GAN(generator, discriminator)
    torch.save(gan.state_dict(), path)
    return path






