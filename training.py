import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader 
from torch.utils.tensorboard import SummaryWriter 
from dcgan_model import Discriminator, Generator, initialise_weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
Z_DIM=100
NUM_EPOCHS = 50
FEATURES_DISC = 64
FEATURES_GEN = 64


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

#dataset = datasets.MNIST(root= "dataset/", train=True,transform=transforms, download=True)
dataset= datasets.ImageFolder(root= 'celeb_dataset', transform=transforms)
loader =DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen= Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc= Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialise_weights(gen)
initialise_weights(disc)

optgen= optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
optdisc= optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
criterion= nn.BCELoss()

fixed_noise= torch.randn(32,Z_DIM, 1,1).to(device)

writer_real= SummaryWriter(f"logs_celeb1/real")
writer_fake= SummaryWriter(f"logs_celeb1/fake")


step=0
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real,_) in enumerate(loader):
        real= real.to(device)
        noise= torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
        fake= gen(noise)

        # Train Discriminator
        disc_real= disc(real).reshape(-1)
        loss_disc_real= criterion(disc_real, torch.ones_like(disc_real))
        disc_fake= disc(fake).reshape(-1)
        loss_disc_fake= criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc= (loss_disc_fake + loss_disc_real)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        optdisc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        optgen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1