import torch
import torch.nn as nn

class MlpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = MlpBlock(in_size, int(in_size / 2))
        self.linear2 = MlpBlock(int(in_size / 2), int(in_size / 4))
        self.linear3 = MlpBlock(int(in_size / 4), latent_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = MlpBlock(latent_size, int(out_size / 4))
        self.linear2 = MlpBlock(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Sequential(
            nn.Linear(int(out_size / 2), out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    
class Usad(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_phase(self, x, n):
        Z = self.encoder(x)
        W1 = self.decoder1(Z)
        W2 = self.decoder2(Z)
        W2_p2 = self.decoder2(self.encoder(W1))

        loss_AE1 = (1 / n) * (torch.mean((x - W1) ** 2)) + (1 - (1 / n)) * (torch.mean((x - W2_p2) ** 2))
        loss_AE2 = (1 / n) * (torch.mean((x - W2) ** 2)) - (1 - (1 / n)) * (torch.mean((x - W2_p2) ** 2))

        return (loss_AE1, loss_AE2)
    
    def testing_phase(self, x, alpha, beta):
        W1 = self.decoder1(self.encoder(x))
        W2_p2 = self.decoder2(self.encoder(W1))
        
        term1 = alpha * (torch.mean((x - W1) ** 2))
        term2 = beta * (torch.mean((x - W2_p2) ** 2))

        return term1 + term2