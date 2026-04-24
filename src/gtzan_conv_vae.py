import torch
import torch.nn as nn
import torch.optim as optim


class GTZANConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(GTZANConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: input shape = (batch, 1, 128, 128)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),   # -> 16 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 8 x 8
            nn.ReLU()
        )

        self.flatten_dim = 128 * 8 * 8

        self.mu_layer = nn.Linear(self.flatten_dim, latent_dim)
        self.logvar_layer = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> 16 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),    # -> 1 x 128 x 128
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(x.size(0), -1)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(z.size(0), 128, 8, 8)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def gtzan_conv_vae_loss(reconstructed, original, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(
        reconstructed, original, reduction="sum"
    )
    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return reconstruction_loss + kl_divergence


def train_gtzan_conv_vae(model, train_loader, val_loader, device, num_epochs=20, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = model(x)
            loss = gtzan_conv_vae_loss(reconstructed, x, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed, mu, logvar = model(x)
                loss = gtzan_conv_vae_loss(reconstructed, x, mu, logvar)
                val_loss += loss.item()

        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    return model, train_losses, val_losses


def extract_gtzan_latent_features(model, X_array, device):
    model.eval()
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        latent_features = mu.cpu().numpy()

    return latent_features
