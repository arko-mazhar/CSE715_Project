import torch
import torch.nn as nn
import torch.optim as optim


class BetaVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, latent_dim=16):
        super(BetaVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )

        self.mu_layer = nn.Linear(hidden_dim2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def beta_vae_loss(reconstructed, original, mu, logvar, beta=4.0):
    reconstruction_loss = nn.functional.mse_loss(
        reconstructed, original, reduction="sum"
    )
    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return reconstruction_loss + beta * kl_divergence


def train_beta_vae(model, train_loader, val_loader, device, num_epochs=30, learning_rate=0.001, beta=4.0):
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
            loss = beta_vae_loss(reconstructed, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed, mu, logvar = model(x)
                loss = beta_vae_loss(reconstructed, x, mu, logvar, beta=beta)
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


def extract_beta_vae_latent_features(model, X_array, device):
    model.eval()
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        latent_features = mu.cpu().numpy()

    return latent_features


class BasicAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, latent_dim=16):
        super(BasicAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


def autoencoder_loss(reconstructed, original):
    return nn.functional.mse_loss(reconstructed, original, reduction="sum")


def train_autoencoder(model, train_loader, val_loader, device, num_epochs=30, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(x)
            loss = autoencoder_loss(reconstructed, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed = model(x)
                loss = autoencoder_loss(reconstructed, x)
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


def extract_autoencoder_latent_features(model, X_array, device):
    model.eval()
    X_tensor = torch.tensor(X_array, dtype=torch.float32).to(device)

    with torch.no_grad():
        latent_features = model.encode(X_tensor).cpu().numpy()

    return latent_features
