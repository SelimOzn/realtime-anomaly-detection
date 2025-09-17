import json,os, time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_layer, num_layers=1):
        super().__init__()

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim, latent_layer)

        self.decoder_fc = nn.Linear(latent_layer, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h_n[-1])

        hidden_dec = torch.relu(self.decoder_fc(latent))
        hidden_dec = hidden_dec.unsqueeze(0)

        seq_len = x.size(1)
        hidden_repeated = hidden_dec.repeat(seq_len, 1, 1).transpose(0, 1)

        output, _ = self.decoder_lstm(hidden_repeated)

        return output


def train_model(type, windows, model_dir="models", epochs=30, batch_size=32, lr=1e-3, hidden_dim=64, latent_dim=16,
                num_layers=1, val_split=0.1, patience=5, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model training on {type}")

    arrs = []
    for w in windows:
        w_np = np.array(w, np.float32)
        if w_np.ndim == 1:
            w_np = w_np.reshape(-1, 1)
        arrs.append(w_np)
    X = np.stack(arrs, axis=0)
    N, T, F = X.shape
    print(f"Training data shape: {X.shape} (N, T, F)")

    mean = X.reshape(-1, F).mean(axis=0)
    std = X.reshape(-1, F).std(axis=0)
    std[std == 0.0] = 1.0
    X_norm = (X - mean.reshape(1, 1, F)) / std.reshape(1, 1, F)

    os.makedirs(model_dir, exist_ok=True)

    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = max(1, int(N * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X_norm[train_idx]
    X_val = X_norm[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = F
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=patience, verbose=True)

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0
        n_batches = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(1,n_batches)

        model.eval()
        val_loss = 0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out = model(x)
                loss = criterion(out, x)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(1,n_val_batches)
        scheduler.step(val_loss)
        print(f"[train_model] Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            model_path = os.path.join(model_dir, f"model_{type}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mean': mean,
                'std': std,
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
            }, model_path)
            np.savez(os.path.join(model_dir, f"scaler_{type}.npz"), mean=mean, std=std)
            print(f"Saved best model: {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s, best_val={best_val:.3f} at epoch {best_epoch}")
    return os.path.join(model_dir, f"model_{type}.pth")
