import json,os, time
from os import error
from sklearn.metrics import roc_auc_score, f1_score
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


def train_model(sensor_type, windows_normal, windows_anomaly, model_dir="models", epochs=30, batch_size=32,
                lr=1e-3, hidden_dim=64, latent_dim=16,
                num_layers=1, val_split=0.1, patience=5, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model training on {sensor_type}")

    arrs = []
    for w in windows_normal:
        w_np = np.array(w, np.float32)
        if w_np.ndim == 1:
            w_np = w_np.reshape(-1, 1)
        arrs.append(w_np)
    X = np.stack(arrs, axis=0)
    X = X[:,:, 0:2]
    N, T, F = X.shape
    print(f"Training data shape: {X.shape} (N, T, F)")

    arrs_anomaly = []
    for w in windows_anomaly:
        w_np = np.array(w, np.float32)
        if w_np.ndim == 1:
            w_np = w_np.reshape(-1, 1)
        arrs_anomaly.append(w_np)
    X_anomaly = np.stack(arrs_anomaly, axis=0)

    os.makedirs(model_dir, exist_ok=True)

    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = max(1, int(N * val_split))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X[train_idx]
    X_val = X[val_idx]

    X_train_sensor = X_train[:, :, 0:1]
    X_train_time = X_train[:, :, 1:2]

    X_val_sensor = X_val[:, :, 0:1]
    X_val_time = X_val[:, :, 1:2]

    X_anom_sensor = X_anomaly[:, :, 0:1]
    X_anom_time = X_anomaly[:, :, 1:2]
    X_anom_labels = X_anomaly[:, :, 2]

    mean = X_train_sensor.reshape(-1, 1).mean(axis=0)
    std = X_train_sensor.reshape(-1, 1).std(axis=0)
    std[std == 0.0] = 1.0

    X_train_sensor_norm = (X_train_sensor - mean.reshape(1, 1, 1)) / std.reshape(1, 1, 1)
    X_val_sensor_norm = (X_val_sensor - mean.reshape(1, 1, 1)) / std.reshape(1, 1, 1)
    X_anom_sensor_norm = (X_anom_sensor-mean.reshape(1, 1, 1)) / std.reshape(1, 1, 1)

    X_train_norm = np.concatenate([X_train_sensor_norm, X_train_time], axis=2)
    X_val_norm = np.concatenate([X_val_sensor_norm, X_val_time], axis=2)
    X_anomaly_norm = np.concatenate([X_anom_sensor_norm, X_anom_time], axis=2)

    train_ds = TensorDataset(torch.from_numpy(X_train_norm).float())
    val_ds = TensorDataset(torch.from_numpy(X_val_norm).float())
    anomaly_ds = TensorDataset(torch.from_numpy(X_anomaly_norm).float(),
                               torch.from_numpy(X_anom_labels).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    anomaly_loader = DataLoader(anomaly_ds, batch_size=batch_size, shuffle=False, drop_last=False)

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
            out_temp = out[:,:,0]
            target_temp = x[:,:,0]
            loss = criterion(out_temp, target_temp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        print(x[0])
        print(out[0])
        train_loss /= max(1,n_batches)

        model.eval()
        val_loss = 0
        n_val_batches = 0


        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                out = model(x)

                out_temp = out[:,:,0]
                target_temp = x[:,:,0]

                loss = criterion(out_temp, target_temp)
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(1,n_val_batches)
        scheduler.step(val_loss)
        print(f"[train_model] Epoch {epoch}/{epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            model_path = os.path.join(model_dir, f"model_{sensor_type}.pth")

            thresholds, scores, labels = [], [], []
            model.eval()
            with torch.no_grad():
                for x in val_loader:
                    x = x[0].to(device)
                    out = model(x)
                    out_temp = out[:,:,0]
                    target_temp = x[:,:,0]
                    err = ((out_temp - target_temp)**2).cpu().numpy()
                    scores.extend(err)
                    labels.extend([[0]*T for _ in range(len(err))])
                for x,y in anomaly_loader:
                    x = x.to(device)
                    out = model(x)
                    out_temp = out[:,:,0]
                    target_temp = x[:,:,0]
                    err = ((out_temp-target_temp)**2).cpu().numpy()

                    scores.extend(err)
                    labels.extend(y.cpu().numpy().astype(int))
            # labels birden fazla array içeriyorsa flatten edelim
            labels = np.concatenate([np.array(l, dtype=int).ravel() for l in labels])
            scores = np.concatenate([np.array(s, dtype=float).ravel() for s in scores])
            np.set_printoptions(threshold=np.inf)
            #print("Scores: ", scores[6:56])
            #print("Labels: ", labels[6:56])
            auc = roc_auc_score(labels, scores)
            print(f"Validation AUC={auc:.3f}")

            best_th, best_f1 = None, -1
            for th in np.percentile(scores, np.linspace(0, 99, 50)):
                preds = (np.array(scores) > th).astype(int)
                f1 = f1_score(labels, preds, average='macro')
                if f1 > best_f1:
                    best_f1, best_th = f1, th

            print(f"Best threshold={best_th:.4f}, F1={best_f1:.4f}")


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
                "threshold": best_th,
            }, model_path)

            np.savez(os.path.join(model_dir, f"scaler_{sensor_type}.npz"), mean=mean, std=std)
            print(f"Saved best model: {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.1f}s, best_val={best_val:.3f} at epoch {best_epoch}")
    return os.path.join(model_dir, f"model_{sensor_type}.pth")
