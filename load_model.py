import torch
from training import LSTMAutoencoder


def load_model(type):

    model_path = f"models/model_{type}.pth"


    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved_model = torch.load(model_path, map_location=device)

    input_dim = saved_model["input_dim"]
    hidden_dim = saved_model["hidden_dim"]
    latent_dim = saved_model["latent_dim"]
    num_layers = saved_model["num_layers"]
    mean = saved_model["mean"]
    std = saved_model["std"]
    threshold = saved_model["threshold"]

    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers).to(torch.device(device))
    model.load_state_dict(saved_model["model_state_dict"])
    model.eval()
    return model, device, mean, std, threshold, input_dim

