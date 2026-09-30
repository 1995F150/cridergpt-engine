import torch

from model import CriderImageAutoencoder


def test_autoencoder_shapes():
    model = CriderImageAutoencoder()
    x = torch.randn(2, 3, 256, 256)
    reconstruction, latent = model(x)
    assert tuple(latent.shape) == (2, 4, 32, 32)
    assert tuple(reconstruction.shape) == (2, 3, 256, 256)


def test_output_range():
    model = CriderImageAutoencoder().eval()
    with torch.no_grad():
        reconstruction, _ = model(torch.randn(1, 3, 256, 256))
    assert reconstruction.min().item() >= -1.0001
    assert reconstruction.max().item() <= 1.0001


def test_gradients_flow():
    model = CriderImageAutoencoder()
    x = torch.randn(1, 3, 256, 256)
    reconstruction, _ = model(x)
    loss = torch.nn.functional.l1_loss(reconstruction, torch.tanh(x))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
