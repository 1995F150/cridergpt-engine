import torch
from model import CriderImageLatentGenerator


def test_generator_shape():
    model = CriderImageLatentGenerator().eval()
    z = torch.randn(2, 4, 32, 32)
    t = torch.tensor([10, 500])
    text = torch.randn(2, 64, 256)
    mask = torch.ones(2, 64, dtype=torch.long)
    with torch.no_grad():
        predicted_noise = model(z, t, text, mask)
    assert tuple(predicted_noise.shape) == tuple(z.shape)


def test_text_conditioning_changes_output():
    torch.manual_seed(42)
    model = CriderImageLatentGenerator().eval()
    z = torch.randn(1, 4, 32, 32)
    t = torch.tensor([100])
    a = torch.zeros(1, 64, 256)
    b = torch.ones(1, 64, 256)
    with torch.no_grad():
        out_a = model(z, t, a)
        out_b = model(z, t, b)
    assert not torch.allclose(out_a, out_b)


def test_backward_pass():
    model = CriderImageLatentGenerator()
    z = torch.randn(1, 4, 32, 32)
    target = torch.randn_like(z)
    predicted = model(z, torch.tensor([50]), torch.randn(1, 64, 256))
    torch.nn.functional.mse_loss(predicted, target).backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
