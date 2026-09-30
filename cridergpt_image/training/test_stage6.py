import torch
from noise_schedule import LinearNoiseSchedule

def test_add_noise_shape():
    schedule = LinearNoiseSchedule(100)
    clean = torch.randn(2, 4, 32, 32)
    noise = torch.randn_like(clean)
    t = torch.tensor([0, 99])
    noisy = schedule.add_noise(clean, noise, t)
    assert noisy.shape == clean.shape

def test_early_timestep_is_closer_to_clean():
    torch.manual_seed(0)
    schedule = LinearNoiseSchedule(100)
    clean = torch.randn(1, 4, 32, 32)
    noise = torch.randn_like(clean)
    early = schedule.add_noise(clean, noise, torch.tensor([0]))
    late = schedule.add_noise(clean, noise, torch.tensor([99]))
    early_error = torch.mean((early-clean)**2)
    late_error = torch.mean((late-clean)**2)
    assert early_error < late_error
