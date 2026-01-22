import numpy as np
import torch

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_num(n):
    """Format large numbers with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def print_model_summary(model, name="Model"):
    """Print model architecture and parameter count."""
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f" {name} Summary")
    print(f"{'='*60}")
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Total parameters:     {total:>10,} ({format_num(total)})")
    print(f"  Trainable parameters: {trainable:>10,} ({format_num(trainable)})")
    print(f"  Non-trainable:        {total - trainable:>10,}")
    
    # Parameter breakdown by layer
    print(f"\n  Layer-wise breakdown:")
    print(f"  {'-'*50}")
    for name, param in model.named_parameters():
        print(f"    {name:<35} {str(list(param.shape)):<15} {param.numel():>8,}")
    print(f"  {'-'*50}")
    print(f"{'='*60}\n")


def print_config(args):
    """Print all configuration parameters."""
    print(f"\n{'='*60}")
    print(f" Training Configuration")
    print(f"{'='*60}")
    
    sections = {
        "Model": ["qmlp", "quantum", "qubits", "num_layers", "num_actions", "input_size"],
        "Environment": ["num_static", "num_dynamic", "env_seed"],
        "Encoder": ["max_freq", "num_timesteps", "dt", "bg_freq", "noise_std", "enc_seed"],
        "Training": ["episodes", "gamma", "lr", "batch_size", "buffer_capacity", "learn_start", "target_update_freq"],
        "Exploration": ["epsilon", "T0", "lambda_T", "T_min"],
        "System": ["device", "seed", "model_dir", "plot_root", "plots"],
    }
    
    for section, keys in sections.items():
        print(f"\n  [{section}]")
        for key in keys:
            if hasattr(args, key):
                value = getattr(args, key)
                print(f"    {key:<20} = {value}")
    
    print(f"\n{'='*60}\n")


def print_encoder_info(enc):
    """Print encoder configuration and expected spike statistics."""
    p_active = 1 - np.exp(-enc.max_freq * enc.dt)
    p_bg = 1 - np.exp(-enc.bg_freq * enc.dt) if enc.bg_freq > 0 else 0
    expected_spikes_active = p_active * enc.num_timesteps
    expected_spikes_bg = p_bg * enc.num_timesteps
    
    print(f"\n{'='*60}")
    print(f" Encoder Configuration")
    print(f"{'='*60}")
    print(f"  max_freq:       {enc.max_freq} Hz")
    print(f"  num_timesteps:  {enc.num_timesteps}")
    print(f"  dt:             {enc.dt} s")
    print(f"  bg_freq:        {enc.bg_freq} Hz")
    print(f"  noise_std:      {enc.noise_std}")
    print(f"  output_dim:     {enc.total} neurons")
    print(f"\n  [Spike Statistics]")
    print(f"    P(spike | active):    {p_active:.2%}")
    print(f"    P(spike | background):{p_bg:.2%}")
    print(f"    Expected spikes/active neuron:  {expected_spikes_active:.1f}")
    print(f"    Expected spikes/bg neuron:      {expected_spikes_bg:.1f}")
    print(f"    Total neurons:        {enc.total}")
    print(f"    Active neurons:       4 (one-hot per field)")
    print(f"    Expected total spikes:{4 * expected_spikes_active + (enc.total - 4) * expected_spikes_bg:.1f}")
    print(f"    Sparsity:             {1 - (4 * p_active + (enc.total - 4) * p_bg) / enc.total:.1%}")
    print(f"{'='*60}\n")


def print_training_header():
    """Print training progress header."""
    print(f"\n{'='*100}")
    print(f" Training Progress")
    print(f"{'='*100}")
    print(f"{'Ep':>6} | {'Reward':>10} | {'Loss':>10} | {'Dist':>8} | {'T':>8} | {'Steps':>6} | {'Buffer':>8} | {'Success':>8}")
    print(f"{'-'*100}")