"""정방향(Forward) 성능 예측 파이프라인 실행 스크립트."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch

from physgenrd import (
    CurveEncoder,
    DiffusionConfig,
    GrainConfig,
    LatentDiffusion,
    NeuralSDFField,
    PhysicsSurrogate,
    PositionalEncoding,
    forward_performance,
    make_grid,
    set_seed,
)


def build_target_curve(cfg: GrainConfig) -> torch.Tensor:
    time_grid = torch.arange(0.0, cfg.t_end + cfg.dt, cfg.dt)
    pc_target = torch.ones_like(time_grid) * 6.0e6
    pc_target[time_grid > 2.2] *= torch.exp(-0.8 * (time_grid[time_grid > 2.2] - 2.2))
    return pc_target


def main() -> None:
    cfg = GrainConfig()
    diff_cfg = DiffusionConfig()

    set_seed(cfg.seed)
    out_dir = "./out_physgenrd"
    os.makedirs(out_dir, exist_ok=True)

    _, _, _, coords_flat = make_grid(cfg)
    coords_flat = coords_flat.to(cfg.device)

    encoder = PositionalEncoding().to(cfg.device)
    curve_encoder = CurveEncoder().to(cfg.device)

    target_curve = build_target_curve(cfg).to(cfg.device)
    cond = curve_encoder(target_curve.unsqueeze(0).unsqueeze(0)).squeeze(0)

    sdf_field = NeuralSDFField(
        enc_dim=encoder(coords_flat[:1]).shape[-1],
        latent_dim=diff_cfg.latent_dim,
        cond_dim=cond.shape[-1],
    ).to(cfg.device)
    surrogate = PhysicsSurrogate(cfg.grid_size).to(cfg.device)
    diffusion = LatentDiffusion(diff_cfg, cond_dim=cond.shape[-1]).to(cfg.device)

    z = diffusion.sample(cond)
    out = forward_performance(sdf_field, encoder, surrogate, coords_flat, cfg, z, cond)

    time_grid = out["time"].cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(time_grid, target_curve.cpu().numpy() / 1e6, "k--", label="Target")
    plt.plot(time_grid, out["Pc"].cpu().numpy() / 1e6, "r-", label="Predicted")
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [MPa]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forward_pressure.png"), dpi=200)

    plt.figure(figsize=(8, 4))
    plt.plot(time_grid, out["F"].cpu().numpy() / 1e3)
    plt.xlabel("Time [s]")
    plt.ylabel("Thrust [kN]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forward_thrust.png"), dpi=200)


if __name__ == "__main__":
    main()
