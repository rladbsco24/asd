"""3D-PhysGenRD: 3D physics-guided generative reverse design utilities.

이 모듈은 3D SDF 기반 형상 표현, 조건부 latent diffusion,
물리 surrogate(에이코날/연소면적)와 역설계 최적화를 포함합니다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim


@dataclass
class GrainConfig:
    seed: int = 1234
    gamma: float = 1.22
    gas_constant: float = 355.0
    chamber_temp: float = 3200.0
    rho_p: float = 1700.0
    burn_a: float = 5.0e-5
    burn_n: float = 0.35
    pa: float = 101325.0
    throat_area: float = 3.0e-4
    length: float = 0.20
    case_radius: float = 0.10
    grid_size: int = 48
    dt: float = 0.001
    t_end: float = 3.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DiffusionConfig:
    latent_dim: int = 512
    time_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    guidance_scale: float = 3.5


@dataclass
class TrainingConfig:
    num_iters: int = 1200
    lr: float = 2e-4
    clip_grad: float = 0.5
    target_loading: float = 0.70
    min_loading: float = 0.35
    max_loading: float = 1.00


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_grid(cfg: GrainConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.linspace(-cfg.case_radius, cfg.case_radius, cfg.grid_size, device=cfg.device)
    y = torch.linspace(-cfg.case_radius, cfg.case_radius, cfg.grid_size, device=cfg.device)
    z = torch.linspace(-cfg.length / 2, cfg.length / 2, cfg.grid_size, device=cfg.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1)
    coords_flat = coords.reshape(-1, 3)
    return xx, yy, zz, coords_flat


class GaussianFourierFeatures(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randn(in_dim, out_dim) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.weight
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int = 3, num_freqs: int = 6, use_gaussian: bool = True) -> None:
        super().__init__()
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)
        self.use_gaussian = use_gaussian
        if use_gaussian:
            self.gaussian = GaussianFourierFeatures(in_dim, num_freqs)
        else:
            self.gaussian = None

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        enc = [coords]
        for d in range(coords.shape[-1]):
            c = coords[:, d:d + 1]
            for f in self.freq_bands.to(coords.device):
                enc.append(torch.sin(f * c))
                enc.append(torch.cos(f * c))
        if self.gaussian is not None:
            enc.append(self.gaussian(coords))
        return torch.cat(enc, dim=-1)


class CurveEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        return self.net(curves)


class NeuralSDFField(nn.Module):
    def __init__(
        self,
        enc_dim: int,
        latent_dim: int,
        cond_dim: int,
        hidden: int = 512,
        layers: int = 9,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.in_dim = enc_dim + latent_dim + cond_dim
        blocks = []
        for i in range(layers):
            in_features = self.in_dim if i == 0 else hidden
            blocks.append(nn.Linear(in_features, hidden))
            blocks.append(nn.SiLU())
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, enc: torch.Tensor, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(0).expand(enc.shape[0], -1)
        cond_expanded = cond.unsqueeze(0).expand(enc.shape[0], -1)
        x = torch.cat([enc, z_expanded, cond_expanded], dim=-1)
        sdf = self.net(x).squeeze(-1)
        return torch.clamp(sdf, -1.0, 1.0)


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + 64, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_mlp(t)
        x = torch.cat([z, cond, t_embed], dim=-1)
        return self.net(x)


class LatentDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig, cond_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.time_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.denoiser = LatentDenoiser(cfg.latent_dim, cond_dim)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_alpha * z0 + sqrt_one_minus * noise

    def predict_eps(self, zt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time = t.float().unsqueeze(-1) / float(self.cfg.time_steps)
        return self.denoiser(zt, time, cond)

    def p_sample(self, zt: torch.Tensor, t: int, cond: torch.Tensor) -> torch.Tensor:
        t_tensor = torch.tensor([t], device=zt.device, dtype=torch.long)
        eps = self.predict_eps(zt, t_tensor, cond)
        alpha = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=zt.device)
        pred_x0 = (zt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(alpha)
        noise = torch.randn_like(zt) if t > 0 else torch.zeros_like(zt)
        sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha))
        return torch.sqrt(alpha_prev) * pred_x0 + sigma * noise

    def sample(self, cond: torch.Tensor, guidance_scale: float | None = None) -> torch.Tensor:
        guidance = guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        z = torch.randn(1, self.cfg.latent_dim, device=cond.device)
        for t in reversed(range(self.cfg.time_steps)):
            cond_drop = torch.zeros_like(cond)
            eps_cond = self.predict_eps(z, torch.tensor([t], device=cond.device), cond)
            eps_uncond = self.predict_eps(z, torch.tensor([t], device=cond.device), cond_drop)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=cond.device)
            pred_x0 = (z - torch.sqrt(1 - alpha) * eps) / torch.sqrt(alpha)
            noise = torch.randn_like(z) if t > 0 else torch.zeros_like(z)
            sigma = torch.sqrt((1 - alpha_prev) / (1 - alpha))
            z = torch.sqrt(alpha_prev) * pred_x0 + sigma * noise
        return z.squeeze(0)


class PhysicsSurrogate(nn.Module):
    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )
        self.grid_size = grid_size

    def forward(self, sdf_grid: torch.Tensor) -> torch.Tensor:
        return self.net(sdf_grid)

    def eikonal_residual(self, w: torch.Tensor, spacing: float) -> torch.Tensor:
        grad = torch.gradient(w, spacing=(spacing, spacing, spacing))
        grad_norm = torch.sqrt(sum(g**2 for g in grad) + 1e-8)
        return (grad_norm - 1.0).pow(2).mean()


def surface_area_from_w(w: torch.Tensor, spacing: float, levels: torch.Tensor) -> torch.Tensor:
    areas = []
    for level in levels:
        mask = (w - level).abs() < spacing
        grad = torch.gradient(w, spacing=(spacing, spacing, spacing))
        grad_norm = torch.sqrt(sum(g**2 for g in grad) + 1e-8)
        areas.append((mask * grad_norm).sum() * spacing**2)
    return torch.stack(areas)


def mdot_choked(cfg: GrainConfig, pc: torch.Tensor) -> torch.Tensor:
    coeff = math.sqrt(cfg.gamma / (cfg.gas_constant * cfg.chamber_temp))
    coeff *= (2.0 / (cfg.gamma + 1.0)) ** ((cfg.gamma + 1.0) / (2.0 * (cfg.gamma - 1.0)))
    return cfg.throat_area * torch.clamp(pc, min=0.0, max=1e8) * coeff


def exhaust_velocity(cfg: GrainConfig, pc: torch.Tensor) -> torch.Tensor:
    pc_eff = torch.clamp(pc, min=cfg.pa * 1.1, max=1e8)
    return torch.sqrt(
        2.0
        * cfg.gamma
        * cfg.gas_constant
        * cfg.chamber_temp
        / (cfg.gamma - 1.0)
        * (1.0 - (cfg.pa / pc_eff) ** ((cfg.gamma - 1.0) / cfg.gamma))
    )


def propellant_volume(phi: torch.Tensor, spacing: float) -> torch.Tensor:
    return torch.clamp(phi, 0.0, 1.0).sum() * spacing**3


def loading_fraction(phi: torch.Tensor, cfg: GrainConfig, spacing: float) -> torch.Tensor:
    total = math.pi * cfg.case_radius**2 * cfg.length
    return propellant_volume(phi, spacing) / total


def sdf_to_occupancy(sdf: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(-10.0 * sdf)


def forward_performance(
    sdf_field: NeuralSDFField,
    encoder: PositionalEncoding,
    surrogate: PhysicsSurrogate,
    coords_flat: torch.Tensor,
    cfg: GrainConfig,
    z: torch.Tensor,
    cond: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    enc = encoder(coords_flat)
    sdf = sdf_field(enc, z, cond).reshape(cfg.grid_size, cfg.grid_size, cfg.grid_size)
    sdf_grid = sdf.unsqueeze(0).unsqueeze(0)
    w = surrogate(sdf_grid).squeeze(0).squeeze(0)
    spacing = max(cfg.length, 2 * cfg.case_radius) / cfg.grid_size
    levels = torch.linspace(0.0, w.max().detach(), int(cfg.t_end / cfg.dt), device=w.device)
    areas = surface_area_from_w(w, spacing, levels)

    time_grid = torch.arange(0.0, cfg.t_end + cfg.dt, cfg.dt, device=w.device)
    pc = torch.tensor(2.0e6, device=w.device)
    pc_hist = []
    f_hist = []

    for i in range(len(time_grid)):
        ab = areas[min(i, len(areas) - 1)]
        r_dot = cfg.burn_a * torch.clamp(pc, min=1.0, max=1e8) ** cfg.burn_n
        mdot_gen = cfg.rho_p * ab * r_dot
        mdot_noz = mdot_choked(cfg, pc)
        phi = sdf_to_occupancy(sdf)
        v_g = torch.clamp(
            (1.0 - phi).sum() * spacing**3,
            min=1e-6,
            max=math.pi * cfg.case_radius**2 * cfg.length * 1.1,
        )
        dpc_dt = (mdot_gen - mdot_noz) * cfg.gas_constant * cfg.chamber_temp / v_g
        pc = torch.clamp(pc + dpc_dt * cfg.dt, min=0.0, max=1e8)
        f_hist.append((mdot_noz * exhaust_velocity(cfg, pc)).item())
        pc_hist.append(pc.item())

    return {
        "time": time_grid,
        "Pc": torch.tensor(pc_hist, device=w.device),
        "F": torch.tensor(f_hist, device=w.device),
        "sdf": sdf,
        "W": w,
        "loading": loading_fraction(sdf_to_occupancy(sdf), cfg, spacing),
    }


def reverse_design(
    target_curve: torch.Tensor,
    cfg: GrainConfig,
    diff_cfg: DiffusionConfig,
    train_cfg: TrainingConfig,
) -> Dict[str, torch.Tensor]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    _, _, _, coords_flat = make_grid(cfg)
    coords_flat = coords_flat.to(device)

    encoder = PositionalEncoding().to(device)
    curve_encoder = CurveEncoder().to(device)
    cond = curve_encoder(target_curve.unsqueeze(0).unsqueeze(0))

    sdf_field = NeuralSDFField(
        enc_dim=encoder(coords_flat[:1]).shape[-1],
        latent_dim=diff_cfg.latent_dim,
        cond_dim=cond.shape[-1],
    ).to(device)
    surrogate = PhysicsSurrogate(cfg.grid_size).to(device)
    diffusion = LatentDiffusion(diff_cfg, cond_dim=cond.shape[-1]).to(device)

    z = diffusion.sample(cond.squeeze(0))
    z = nn.Parameter(z)

    optimizer = optim.AdamW(list(sdf_field.parameters()) + [z], lr=train_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_iters)

    for it in range(train_cfg.num_iters):
        optimizer.zero_grad()
        out = forward_performance(sdf_field, encoder, surrogate, coords_flat, cfg, z, cond.squeeze(0))
        pc = out["Pc"]
        fit_loss = nnF.mse_loss(pc, target_curve)

        spacing = max(cfg.length, 2 * cfg.case_radius) / cfg.grid_size
        eikonal_loss = surrogate.eikonal_residual(out["W"], spacing)
        load_frac = out["loading"]
        load_loss = (train_cfg.target_loading - load_frac) ** 2
        load_loss += 10.0 * nnF.relu(train_cfg.min_loading - load_frac) ** 2
        load_loss += 50.0 * nnF.relu(load_frac - train_cfg.max_loading) ** 2

        loss = 20.0 * fit_loss + 2.0 * eikonal_loss + 10.0 * load_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(sdf_field.parameters()) + [z], train_cfg.clip_grad)
        optimizer.step()
        scheduler.step()

        if it % 100 == 0:
            rel_err = torch.mean(torch.abs(pc - target_curve) / (target_curve + 1e-6))
            print(f"Iter {it:4d} | Loss {loss.item():.3e} | RelErr {rel_err:.3f} | Load {load_frac:.3f}")

    final_out = forward_performance(sdf_field, encoder, surrogate, coords_flat, cfg, z, cond.squeeze(0))
    return {
        "Pc": final_out["Pc"],
        "F": final_out["F"],
        "sdf": final_out["sdf"],
        "W": final_out["W"],
        "loading": final_out["loading"],
        "latent": z.detach(),
    }
