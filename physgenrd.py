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
    loop_steps: int = 3
    inner_iters: int = 400
    performance_tol: float = 0.05
    guidance_scale: float = 4.0
    pinn_weight: float = 2.5
    loading_weight: float = 8.0
    smooth_weight: float = 0.25
    latent_weight: float = 0.05
    cond_dropout: float = 0.1


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


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device).float() / (half - 1)
        )
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = False) -> None:
        super().__init__()
        self.residual = residual and in_dim == out_dim
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        if self.residual:
            out = out + x
        return self.act(out)


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
        self.layers = nn.ModuleList()
        for i in range(layers):
            in_features = self.in_dim if i == 0 else hidden
            self.layers.append(MLPBlock(in_features, hidden, residual=i > 0))
        self.skip_index = layers // 2
        self.out = nn.Linear(hidden + self.in_dim, 1)

    def forward(self, enc: torch.Tensor, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(0).expand(enc.shape[0], -1)
        cond_expanded = cond.unsqueeze(0).expand(enc.shape[0], -1)
        x = torch.cat([enc, z_expanded, cond_expanded], dim=-1)
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx == self.skip_index:
                h = torch.cat([h, x], dim=-1)
        sdf = self.out(h).squeeze(-1)
        return torch.clamp(sdf, -1.0, 1.0)


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(128)
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.cond_proj = nn.Linear(cond_dim, 256)
        self.blocks = nn.Sequential(
            MLPBlock(latent_dim + 256 + 256, 1024),
            MLPBlock(1024, 1024, residual=True),
            MLPBlock(1024, 1024, residual=True),
        )
        self.out = nn.Linear(1024, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if t.dim() > 1 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_embed = self.time_mlp(self.time_embed(t))
        cond_embed = self.cond_proj(cond)
        x = torch.cat([z, cond_embed, t_embed], dim=-1)
        return self.out(self.blocks(x))


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
        time = t.float() / float(self.cfg.time_steps)
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
        self.net = FNO3D(in_channels=1, hidden=32, modes=8, layers=3)
        self.grid_size = grid_size

    def forward(self, sdf_grid: torch.Tensor) -> torch.Tensor:
        return self.net(sdf_grid)

    def eikonal_residual(self, w: torch.Tensor, spacing: float) -> torch.Tensor:
        grad = torch.gradient(w, spacing=(spacing, spacing, spacing))
        grad_norm = torch.sqrt(sum(g**2 for g in grad) + 1e-8)
        return (grad_norm - 1.0).pow(2).mean()


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, nx, ny, nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny,
            nz // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        mx = min(self.modes, nx)
        my = min(self.modes, ny)
        mz = min(self.modes, nz // 2 + 1)
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bixyz,ioxyz->boxyz", x_ft[:, :, :mx, :my, :mz], self.weight[:, :, :mx, :my, :mz]
        )
        x = torch.fft.irfftn(out_ft, s=(nx, ny, nz))
        return x


class FNOBlock(nn.Module):
    def __init__(self, hidden: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(hidden, hidden, modes)
        self.pointwise = nn.Conv3d(hidden, hidden, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class FNO3D(nn.Module):
    def __init__(self, in_channels: int, hidden: int, modes: int, layers: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(hidden, modes) for _ in range(layers)])
        self.output_proj = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


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


def occupancy_smoothness(phi: torch.Tensor) -> torch.Tensor:
    return (phi * (1.0 - phi)).mean()


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

        loss = 20.0 * fit_loss + train_cfg.pinn_weight * eikonal_loss + train_cfg.loading_weight * load_loss
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


def _build_optimizer_modules(
    cfg: GrainConfig,
    diff_cfg: DiffusionConfig,
    target_curve: torch.Tensor,
    coords_flat: torch.Tensor,
) -> Tuple[PositionalEncoding, CurveEncoder, NeuralSDFField, PhysicsSurrogate, LatentDiffusion, torch.Tensor]:
    encoder = PositionalEncoding().to(cfg.device)
    curve_encoder = CurveEncoder().to(cfg.device)
    cond = curve_encoder(target_curve.unsqueeze(0).unsqueeze(0))
    sdf_field = NeuralSDFField(
        enc_dim=encoder(coords_flat[:1]).shape[-1],
        latent_dim=diff_cfg.latent_dim,
        cond_dim=cond.shape[-1],
    ).to(cfg.device)
    surrogate = PhysicsSurrogate(cfg.grid_size).to(cfg.device)
    diffusion = LatentDiffusion(diff_cfg, cond_dim=cond.shape[-1]).to(cfg.device)
    return encoder, curve_encoder, sdf_field, surrogate, diffusion, cond.squeeze(0)


def reverse_design_loop(
    target_curve: torch.Tensor,
    cfg: GrainConfig,
    diff_cfg: DiffusionConfig,
    train_cfg: TrainingConfig,
) -> Dict[str, torch.Tensor]:
    """STEP 1~3 루프 기반 역설계.

    STEP 1: 성능 요구사항 분석 (조건 임베딩 및 목표 정규화)
    STEP 2: 최적화 (latent + SDF 필드 + PINN 잔차)
    STEP 3: SRM 성능 체크 (forward 성능 검증 및 허용오차 평가)
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    _, _, _, coords_flat = make_grid(cfg)
    coords_flat = coords_flat.to(device)

    encoder, _, sdf_field, surrogate, diffusion, cond = _build_optimizer_modules(
        cfg, diff_cfg, target_curve, coords_flat
    )
    if train_cfg.cond_dropout > 0:
        drop_mask = (torch.rand_like(cond) > train_cfg.cond_dropout).float()
        cond = cond * drop_mask

    z = nn.Parameter(diffusion.sample(cond, guidance_scale=train_cfg.guidance_scale))
    optimizer = optim.AdamW(list(sdf_field.parameters()) + [z], lr=train_cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.inner_iters)

    best = None
    for step in range(train_cfg.loop_steps):
        for it in range(train_cfg.inner_iters):
            optimizer.zero_grad()
            out = forward_performance(sdf_field, encoder, surrogate, coords_flat, cfg, z, cond)
            pc = out["Pc"]
            fit_loss = nnF.mse_loss(pc, target_curve)
            spacing = max(cfg.length, 2 * cfg.case_radius) / cfg.grid_size
            eikonal_loss = surrogate.eikonal_residual(out["W"], spacing)
            load_frac = out["loading"]
            load_loss = (train_cfg.target_loading - load_frac) ** 2
            load_loss += 10.0 * nnF.relu(train_cfg.min_loading - load_frac) ** 2
            load_loss += 50.0 * nnF.relu(load_frac - train_cfg.max_loading) ** 2
            smooth_loss = occupancy_smoothness(sdf_to_occupancy(out["sdf"]))
            latent_loss = (z.pow(2).mean())

            loss = (
                20.0 * fit_loss
                + train_cfg.pinn_weight * eikonal_loss
                + train_cfg.loading_weight * load_loss
                + train_cfg.smooth_weight * smooth_loss
                + train_cfg.latent_weight * latent_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(sdf_field.parameters()) + [z], train_cfg.clip_grad)
            optimizer.step()
            scheduler.step()

            if it % 200 == 0:
                rel_err = torch.mean(torch.abs(pc - target_curve) / (target_curve + 1e-6))
                print(
                    f"Loop {step + 1}/{train_cfg.loop_steps} | Iter {it:4d} | "
                    f"Loss {loss.item():.3e} | RelErr {rel_err:.3f} | Load {load_frac:.3f}"
                )

        with torch.no_grad():
            out = forward_performance(sdf_field, encoder, surrogate, coords_flat, cfg, z, cond)
            rel_err = torch.mean(torch.abs(out["Pc"] - target_curve) / (target_curve + 1e-6))
            candidate = {
                "Pc": out["Pc"],
                "F": out["F"],
                "sdf": out["sdf"],
                "W": out["W"],
                "loading": out["loading"],
                "latent": z.detach().clone(),
                "rel_err": rel_err,
                "loop": step + 1,
            }
            if best is None or rel_err < best["rel_err"]:
                best = candidate
            if rel_err <= train_cfg.performance_tol:
                break

        z = nn.Parameter(diffusion.sample(cond, guidance_scale=train_cfg.guidance_scale))
        optimizer = optim.AdamW(list(sdf_field.parameters()) + [z], lr=train_cfg.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.inner_iters)

    return best
