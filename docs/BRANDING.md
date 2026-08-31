# Turbo-WinFare Dense — Brand & Asset Guidelines

This document outlines the official visual identity, color palette, and asset guidelines for **Turbo-WinFare Dense**.

---

## 🦅 Brand Concept & Identity

**Turbo-WinFare Dense** is an APU-optimized Vulkan 1.3 streaming inference engine for **Gemma 4 31B Dense**, engineered for AMD handhelds (Lenovo Legion Go S / Ryzen Z1 Extreme / Radeon 780M).

The visual identity embodies three core themes:
1. **Turbo**: Aerodynamic speed, NVMe DMA layer streaming, and rapid speculative draft generation (Gemma 4 E2B).
2. **WinFare**: Precision engineering, tactical APU optimization, and cyberpunk glassmorphic HUD aesthetics.
3. **Dense**: 60-layer deep neural transformer model, dense tensor math, and unified memory architecture.

The official symbol is a **Cybernetic Peregrine Falcon** in flight, integrated with a glowing **Turbocharger Compressor Turbine** at its core and energized by high-voltage **Electric Lightning Discharges**.

---

## 🎨 Color Palette

| Token | Hex Code | Swatch / Usage |
|---|---|---|
| **Turbo Cyan** | `#00f5ff` | Primary neon accent, core turbine glow, telemetry status |
| **Sky Neon** | `#38bdf8` | Secondary highlight, wing feathers, link accents |
| **Sapphire Cobalt** | `#3b82f6` | Vulkan compute accents, primary buttons, borders |
| **Ultraviolet Magenta** | `#a855f7` | Speculative decoding badges, attention layers |
| **Neon Purple** | `#c084fc` | Gradient transitions, secondary highlights |
| **Obsidian Dark** | `#080a0f` | Background canvas, HUD container fill |
| **Glass Border** | `rgba(255, 255, 255, 0.12)` | Glassmorphism cards, HUD dividers |

---

## 📁 Asset Inventory

All official assets are stored within the repository:

| Filepath | Format | Dimensions | Usage |
|---|---|---|---|
| [`gui/logo.svg`](../gui/logo.svg) | Scalable Vector SVG | Resolution Independent | Web GUI header, Favicon, Assistant Avatars, Scalable UI |
| [`docs/assets/logo.png`](assets/logo.png) | High-Res PNG | 1024 × 1024 | App icon, profile avatars, release package icon |
| [`docs/assets/logo_with_title.png`](assets/logo_with_title.png) | High-Res PNG | 1024 × 1024 | Titled square logo badge for social shares & documentation |
| [`docs/assets/banner.png`](assets/banner.png) | High-Res PNG | 1920 × 1080 (16:9) | GitHub README hero banner, documentation headers |

---

## 📝 GitHub README Embedding Instructions

### Method 1: Centered Hero Banner (Recommended)

Add this snippet to the top of your `README.md`:

```markdown
<div align="center">
  <img src="docs/assets/banner.png" alt="Turbo-WinFare Dense" width="800" />

  # Turbo-WinFare Dense

  **APU-Optimized Native Vulkan Streaming Inference Engine for Gemma 4 31B Dense**

  [![Vulkan 1.3](https://img.shields.io/badge/Vulkan-1.3-red?style=flat-square&logo=vulkan&logoColor=white)](https://www.vulkan.org/)
  [![Gemma 4 31B](https://img.shields.io/badge/Target_Model-Gemma_4_31B_Dense-blue?style=flat-square)](https://huggingface.co/google/gemma-4-31b-it)
  [![Gemma 4 E2B](https://img.shields.io/badge/Draft_Model-Gemma_4_E2B-green?style=flat-square)]()
  [![AMD Radeon 780M](https://img.shields.io/badge/Hardware-AMD_Radeon_780M-ED1C24?style=flat-square&logo=amd&logoColor=white)](https://www.amd.com/)
  [![INT4 Quant](https://img.shields.io/badge/Quantization-MLX_INT4_G64-blueviolet?style=flat-square)]()

</div>
```

### Method 2: Centered Square Icon Mark

If you prefer a compact logo mark at the top:

```markdown
<div align="center">
  <img src="docs/assets/logo.png" alt="Turbo-WinFare Dense Logo" width="180" />

  # Turbo-WinFare Dense
  
  **APU-Optimized Native Vulkan Streaming Inference Engine for Gemma 4 31B Dense**
</div>
```

### Method 3: Using Direct GitHub Raw URL

If referencing assets from external documentation or wikis:

```markdown
<p align="center">
  <img src="https://raw.githubusercontent.com/justin-creator222/Turbo-WinFare-Dense/main/docs/assets/banner.png" alt="Turbo-WinFare Dense Banner" width="800" />
</p>
```
