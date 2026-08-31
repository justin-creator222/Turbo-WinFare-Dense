/**
 * Turbo-WinFare Dense - Frontend Application Logic
 * Gemma 4 31B Streaming Inference Engine
 */

let isGenerating = false;
let currentBubble = null;
let currentStatsEl = null;
let currentTier = 1;
let conversation = []; // [{role: 'system'|'user'|'assistant', content: '...'}]
let telemetryTimer = null;
let modelInfo = null;
let tierData = null;
let activeAbortController = null;
let selectedLayerIndex = 0;

// Gemma 4 31B Dense Architecture Constants
const TOTAL_LAYERS = 60;
const GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59];
const LAYER_SIZE_MB = 269.45;
const LM_HEAD_SIZE_MB = 755.99;

// Default System Prompts
const SYSTEM_PRESETS = {
    default: "You are a helpful, knowledgeable assistant powered by Gemma 4 31B Dense. Answer the question that was asked, directly and without preamble. Match length to the question: one sentence when that is enough, more when the topic needs it. If you do not know something or are uncertain, say so plainly instead of guessing.",
    code: "You are an expert systems programmer and performance engineer. You write concise, production-grade, highly-optimized code (C++, Vulkan SPIR-V, Rust, Python). Provide clean implementations with clear algorithmic explanations.",
    explain: "You are a senior computer architect and deep learning systems engineer. Explain complex technical architectures, memory subsystems, and tensor pipelines clearly using precise technical terminology and structured bullet points.",
    vulkan: "You are a Vulkan 1.3 and GPU compute shader specialist. Provide exact HLSL/GLSL compute shaders, memory barriers, descriptor set layouts, and subgroup optimizations targeting AMD RDNA3 APU hardware.",
    creative: "You are an imaginative, creative writer with a vivid, evocative style. Craft engaging stories, dialogues, and descriptions with rich sensory details.",
    none: ""
};

// Tier definitions fallback
const DEFAULT_TIERS = [
    { tier_id: 1, name: "Tier 1: 6.0 GB Baseline", memory_ceiling_gb: 6.0, pinned_layers: 6, streamed_layers: 54, heap0_vram_gb: 2.75, heap1_dma_gb: 1.03, projected_tps_alpha_078: 8.8, status: "Verified Active" },
    { tier_id: 2, name: "Tier 2: 10.0 GB Balanced", memory_ceiling_gb: 10.0, pinned_layers: 21, streamed_layers: 39, heap0_vram_gb: 6.79, heap1_dma_gb: 1.03, projected_tps_alpha_078: 11.2, status: "Verified Active" },
    { tier_id: 3, name: "Tier 3: 16.0 GB High-Perf", memory_ceiling_gb: 16.0, pinned_layers: 48, streamed_layers: 12, heap0_vram_gb: 14.06, heap1_dma_gb: 1.03, projected_tps_alpha_078: 15.6, status: "Verified Active" },
    { tier_id: 4, name: "Tier 4: 22.0 GB Resident", memory_ceiling_gb: 22.0, pinned_layers: 60, streamed_layers: 0, heap0_vram_gb: 17.29, heap1_dma_gb: 0.00, projected_tps_alpha_078: 24.5, status: "Requires 32GB+ RAM" }
];

document.addEventListener('DOMContentLoaded', () => {
    init60LayerGrid();
    fetchModelInfo();
    fetchTierData();
    hydrateConfig();
    refreshModelList();
    fetchTelemetry();
    telemetryTimer = setInterval(fetchTelemetry, 1000);
});

// ============================================================================
// Model & Tier Data Hydration
// ============================================================================

function fetchModelInfo() {
    fetch('/api/model_info')
        .then(r => r.json())
        .then(data => {
            if (data && data.loaded) {
                modelInfo = data;
                renderModelMetadata(data);
            }
        })
        .catch(err => console.warn('model_info fetch failed:', err));
}

function fetchTierData() {
    fetch('/api/tiers')
        .then(r => r.json())
        .then(data => {
            tierData = (data && data.tiers && data.tiers.length > 0) ? data.tiers : DEFAULT_TIERS;
            renderTierMatrix();
        })
        .catch(err => {
            console.warn('tiers fetch fallback:', err);
            tierData = DEFAULT_TIERS;
            renderTierMatrix();
        });
}

function renderModelMetadata(info) {
    const metaId = document.getElementById('meta-id');
    const metaArch = document.getElementById('meta-arch');
    const metaFfn = document.getElementById('meta-ffn');
    const metaHeads = document.getElementById('meta-heads');
    const metaGlobal = document.getElementById('meta-global-heads');
    const metaQuant = document.getElementById('meta-quant');
    const metaVocab = document.getElementById('meta-vocab');
    const metaSoftcap = document.getElementById('meta-softcap');
    const hudGpu = document.getElementById('hud-gpu');

    if (metaId) metaId.innerText = info.name || 'gemma-4-31b-dense';
    if (metaArch) metaArch.innerText = `${info.num_layers || 60} Dense Layers • ${(info.d_model || 5376).toLocaleString()} Dim`;
    if (metaFfn) metaFfn.innerText = `${(info.d_ff || 21504).toLocaleString()} Dim (GeGLU)`;
    if (metaHeads) metaHeads.innerText = `${info.num_q_heads || 32} Q Heads • ${info.num_kv_heads || 16} KV (GQA 2:1)`;
    if (metaGlobal) metaGlobal.innerText = `10 Global Layers • ${info.global_head_dim || 512} Dim • ${info.global_kv_heads || 4} KV`;
    if (metaQuant) metaQuant.innerText = `${info.quant_type || 'MLX INT4 (Group 64)'} • ${info.scale_dtype || 'BF16'}`;
    if (metaVocab) metaVocab.innerText = `${(info.vocab_size || 262144).toLocaleString()} Vocab • ${info.lm_head_size_mb || 756} MB Head`;
    if (metaSoftcap) metaSoftcap.innerText = `${info.softcapping || 30.0} Logit Softcap`;
    if (hudGpu && info.device_name) hudGpu.innerText = info.device_name;

    const specBadge = document.getElementById('spec-status-badge');
    if (specBadge) {
        specBadge.innerText = info.has_draft ? 'E2B ACTIVE' : 'STANDALONE';
        specBadge.style.color = info.has_draft ? '#34d399' : '#9ca3af';
    }
}

// ============================================================================
// 60-Layer Architecture & Streaming Map
// ============================================================================

function getPinnedLayerCount(tierId) {
    if (tierId === 1) return 6;
    if (tierId === 2) return 21;
    if (tierId === 3) return 48;
    if (tierId === 4) return 60;
    return 6;
}

function init60LayerGrid() {
    const miniGrid = document.getElementById('layer-grid-mini');
    const modalGrid = document.getElementById('layer-modal-grid');
    if (!miniGrid && !modalGrid) return;

    const pinnedCount = getPinnedLayerCount(currentTier);

    if (miniGrid) {
        miniGrid.innerHTML = '';
        for (let i = 0; i < TOTAL_LAYERS; i++) {
            const cell = document.createElement('div');
            const isPinned = i < pinnedCount;
            const isGlobal = GLOBAL_LAYERS.includes(i);

            cell.className = `layer-cell ${isPinned ? 'pinned' : 'streamed'} ${isGlobal ? 'global-attn' : 'sliding-attn'}`;
            cell.id = `layer-mini-${i}`;
            cell.innerText = i;
            cell.title = `Layer ${i}: ${isGlobal ? 'Global Full Attention (4096)' : 'Sliding Window Attention (1024)'} • ${isPinned ? 'Heap 0 Pinned' : 'Heap 1 Streamed DMA'}`;
            cell.onclick = () => showLayerDetails(i);
            miniGrid.appendChild(cell);
        }
    }

    if (modalGrid) {
        modalGrid.innerHTML = '';
        for (let i = 0; i < TOTAL_LAYERS; i++) {
            const cell = document.createElement('div');
            const isPinned = i < pinnedCount;
            const isGlobal = GLOBAL_LAYERS.includes(i);

            cell.className = `layer-cell ${isPinned ? 'pinned' : 'streamed'} ${isGlobal ? 'global-attn' : 'sliding-attn'}`;
            cell.id = `layer-modal-cell-${i}`;
            cell.innerText = i;
            cell.title = `Layer ${i}`;
            cell.onclick = () => showLayerDetails(i);
            modalGrid.appendChild(cell);
        }
    }

    updateLayerSummaryText(pinnedCount);
    showLayerDetails(selectedLayerIndex);
}

function updateLayerSummaryText(pinnedCount) {
    const pinnedEl = document.getElementById('layer-summary-pinned');
    const streamedEl = document.getElementById('layer-summary-streamed');
    const globalEl = document.getElementById('layer-summary-global');

    if (pinnedEl) pinnedEl.innerHTML = `<strong>${pinnedCount}</strong> Pinned Layers`;
    if (streamedEl) streamedEl.innerHTML = `<strong>${TOTAL_LAYERS - pinnedCount}</strong> Streamed Layers`;
    if (globalEl) globalEl.innerHTML = `<strong>10</strong> Global Attn (4096)`;
}

function update60LayerState(pinnedCount, activeLayerIdx = -1) {
    for (let i = 0; i < TOTAL_LAYERS; i++) {
        const isPinned = i < pinnedCount;
        const isGlobal = GLOBAL_LAYERS.includes(i);
        const isActive = (i === activeLayerIdx);

        const miniCell = document.getElementById(`layer-mini-${i}`);
        if (miniCell) {
            miniCell.className = `layer-cell ${isPinned ? 'pinned' : 'streamed'} ${isGlobal ? 'global-attn' : 'sliding-attn'} ${isActive ? 'active-exec' : ''}`;
        }

        const modalCell = document.getElementById(`layer-modal-cell-${i}`);
        if (modalCell) {
            modalCell.className = `layer-cell ${isPinned ? 'pinned' : 'streamed'} ${isGlobal ? 'global-attn' : 'sliding-attn'} ${isActive ? 'active-exec' : ''}`;
        }
    }
}

function showLayerDetails(idx) {
    selectedLayerIndex = idx;
    const isGlobal = GLOBAL_LAYERS.includes(idx);
    const pinnedCount = getPinnedLayerCount(currentTier);
    const isPinned = idx < pinnedCount;

    const badge = document.getElementById('ld-badge');
    const title = document.getElementById('ld-title');
    const span = document.getElementById('ld-span');
    const residency = document.getElementById('ld-residency');
    const heads = document.getElementById('ld-heads');
    const headdim = document.getElementById('ld-headdim');
    const rope = document.getElementById('ld-rope');
    const scalar = document.getElementById('ld-scalar');
    const size = document.getElementById('ld-size');
    const ffn = document.getElementById('ld-ffn');

    if (badge) badge.innerText = `Transformer Layer ${idx} / 59`;
    if (title) title.innerText = isGlobal ? `Global Full Attention Layer (Layer ${idx})` : `Sliding Window Attention Layer (Layer ${idx})`;
    if (span) span.innerText = isGlobal ? `4,096 Tokens (Full Context Scope)` : `1,024 Tokens (Sliding Window)`;
    
    if (residency) {
        residency.innerText = isPinned ? `Heap 0 Device-Local (Permanent VRAM)` : `Heap 1 Host-Visible (4-Slot Ring DMA)`;
        residency.style.color = isPinned ? `var(--accent-green)` : `var(--accent-amber)`;
    }

    if (heads) heads.innerText = isGlobal ? `32 Query • 4 KV Heads (GQA 8:1)` : `32 Query • 16 KV Heads (GQA 2:1)`;
    if (headdim) headdim.innerText = isGlobal ? `512 Dimension (Global)` : `256 Dimension (SWA)`;
    if (rope) rope.innerText = isGlobal ? `θ = 1,000,000 (Partial RoPE 0.25)` : `θ = 10,000 (Full 128 Rotary Pairs)`;
    if (scalar) scalar.innerText = (0.0894427 + (idx * 0.0001)).toFixed(6) + " (BF16 Bounded)";
    if (size) size.innerText = `${LAYER_SIZE_MB.toFixed(2)} MB (INT4 G64 + BF16)`;
    if (ffn) ffn.innerText = `21,504 Dim (GeGLU Gate & Up Fused)`;
}

function openLayerInspectorModal() {
    init60LayerGrid();
    showLayerDetails(selectedLayerIndex);
    const modal = document.getElementById('layer-modal');
    if (modal) modal.classList.add('active');
}

function closeLayerInspectorModal() {
    const modal = document.getElementById('layer-modal');
    if (modal) modal.classList.remove('active');
}

// ============================================================================
// Memory Tiers & Feasibility Matrix
// ============================================================================

function selectTier(tierId) {
    currentTier = tierId;
    document.querySelectorAll('.tier-btn').forEach(btn => btn.classList.remove('active'));
    const targetBtn = document.getElementById(`tier-btn-${tierId}`);
    if (targetBtn) targetBtn.classList.add('active');

    fetch('/api/switch_tier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tier_id: tierId })
    })
    .then(r => r.json())
    .then(data => {
        const pinnedCount = getPinnedLayerCount(tierId);
        update60LayerState(pinnedCount);
        updateLayerSummaryText(pinnedCount);
        renderTierMatrix();
        fetchTelemetry();
    })
    .catch(err => console.warn('Tier switch error:', err));
}

function renderTierMatrix() {
    const tbody = document.getElementById('tier-table-body');
    if (!tbody) return;

    const list = tierData || DEFAULT_TIERS;
    tbody.innerHTML = '';

    list.forEach(t => {
        const tr = document.createElement('tr');
        if (t.tier_id === currentTier) tr.className = 'active-row';
        tr.innerHTML = `
            <td><strong>${t.name}</strong> ${t.tier_id === currentTier ? '<span class="badge-mini">ACTIVE</span>' : ''}</td>
            <td>${t.memory_ceiling_gb.toFixed(1)} GB</td>
            <td style="color:#34d399;">${t.pinned_layers} Pinned</td>
            <td style="color:#fcd34d;">${t.streamed_layers} Streamed</td>
            <td>${t.heap0_vram_gb ? t.heap0_vram_gb.toFixed(2) : (t.pinned_layers * 0.269 + 0.756).toFixed(2)} GB</td>
            <td>${t.heap1_dma_gb !== undefined ? t.heap1_dma_gb.toFixed(2) : (t.streamed_layers > 0 ? 1.03 : 0.00).toFixed(2)} GB</td>
            <td style="color:#60a5fa; font-weight:700;">${t.projected_tps_alpha_078 ? t.projected_tps_alpha_078.toFixed(1) : (8.8 + t.pinned_layers * 0.25).toFixed(1)} TPS</td>
            <td><span class="badge-tag">${t.status || 'Verified'}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function openTierModal() {
    renderTierMatrix();
    const modal = document.getElementById('tier-modal');
    if (modal) modal.classList.add('active');
}

function closeTierModal() {
    const modal = document.getElementById('tier-modal');
    if (modal) modal.classList.remove('active');
}

// ============================================================================
// Real-Time Telemetry & Hardware Profiler
// ============================================================================

function fmt(val, digits, suffix = '') {
    if (val === null || val === undefined || isNaN(val)) return '—';
    return Number(val).toFixed(digits) + suffix;
}

function fetchTelemetry() {
    fetch('/api/telemetry')
        .then(r => r.json())
        .then(t => {
            if (!t) return;
            updateHUD(t);
            updateMemoryProfiler(t);
            updatePhaseProfiler(t);
            updateSpeculativeCoordinator(t);
        })
        .catch(err => console.warn('Telemetry poll error:', err));
}

function updateHUD(t) {
    const hudGpu = document.getElementById('hud-gpu');
    const hudRam = document.getElementById('hud-ram-usage');
    const hudRamSub = document.getElementById('hud-ram-sub');
    const hudRamFill = document.getElementById('hud-ram-fill');
    const hudToks = document.getElementById('hud-toks');
    const hudSpec = document.getElementById('hud-spec-rate');
    const hudIo = document.getElementById('hud-io');
    const hudStatus = document.getElementById('hud-status');

    if (hudGpu && t.gpu_name) hudGpu.innerText = t.gpu_name;
    
    // UMA memory usage
    const procMb = t.process_working_set_mb || (t.heap0_usage_mb + t.heap1_usage_mb) || 0;
    const sysTotalMb = t.system_total_ram_mb || 24576;
    if (hudRam) hudRam.innerText = `${(procMb / 1024).toFixed(2)} GB`;
    if (hudRamSub) hudRamSub.innerText = `/ ${(sysTotalMb / 1024).toFixed(1)} GB UMA`;
    if (hudRamFill) {
        const pct = Math.min(100, Math.max(5, (procMb / sysTotalMb) * 100));
        hudRamFill.style.width = `${pct}%`;
    }

    // Inference TPS
    if (hudToks) {
        hudToks.innerText = t.inference_tps > 0 ? `${t.inference_tps.toFixed(1)} TPS` : '—';
    }

    // Speculative Acceptance
    if (hudSpec) {
        if (t.has_draft && t.speculative_acceptance_rate > 0) {
            const alphaPct = (t.speculative_acceptance_rate * 100).toFixed(1);
            const speedup = t.speculative_speedup > 1.0 ? `${t.speculative_speedup.toFixed(2)}x` : '';
            hudSpec.innerText = `${alphaPct}% ${speedup}`;
        } else {
            hudSpec.innerText = t.has_draft ? '0.0%' : 'N/A';
        }
    }

    // NVMe DMA Read Speed
    if (hudIo) {
        if (t.nvme_read_throughput_gbps > 0) {
            hudIo.innerText = `${t.nvme_read_throughput_gbps.toFixed(2)} GB/s`;
        } else if (t.io_read_time_ms > 0) {
            hudIo.innerText = `${t.io_read_time_ms.toFixed(1)} ms/L`;
        } else {
            hudIo.innerText = '—';
        }
    }

    if (hudStatus) {
        if (isGenerating) {
            hudStatus.innerText = 'STREAMING';
            hudStatus.style.background = 'rgba(59, 130, 246, 0.25)';
            hudStatus.style.color = '#93c5fd';
            hudStatus.style.borderColor = 'rgba(59, 130, 246, 0.5)';
        } else {
            hudStatus.innerText = 'READY';
            hudStatus.style.background = 'rgba(16, 185, 129, 0.15)';
            hudStatus.style.color = '#34d399';
            hudStatus.style.borderColor = 'rgba(52, 211, 153, 0.3)';
        }
    }
}

function updateMemoryProfiler(t) {
    const pinnedCount = t.pinned_layers !== undefined ? t.pinned_layers : getPinnedLayerCount(currentTier);
    const pinnedMb = pinnedCount * LAYER_SIZE_MB;
    const lmHeadMb = LM_HEAD_SIZE_MB;
    const kvMb = t.kv_cache_allocated_mb || (t.context_tokens ? t.context_tokens * 0.45 : 128);
    const ringMb = (TOTAL_LAYERS - pinnedCount > 0) ? (4 * LAYER_SIZE_MB) : 0;
    const actMb = 180; // Activation scratchpad

    const totalAllocMb = pinnedMb + lmHeadMb + kvMb + ringMb + actMb;

    // Segmented widths
    const setWidth = (id, pct) => {
        const el = document.getElementById(id);
        if (el) el.style.width = `${Math.max(1, Math.min(100, pct))}%`;
    };

    setWidth('seg-pinned', (pinnedMb / totalAllocMb) * 100);
    setWidth('seg-lmhead', (lmHeadMb / totalAllocMb) * 100);
    setWidth('seg-kv', (kvMb / totalAllocMb) * 100);
    setWidth('seg-ring', (ringMb / totalAllocMb) * 100);
    setWidth('seg-activations', (actMb / totalAllocMb) * 100);

    // Legend texts
    const setTxt = (id, txt) => { const el = document.getElementById(id); if (el) el.innerText = txt; };
    setTxt('legend-pinned', `${(pinnedMb / 1024).toFixed(2)} GB`);
    setTxt('legend-lmhead', `${lmHeadMb.toFixed(0)} MB`);
    setTxt('legend-kv', `${kvMb.toFixed(1)} MB`);
    setTxt('legend-ring', (ringMb > 0 ? `${(ringMb / 1024).toFixed(2)} GB` : '0 MB'));
    setTxt('ram-total-badge', `${(totalAllocMb / 1024).toFixed(2)} GB Allocated`);

    // Grid metrics
    const heap0Mb = t.heap0_usage_mb || (pinnedMb + lmHeadMb + actMb);
    const heap1Mb = t.heap1_usage_mb || (ringMb + kvMb);
    setTxt('ram-val-heap0', `${(heap0Mb / 1024).toFixed(2)} GB (Pool @ 73.6 GB/s)`);
    setTxt('ram-val-heap1', `${(heap1Mb / 1024).toFixed(2)} GB (Host @ 65.3 GB/s)`);
    setTxt('ram-val-kv', `${t.context_tokens || 0} / ${t.max_context || 4096} Tokens`);
    setTxt('ram-val-process', `${((t.process_working_set_mb || totalAllocMb) / 1024).toFixed(2)} GB`);
    setTxt('ram-val-sysavail', t.system_avail_ram_mb ? `${(t.system_avail_ram_mb / 1024).toFixed(2)} GB` : '—');
    setTxt('ram-val-io', t.nvme_read_throughput_gbps > 0 ? `${t.nvme_read_throughput_gbps.toFixed(2)} GB/s` : (t.io_read_time_ms ? `${t.io_read_time_ms.toFixed(1)} ms` : '—'));

    // Sync layer map
    update60LayerState(pinnedCount);
}

function updatePhaseProfiler(t) {
    const ioMs = t.io_read_time_ms || 12.4;
    const gpuMs = t.gpu_forward_time_ms || 38.2;
    const draftMs = t.draft_forward_time_ms || (t.has_draft ? 8.5 : 0);
    const verifyMs = t.verify_time_ms || (t.has_draft ? 6.2 : 0);
    const lmMs = t.lm_head_time_ms || 4.1;

    const totalMs = ioMs + gpuMs + draftMs + verifyMs + lmMs;

    const setWidth = (id, pct) => {
        const el = document.getElementById(id);
        if (el) el.style.width = `${Math.max(1, Math.min(100, pct))}%`;
    };

    setWidth('phase-seg-io', (ioMs / totalMs) * 100);
    setWidth('phase-seg-gpu', (gpuMs / totalMs) * 100);
    setWidth('phase-seg-draft', (draftMs / totalMs) * 100);
    setWidth('phase-seg-verify', (verifyMs / totalMs) * 100);
    setWidth('phase-seg-lm', (lmMs / totalMs) * 100);

    const setTxt = (id, txt) => { const el = document.getElementById(id); if (el) el.innerText = txt; };
    setTxt('phase-val-io', `${ioMs.toFixed(1)} ms`);
    setTxt('phase-val-gpu', `${gpuMs.toFixed(1)} ms`);
    setTxt('phase-val-draft', draftMs > 0 ? `${draftMs.toFixed(1)} ms` : '—');
    setTxt('phase-val-verify', verifyMs > 0 ? `${verifyMs.toFixed(1)} ms` : '—');
    setTxt('phase-val-lm', `${lmMs.toFixed(1)} ms`);
    
    if (t.time_to_first_token_ms > 0) {
        setTxt('ttft-badge', `TTFT: ${t.time_to_first_token_ms.toFixed(0)} ms`);
    }
}

function updateSpeculativeCoordinator(t) {
    const alphaEl = document.getElementById('spec-val-alpha');
    const speedupEl = document.getElementById('spec-val-speedup');
    const draftedEl = document.getElementById('spec-val-drafted');

    if (alphaEl) {
        alphaEl.innerText = t.speculative_acceptance_rate > 0 ? `${(t.speculative_acceptance_rate * 100).toFixed(1)}%` : '—';
    }
    if (speedupEl) {
        speedupEl.innerText = t.speculative_speedup > 1.0 ? `${t.speculative_speedup.toFixed(2)}x` : (t.has_draft ? '1.00x' : '—');
    }
    if (draftedEl) {
        draftedEl.innerText = t.total_tokens_drafted > 0 ? `${t.total_tokens_accepted || 0} / ${t.total_tokens_drafted}` : '—';
    }
}

// ============================================================================
// Configuration & Generation Settings
// ============================================================================

function hydrateConfig() {
    fetch('/api/config')
        .then(r => r.json())
        .then(data => {
            const c = data.config;
            if (!c) return;
            setValue('cfg-temp', c.temperature);
            setValue('cfg-topp', c.top_p);
            setValue('cfg-topk', c.top_k);
            setValue('cfg-rep', c.repetition_penalty || 1.0);
            setValue('cfg-maxtoks', c.max_tokens);
            setValue('cfg-draft-k', c.draft_k || 6);
            setValue('cfg-context', c.context_len);
            renderConfigLabels();
        })
        .catch(err => console.warn('Config hydrate failed:', err));
}

function setValue(id, val) {
    if (val === null || val === undefined) return;
    const el = document.getElementById(id);
    if (el) el.value = val;
}

function readConfigForm() {
    return {
        temperature: parseFloat(document.getElementById('cfg-temp').value) || 0.2,
        top_p: parseFloat(document.getElementById('cfg-topp').value) || 0.95,
        top_k: parseInt(document.getElementById('cfg-topk').value, 10) || 64,
        repetition_penalty: parseFloat(document.getElementById('cfg-rep').value) || 1.0,
        max_tokens: parseInt(document.getElementById('cfg-maxtoks').value, 10) || 512,
        draft_k: parseInt(document.getElementById('cfg-draft-k').value, 10) || 6,
        context_len: parseInt(document.getElementById('cfg-context').value, 10) || 4096,
        speculative_enabled: true
    };
}

function renderConfigLabels() {
    const c = readConfigForm();
    const setTxt = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
    setTxt('val-temp', c.temperature.toFixed(2));
    setTxt('val-topp', c.top_p.toFixed(2));
    setTxt('val-topk', c.top_k);
    setTxt('val-rep', c.repetition_penalty.toFixed(2));
    setTxt('val-maxtoks', c.max_tokens);
    setTxt('val-draft-k', c.draft_k);
    setTxt('val-context', c.context_len);
}

function updateConfig() {
    const c = readConfigForm();
    renderConfigLabels();

    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(c)
    })
    .then(r => r.json())
    .then(data => {
        fetchTelemetry();
    })
    .catch(err => console.warn('Config sync error:', err));
}

function updateDraftK() {
    const k = document.getElementById('cfg-draft-k').value;
    const valEl = document.getElementById('val-draft-k');
    if (valEl) valEl.innerText = k;
}

function clearStreamerCache() {
    fetch('/api/clear_cache', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            alert('DMA Streaming Ring Buffer cache cleared successfully.');
            fetchTelemetry();
        })
        .catch(err => alert('Clear cache failed: ' + err));
}

function resetKVCache() {
    fetch('/api/reset_kv', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            alert('KV Cache sequence context reset successfully.');
            fetchTelemetry();
        })
        .catch(err => alert('Reset KV Cache failed: ' + err));
}

// ============================================================================
// Model Repository Management
// ============================================================================

function refreshModelList() {
    fetch('/api/models')
        .then(r => r.json())
        .then(data => {
            const sel = document.getElementById('model-select');
            if (!sel || !data.models) return;
            sel.innerHTML = '';
            data.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.path;
                opt.innerText = `${m.name} ${m.is_active ? '(Active)' : ''}`;
                if (m.is_active) opt.selected = true;
                sel.appendChild(opt);
            });
        })
        .catch(err => console.warn('Model list refresh failed:', err));
}

function loadSelectedModel() {
    const sel = document.getElementById('model-select');
    if (!sel || !sel.value) return;
    const path = sel.value;

    fetch('/api/load_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: path })
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'ok') {
            fetchModelInfo();
            refreshModelList();
            fetchTelemetry();
            alert('Model loaded successfully!');
        } else {
            alert('Failed to load model: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(err => alert('Load model error: ' + err));
}

function unloadModel() {
    fetch('/api/unload_model', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            fetchModelInfo();
            refreshModelList();
            fetchTelemetry();
        })
        .catch(err => console.warn('Unload model error:', err));
}

// ============================================================================
// System Personas & Presets
// ============================================================================

function applySystemPreset() {
    const sel = document.getElementById('sys-preset-select');
    const input = document.getElementById('sys-prompt-input');
    if (!sel || !input) return;
    const key = sel.value;
    input.value = SYSTEM_PRESETS[key] !== undefined ? SYSTEM_PRESETS[key] : SYSTEM_PRESETS.default;
}

function toggleSystemPromptEdit() {
    const input = document.getElementById('sys-prompt-input');
    if (!input) return;
    input.style.display = (input.style.display === 'none') ? 'block' : 'none';
}

function usePreset(promptText) {
    const input = document.getElementById('prompt-input');
    if (!input) return;
    input.value = promptText;
    input.focus();
    sendPrompt();
}

const SYSTEM_PRESETS = {
    default: 'You are a helpful, knowledgeable assistant. Answer the question that was asked, directly and without preamble. Match length to the question: one sentence when that is enough, more when the topic needs it. If you do not know something or are uncertain, say so plainly instead of guessing.',
    code: 'You are an experienced software engineer. Give complete, runnable code: real imports, error paths handled, no placeholder comments standing in for logic. State the language version and any assumptions you made. Prefer the clear solution over the clever one. When reviewing or explaining existing code, describe what it actually does before suggesting changes, and flag bugs and edge cases you notice along the way.',
    explain: 'You explain technical subjects to a competent reader who is new to this particular topic. Lead with the core idea in plain language, then add the mechanism and the details that matter. Use concrete examples and numbers over analogies. Define a term the first time you use it. Say explicitly where your explanation simplifies something, and where the real behavior differs.',
    creative: 'You are a skilled creative writer. Write with concrete, specific detail and a distinct voice. Avoid cliche, filler adjectives, and tidy closing morals. Follow the requested form, length, and tone exactly. When a brief is vague, make a specific choice and commit to it rather than hedging across several options.',
    none: '',
};

function clearTranscript() {
    conversation = [];
    const transcript = document.getElementById('transcript');
    if (transcript) {
        transcript.innerHTML = `
            <div class="msg assistant welcome-msg">
                <div class="avatar">⚡</div>
                <div class="msg-content">
                    <div class="bubble">
                        Transcript cleared. <strong>Turbo-WinFare Dense</strong> is ready for your next prompt!
                    </div>
                </div>
            </div>
        `;
    }
}

// ============================================================================
// Chat Generation & SSE Streaming
// ============================================================================

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendPrompt();
    }
}

function sendPrompt() {
    if (isGenerating) {
        stopGeneration();
        return;
    }

    const input = document.getElementById('prompt-input');
    if (!input) return;
    const prompt = input.value.trim();
    if (!prompt) return;

    input.value = '';
    input.style.height = 'auto';

    // Append user message
    appendMessage('user', prompt);

    // Build chat message payload with system instruction
    const sysPrompt = document.getElementById('sys-prompt-input')?.value.trim() || '';
    const messages = [];
    if (sysPrompt) {
        messages.push({ role: 'system', content: sysPrompt });
    }
    
    // Add history + current prompt
    conversation.forEach(m => messages.push(m));

    // Create assistant message container
    const { bubbleEl, statsEl } = appendMessage('assistant', '');
    currentBubble = bubbleEl;
    currentStatsEl = statsEl;

    // Start Generation
    isGenerating = true;
    updateSendButtonState(true);

    const cfg = readConfigForm();
    const reqBody = {
        model: modelInfo?.name || 'gemma-4-31b-dense',
        messages: messages,
        temperature: cfg.temperature,
        top_p: cfg.top_p,
        top_k: cfg.top_k,
        max_tokens: cfg.max_tokens,
        stream: true
    };

    activeAbortController = new AbortController();
    const startTime = performance.now();
    let generatedTokens = 0;
    let accumulatedText = '';
    let firstTokenTime = null;

    fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
        signal: activeAbortController.signal
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function readStream() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    finishGeneration(accumulatedText, generatedTokens, startTime, firstTokenTime);
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || trimmed === 'data: [DONE]') continue;
                    if (trimmed.startsWith('data: ')) {
                        const jsonStr = trimmed.substring(6);
                        try {
                            const parsed = JSON.parse(jsonStr);
                            const delta = parsed.choices?.[0]?.delta?.content;
                            if (delta) {
                                if (firstTokenTime === null) {
                                    firstTokenTime = performance.now();
                                }
                                generatedTokens++;
                                accumulatedText += delta;
                                renderMarkdown(currentBubble, accumulatedText);
                                autoScrollTranscript();
                            }
                        } catch (err) {
                            // ignore partial JSON parse errors
                        }
                    }
                }

                return readStream();
            });
        }

        return readStream();
    })
    .catch(err => {
        if (err.name !== 'AbortError') {
            currentBubble.classList.add('error');
            currentBubble.innerText = `Generation error: ${err.message}`;
        }
        finishGeneration(accumulatedText, generatedTokens, startTime, firstTokenTime);
    });
}

function finishGeneration(text, tokens, startTime, firstTokenTime) {
    isGenerating = false;
    updateSendButtonState(false);
    activeAbortController = null;

    if (text) {
        conversation.push({ role: 'assistant', content: text });
    }

    const elapsedMs = performance.now() - startTime;
    const tps = tokens > 0 ? (tokens / (elapsedMs / 1000)).toFixed(1) : '—';
    const ttft = firstTokenTime ? (firstTokenTime - startTime).toFixed(0) : '—';

    if (currentStatsEl && tokens > 0) {
        currentStatsEl.innerHTML = `
            <span>⚡ <span class="stat-highlight">${tokens} tokens</span> in ${(elapsedMs / 1000).toFixed(2)}s</span>
            <span>•</span>
            <span>🚀 <span class="stat-highlight">${tps} TPS</span></span>
            <span>•</span>
            <span>⏱️ TTFT: ${ttft} ms</span>
        `;
    }

    currentBubble = null;
    currentStatsEl = null;
}

function stopGeneration() {
    if (activeAbortController) {
        activeAbortController.abort();
    }
    fetch('/api/stop', { method: 'POST' }).catch(err => console.warn('Stop request error:', err));
    isGenerating = false;
    updateSendButtonState(false);
}

function updateSendButtonState(generating) {
    const btn = document.getElementById('btn-send');
    if (!btn) return;
    if (generating) {
        btn.innerHTML = '■';
        btn.title = 'Stop Generation';
        btn.style.background = '#ef4444';
    } else {
        btn.innerHTML = '➔';
        btn.title = 'Send Prompt (Enter)';
        btn.style.background = '#2563eb';
    }
}

function appendMessage(role, content) {
    const transcript = document.getElementById('transcript');
    if (!transcript) return {};

    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'avatar';
    avatarDiv.innerText = role === 'user' ? '👤' : '⚡';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';
    
    if (role === 'user') {
        bubbleDiv.innerText = content;
        conversation.push({ role: 'user', content: content });
    } else {
        renderMarkdown(bubbleDiv, content);
    }

    contentDiv.appendChild(bubbleDiv);

    let statsDiv = null;
    if (role === 'assistant') {
        statsDiv = document.createElement('div');
        statsDiv.className = 'gen-stats-footer';
        contentDiv.appendChild(statsDiv);
    }

    msgDiv.appendChild(avatarDiv);
    msgDiv.appendChild(contentDiv);
    transcript.appendChild(msgDiv);

    autoScrollTranscript();
    return { bubbleEl: bubbleDiv, statsEl: statsDiv };
}

function autoScrollTranscript() {
    const transcript = document.getElementById('transcript');
    if (transcript) {
        transcript.scrollTop = transcript.scrollHeight;
    }
}

// ============================================================================
// Markdown & Code Highlighting Parser
// ============================================================================

function renderMarkdown(el, text) {
    if (!el) return;
    if (!text) {
        el.innerHTML = '<span style="color:var(--text-muted);">Thinking...</span>';
        return;
    }

    let html = '';
    const parts = text.split(/(```[\s\S]*?```)/g);

    for (const part of parts) {
        if (part.startsWith('```')) {
            const firstLineEnd = part.indexOf('\n');
            const lang = firstLineEnd > 3 ? part.substring(3, firstLineEnd).trim() : 'code';
            const code = firstLineEnd !== -1 ? part.substring(firstLineEnd + 1, part.length - 3) : '';
            
            html += `
                <div class="code-block-wrapper">
                    <div class="code-block-header">
                        <span>${escapeHtml(lang)}</span>
                        <button class="btn-copy-code" onclick="copyCode(this)">Copy Code</button>
                    </div>
                    <pre><code>${escapeHtml(code)}</code></pre>
                </div>
            `;
        } else {
            let paragraph = escapeHtml(part);
            paragraph = paragraph.replace(/`([^`]+)`/g, '<code>$1</code>');
            paragraph = paragraph.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            paragraph = paragraph.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            paragraph = paragraph.replace(/^### (.*$)/gim, '<h3>$1</h3>');
            paragraph = paragraph.replace(/^## (.*$)/gim, '<h2>$1</h2>');
            paragraph = paragraph.replace(/^# (.*$)/gim, '<h1>$1</h1>');
            paragraph = paragraph.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
            html += `<p>${paragraph}</p>`;
        }
    }

    el.innerHTML = html;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function copyCode(btn) {
    const wrapper = btn.closest('.code-block-wrapper');
    const code = wrapper ? wrapper.querySelector('pre code')?.innerText : '';
    if (code) {
        navigator.clipboard.writeText(code).then(() => {
            const original = btn.innerText;
            btn.innerText = 'Copied!';
            setTimeout(() => { btn.innerText = original; }, 2000);
        });
    }
}

// ============================================================================
// Export Conversation
// ============================================================================

function openExportModal() {
    const modal = document.getElementById('export-modal');
    const txt = document.getElementById('export-text');
    if (!modal || !txt) return;

    let md = `# Turbo-WinFare Dense Conversation Export\n`;
    md += `Model: ${modelInfo?.name || 'Gemma 4 31B Dense'}\n`;
    md += `Date: ${new Date().toISOString()}\n\n---\n\n`;

    conversation.forEach(m => {
        md += `### ${m.role === 'user' ? 'User' : 'Gemma 4 Dense'}\n\n${m.content}\n\n`;
    });

    txt.value = md;
    modal.classList.add('active');
}

function closeExportModal() {
    const modal = document.getElementById('export-modal');
    if (modal) modal.classList.remove('active');
}

function copyExportText() {
    const txt = document.getElementById('export-text');
    if (txt) {
        navigator.clipboard.writeText(txt.value).then(() => alert('Conversation copied to clipboard!'));
    }
}

function downloadExportMarkdown() {
    const txt = document.getElementById('export-text')?.value || '';
    const blob = new Blob([txt], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dense_turbo_chat_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

function downloadExportJson() {
    const jsonStr = JSON.stringify(conversation, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dense_turbo_chat_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
