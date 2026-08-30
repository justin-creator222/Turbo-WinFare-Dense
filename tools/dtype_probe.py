"""Which reading of the non-quantized tensors is the real one: IEEE FP16 or BF16?

The safetensors header tags every non-quantized tensor BF16, but the two readings of the same
bytes are a bijection on this value range, so 'which looks nicer' is circular. This measures
the two things that are NOT circular:

  1. attention score magnitude before softmax (self.scaling = 1.0 upstream, so nothing else
     controls it), and
  2. how a 1e-6 relative perturbation of the layer-0 input propagates through the stack.

A working transformer has O(1..10) pre-softmax scores and does not amplify rounding.
"""
import glob, json, os, struct, sys
import numpy as np


CKPT = os.path.join("models", "gemma-4-31b-it-4bit")

idx = {}
for p in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        idx[k] = (p, base + s, e - s, tuple(v["shape"]))


def raw(name):
    key = name if name in idx else "language_model." + name
    p, o, nb, sh = idx[key]
    with open(p, "rb") as f:
        f.seek(o)
        return f.read(nb), sh


def as_bf16(d):
    return (np.frombuffer(d, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def as_fp16(d):
    return np.frombuffer(d, dtype=np.float16).astype(np.float32)


def norm_weight(name, mode):
    d, sh = raw(name)
    return (as_fp16(d) if mode == "fp16" else as_bf16(d)).reshape(sh)


def dequant(prefix):
    wd, wsh = raw(prefix + ".weight")
    sd, ssh = raw(prefix + ".scales")
    bd, _ = raw(prefix + ".biases")
    rows, pc = wsh
    cols = pc * 8
    groups = ssh[1]
    g = cols // groups
    packed = np.frombuffer(wd, dtype=np.uint32).reshape(rows, pc)
    q = np.empty((rows, cols), dtype=np.float32)
    for n in range(8):
        q[:, n::8] = ((packed >> (n * 4)) & 0xF).astype(np.float32)
    return q * np.repeat(as_bf16(sd).reshape(rows, groups), g, axis=1) \
             + np.repeat(as_bf16(bd).reshape(rows, groups), g, axis=1)


def rms(x, w, eps=1e-6):
    return x / np.sqrt(np.mean(x * x) + eps) * w


cfg = json.load(open(os.path.join(CKPT, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
d_model, head_dim = tc["hidden_size"], tc["head_dim"]
n_q, n_kv = tc["num_attention_heads"], tc["num_key_value_heads"]
eps = tc.get("rms_norm_eps", 1e-6)
theta = tc["rope_parameters"]["sliding_attention"]["rope_theta"]
NL = int(os.environ.get("PROBE_LAYERS", "8"))

embed = dequant("model.embed_tokens")
for mode in ("fp16", "bf16"):
    print("\n" + "=" * 72)
    print("  non-quantized tensors read as %s" % mode.upper())
    print("=" * 72)

    x0 = embed[2].astype(np.float32) * np.sqrt(np.float32(d_model))
    # A perturbed twin, 1e-6 relative. Its divergence measures the stack's conditioning.
    rng = np.random.default_rng(0)
    x1 = x0 * (1.0 + 1e-6 * rng.standard_normal(d_model).astype(np.float32))

    for l in range(NL):
        P = "model.layers.%d." % l
        if tc["layer_types"][l] != "sliding_attention":
            print("  layer %2d GLOBAL  (skipped: probe covers sliding layers only)" % l)
            continue
        W_in = norm_weight(P + "input_layernorm.weight", mode)
        W_pa = norm_weight(P + "post_attention_layernorm.weight", mode)
        W_pf = norm_weight(P + "pre_feedforward_layernorm.weight", mode)
        W_po = norm_weight(P + "post_feedforward_layernorm.weight", mode)
        qn = norm_weight(P + "self_attn.q_norm.weight", mode)
        kn = norm_weight(P + "self_attn.k_norm.weight", mode)
        ls = (as_fp16 if mode == "fp16" else as_bf16)(raw(P + "layer_scalar")[0])[0]
        Wq, Wk, Wv = dequant(P + "self_attn.q_proj"), dequant(P + "self_attn.k_proj"), dequant(P + "self_attn.v_proj")
        Wo = dequant(P + "self_attn.o_proj")
        Wg, Wu, Wd = dequant(P + "mlp.gate_proj"), dequant(P + "mlp.up_proj"), dequant(P + "mlp.down_proj")

        scores_seen = []
        outs = []
        for xi, x in enumerate((x0, x1)):
            res = x.copy()
            h = rms(x, W_in, eps)
            q, k, v = Wq @ h, Wk @ h, Wv @ h
            for i in range(n_q):
                sl = slice(i * head_dim, (i + 1) * head_dim)
                q[sl] = rms(q[sl], qn, eps)
            for i in range(n_kv):
                sl = slice(i * head_dim, (i + 1) * head_dim)
                k[sl] = rms(k[sl], kn, eps)
                v[sl] = rms(v[sl], 1.0, eps)
            # Position 0: RoPE is identity, and softmax over one key is 1. Record the score
            # magnitude that WOULD go into softmax -- that is what the probe is after.
            if xi == 0:
                for i in range(n_q):
                    kvh = i // (n_q // n_kv)
                    scores_seen.append(float(q[i * head_dim:(i + 1) * head_dim]
                                             @ k[kvh * head_dim:(kvh + 1) * head_dim]))
            ctx = np.empty(n_q * head_dim, dtype=np.float32)
            for i in range(n_q):
                kvh = i // (n_q // n_kv)
                ctx[i * head_dim:(i + 1) * head_dim] = v[kvh * head_dim:(kvh + 1) * head_dim]
            a = rms(Wo @ ctx, W_pa, eps)
            x = res + a
            res = x.copy()
            h2 = rms(x, W_pf, eps)
            g_ = Wg @ h2
            act = 0.5 * g_ * (1.0 + np.tanh(0.7978845608028654 * (g_ + 0.044715 * g_ ** 3)))
            x = res + rms(Wd @ (act * (Wu @ h2)), W_po, eps)
            outs.append(x * ls)
        x0, x1 = outs
        s = np.array(scores_seen)
        rel = np.abs(x1 - x0).max() / max(np.abs(x0).max(), 1e-30)
        print("  layer %2d  |score| mean %8.2f max %8.2f   out_rms %10.4g   perturbation rel %.3e"
              % (l, np.abs(s).mean(), np.abs(s).max(), float(np.sqrt(np.mean(x0 * x0))), rel))
