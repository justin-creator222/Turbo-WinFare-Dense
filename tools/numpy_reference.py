"""Independent NumPy reference for the Gemma 4 31B forward pass.

Reads the MLX safetensors checkpoint directly, recomputes the forward pass, and diffs every
layer against the C++ CPU reference's --dump-tensors output. The first tensor that diverges
localizes the defect.

This exists because GPU-vs-CPU parity cannot detect a convention both paths get wrong, and
several such bugs have shipped that way (LayerNorm dtype, attention scale, V-from-K, missing
v_norm, layer_scalar placement). An implementation written from the checkpoint and the
published architecture, sharing no code with the engine, is what catches those.

**It is not the authority.** It was itself written from the same assumptions as the engine, so
agreement only proves the two implement the same intent. `transformers/models/gemma4` settles
what that intent should be; see docs/FORWARD_PASS.md section 6 for the conventions it fixed.

  # Position 0 only, all 60 layers
  python tools/numpy_reference.py --layers 60

  # Multi-position: prefill a token sequence with a real KV cache and sliding-window masking.
  # Requires dumps from a matching prefill:
  #   run_cpu_reference_test.exe <model> <dump-dir> 2 "2,1596,1902,1117"
  python tools/numpy_reference.py --layers 6 --tokens 2,1596,1902,1117

Deliberately slow and simple: one token at a time, full float32, no batching.
"""

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

CKPT = os.path.join("models", "gemma-4-31b-it-4bit")


# ----------------------------------------------------------------------------------------
# Checkpoint access
# ----------------------------------------------------------------------------------------

def build_index(ckpt_dir):
    """Maps tensor name -> (file, absolute data offset, nbytes, dtype, shape)."""
    index = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
            base = 8 + hdr_len
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            index[name] = (path, base + start, end - start, meta["dtype"], tuple(meta["shape"]))
    return index


def raw(index, name):
    key = name if name in index else "language_model." + name
    if key not in index:
        raise KeyError(f"{name} not in checkpoint")
    path, off, nbytes, dtype, shape = index[key]
    with open(path, "rb") as f:
        f.seek(off)
        data = f.read(nbytes)
    return data, dtype, shape


def as_bf16(data):
    u = np.frombuffer(data, dtype=np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def as_fp16(data):
    return np.frombuffer(data, dtype=np.float16).astype(np.float32)


def norm_weight(index, name):
    """LayerNorm-family tensors. BF16, exactly as the safetensors header says.

    An earlier version of this file read them as IEEE FP16, on the grounds that the BF16
    decode "looks wrong" (model.norm.weight median 6.28, max 510) while FP16 looks like a
    textbook norm weight. On this value range the two decodings are a bijection, so both are
    self-consistent and appearance cannot choose between them -- and this reference agreed
    with the engine only because both made the same assumption. Behaviour decides it: read as
    FP16, pre-softmax scores run 192-327 (max 549) and the hidden state doubles every layer;
    read as BF16 they run 5-16 (max 24) and the residual stream is stable at rms ~1.7."""
    data, _, shape = raw(index, name)
    return as_bf16(data).reshape(shape)


def dequant(index, prefix):
    """Affine INT4, group 64, BF16 scales and biases: w = q * scale + bias."""
    wdata, _, wshape = raw(index, prefix + ".weight")
    sdata, _, sshape = raw(index, prefix + ".scales")
    bdata, _, _ = raw(index, prefix + ".biases")

    rows, packed_cols = wshape
    cols = packed_cols * 8
    groups = sshape[1]
    gsize = cols // groups

    packed = np.frombuffer(wdata, dtype=np.uint32).reshape(rows, packed_cols)
    scales = as_bf16(sdata).reshape(rows, groups)
    biases = as_bf16(bdata).reshape(rows, groups)

    # Unpack 8 nibbles per uint32, low nibble first.
    q = np.empty((rows, cols), dtype=np.float32)
    for n in range(8):
        q[:, n::8] = ((packed >> (n * 4)) & 0xF).astype(np.float32)

    s = np.repeat(scales, gsize, axis=1)
    b = np.repeat(biases, gsize, axis=1)
    return q * s + b


# ----------------------------------------------------------------------------------------
# Reference maths
# ----------------------------------------------------------------------------------------

def rms_norm(x, w, eps=1e-6, plus_one=False):
    """Gemma RMSNorm: out = x_hat * w, a plain multiply (Gemma3nRMSNorm, NOT the 1+w form).

    `w` may be the scalar 1.0 for a with_scale=False norm. `plus_one` is retained only to
    document the convention this model does NOT use."""
    xhat = x / np.sqrt(np.mean(x * x) + eps)
    return xhat * ((1.0 + w) if plus_one else w)


def rope(vec, pos, head_dim, n_pairs, theta):
    """NeoX split-half rotation over the first `n_pairs` pairs."""
    out = vec.copy()
    half = head_dim // 2
    for p in range(n_pairs):
        freq = 1.0 / (theta ** (2.0 * p / head_dim))
        ang = pos * freq
        c, s = np.cos(ang), np.sin(ang)
        x0, x1 = vec[p], vec[p + half]
        out[p] = x0 * c - x1 * s
        out[p + half] = x0 * s + x1 * c
    return out


def stats(name, a):
    return "%-30s mean=%+10.5f std=%9.5f min=%+10.4f max=%+10.4f rms=%9.5f" % (
        name, a.mean(), a.std(), a.min(), a.max(), np.sqrt(np.mean(a * a)))


def compare(name, ours, dump_dir, filename):
    path = os.path.join(dump_dir, filename)
    if not os.path.exists(path):
        print("  (no dump at %s to compare)" % path)
        return
    theirs = np.fromfile(path, dtype=np.float32)
    if theirs.size != ours.size:
        print("  SIZE MISMATCH %s: numpy=%d engine=%d" % (name, ours.size, theirs.size))
        return
    d = np.abs(ours - theirs)
    verdict = "MATCH" if d.max() < 1e-2 else "*** DIVERGES ***"
    print("  %-22s max|diff|=%.6e  mean|diff|=%.6e   %s" % (name, d.max(), d.mean(), verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=os.path.join("tests", "fixtures", "oracle_tensors"))
    ap.add_argument("--token", type=int, default=2)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layers", type=int, default=6,
                    help="layers to reproduce; 6 reaches the first global layer (5)")
    ap.add_argument("--tokens", default=None,
                    help="comma-separated token sequence to prefill. Attention over history is "
                         "only exercised with two or more. Defaults to the single --token.")
    args = ap.parse_args()

    if args.tokens:
        tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    else:
        tokens = [args.token]

    if not os.path.isdir(args.ckpt):
        print("checkpoint not found:", args.ckpt)
        return 1

    print("Indexing checkpoint ...")
    idx = build_index(args.ckpt)

    cfg = json.load(open(os.path.join(args.ckpt, "config.json"), encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    d_model = tc["hidden_size"]
    head_dim = tc["head_dim"]
    n_q = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    eps = tc.get("rms_norm_eps", 1e-6)
    theta = tc["rope_parameters"]["sliding_attention"]["rope_theta"]

    print("d_model=%d head_dim=%d q_heads=%d kv_heads=%d" % (d_model, head_dim, n_q, n_kv))

    # ---- Embedding -------------------------------------------------------------------
    print("\nEmbedding")
    embed_mat = dequant(idx, "model.embed_tokens")
    scale = np.sqrt(np.float32(d_model))
    xs = [embed_mat[t].astype(np.float32) * scale for t in tokens]
    for pos, x in enumerate(xs):
        print("  pos %d token %-7d %s" % (pos, tokens[pos], stats("embed (scaled)", x)))
        compare("embed@%d" % pos, x, args.dump_dir, "token%d_embed.bin" % pos)

    # ---- Layers -----------------------------------------------------------------------
    layer_types = tc["layer_types"]
    n_global_kv = tc.get("num_global_key_value_heads", 4)
    global_hd = tc.get("global_head_dim", 512)
    theta_glob = tc["rope_parameters"]["full_attention"]["rope_theta"]
    prf = tc["rope_parameters"]["full_attention"].get("partial_rotary_factor", 0.25)
    sliding_window = tc.get("sliding_window", 1024)

    print("\nLayers 0..%d over %d position(s), each diffed against the engine dump"
          % (args.layers - 1, len(xs)))

    # Layer-outer / position-inner. A layer's weights are dequantized once and reused for
    # every position, which matters: dequantizing all 60 layers is already the bulk of the
    # runtime. The ordering is valid because layer l at position p needs only the layer l-1
    # output at position p (kept in `xs`) plus layer l's own K/V at positions <= p.
    for l in range(args.layers):
        P = "model.layers.%d." % l
        is_global = layer_types[l] == "full_attention"
        hd = global_hd if is_global else head_dim
        nkv = n_global_kv if is_global else n_kv
        theta_l = theta_glob if is_global else theta
        pairs = (int(prf * hd) // 2) if is_global else (hd // 2)

        W_in_norm = norm_weight(idx, P + "input_layernorm.weight")
        W_post_attn = norm_weight(idx, P + "post_attention_layernorm.weight")
        W_pre_ffn = norm_weight(idx, P + "pre_feedforward_layernorm.weight")
        W_post_ffn = norm_weight(idx, P + "post_feedforward_layernorm.weight")
        qn = norm_weight(idx, P + "self_attn.q_norm.weight")
        kn = norm_weight(idx, P + "self_attn.k_norm.weight")
        W_q = dequant(idx, P + "self_attn.q_proj")
        W_k = dequant(idx, P + "self_attn.k_proj")
        W_v = None if is_global else dequant(idx, P + "self_attn.v_proj")
        W_o = dequant(idx, P + "self_attn.o_proj")
        W_gate = dequant(idx, P + "mlp.gate_proj")
        W_up = dequant(idx, P + "mlp.up_proj")
        W_down = dequant(idx, P + "mlp.down_proj")
        layer_scalar = as_bf16(raw(idx, P + "layer_scalar")[0])[0]

        k_cache = []   # one [nkv*hd] vector per position
        v_cache = []

        for pos in range(len(xs)):
            x = xs[pos]
            residual = x.copy()
            h = rms_norm(x, W_in_norm, eps, False)

            q = W_q @ h
            k = W_k @ h
            # Full-attention layers have no v_proj (attention_k_eq_v), so upstream binds
            # value_states to the RAW k_proj output; k_norm then returns a NEW tensor and
            # rebinds key_states, leaving V untouched by it. Copy before touching k.
            v = k.copy() if is_global else (W_v @ h)

            for i in range(n_q):
                sl = slice(i * hd, (i + 1) * hd)
                q[sl] = rope(rms_norm(q[sl], qn, eps, False), pos, hd, pairs, theta_l)
            for i in range(nkv):
                sl = slice(i * hd, (i + 1) * hd)
                k[sl] = rms_norm(k[sl], kn, eps, False)
                # v_norm = Gemma4RMSNorm(head_dim, with_scale=False): UNWEIGHTED, and applied
                # on every layer -- only is_kv_shared_layer skips it, and this config sets
                # num_kv_shared_layers = 0.
                v[sl] = rms_norm(v[sl], 1.0, eps, False)
            for i in range(nkv):
                sl = slice(i * hd, (i + 1) * hd)
                k[sl] = rope(k[sl], pos, hd, pairs, theta_l)

            k_cache.append(k)
            v_cache.append(v)

            # Causal attention over [first, pos]. Sliding layers see only the last
            # `sliding_window` positions; full-attention layers see everything.
            first = 0 if is_global else max(0, pos - sliding_window + 1)
            span = list(range(first, pos + 1))

            ctx = np.empty(n_q * hd, dtype=np.float32)
            for i in range(n_q):
                kvh = i // (n_q // nkv)
                ksl = slice(kvh * hd, (kvh + 1) * hd)
                qh = q[i * hd:(i + 1) * hd]
                # self.scaling = 1.0 in Gemma4TextAttention -- no head_dim^-0.5 here.
                sc = np.array([float(qh @ k_cache[t][ksl]) for t in span], dtype=np.float32)
                sc -= sc.max()
                w = np.exp(sc)
                w /= w.sum()
                acc = np.zeros(hd, dtype=np.float32)
                for wi, t in zip(w, span):
                    acc += wi * v_cache[t][ksl]
                ctx[i * hd:(i + 1) * hd] = acc

            attn = W_o @ ctx
            attn = rms_norm(attn, W_post_attn, eps, False)
            x = residual + attn

            residual = x.copy()
            h2 = rms_norm(x, W_pre_ffn, eps, False)
            gate = W_gate @ h2
            up = W_up @ h2
            act = 0.5 * gate * (1.0 + np.tanh(0.7978845608028654 * (gate + 0.044715 * gate ** 3)))
            ffn = W_down @ (act * up)
            ffn = rms_norm(ffn, W_post_ffn, eps, False)
            x = residual + ffn

            # `hidden_states *= self.layer_scalar` -- the last statement of the decoder layer,
            # after both residual adds. The per-layer-input block that would otherwise sit
            # between them is skipped when hidden_size_per_layer_input == 0, as it is here.
            # layer_scalar is genuinely BF16 (0.089355 at layer 0), unlike the LayerNorm family.
            x = x * layer_scalar
            xs[pos] = x

        for pos, x in enumerate(xs):
            print("  layer %2d %-8s pos %d  rms=%9.5f"
                  % (l, "GLOBAL" if is_global else "sliding", pos, np.sqrt(np.mean(x * x))))
            compare("layer%d@%d" % (l, pos), x, args.dump_dir,
                    "token%d_layer%d_hidden.bin" % (pos, l))

    # ---- Final norm + LM head (only meaningful once every layer has run) ---------------
    if args.layers >= len(layer_types):
        print("\nFinal norm and LM head")
        fn = norm_weight(idx, "model.norm.weight")
        cap = tc.get("final_logit_softcapping", 30.0)
        for pos, x in enumerate(xs):
            normed = rms_norm(x, fn, eps, False)
            print("  pos %d %s" % (pos, stats("final_normed", normed)))
            compare("final_normed@%d" % pos, normed, args.dump_dir,
                    "token%d_final_normed.bin" % pos)

            # Tied head: the embedding matrix reused as the output projection.
            logits = embed_mat @ normed
            if cap:
                logits = cap * np.tanh(logits / cap)
            print("  pos %d %s" % (pos, stats("logits", logits)))
            top = np.argsort(logits)[::-1][:5]
            print("  pos %d numpy top-5 token ids: %s" % (pos, top.tolist()))
            compare("logits@%d" % pos, logits, args.dump_dir, "token%d_logits.bin" % pos)

    print("\nThe FIRST tensor that DIVERGES localizes the defect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
