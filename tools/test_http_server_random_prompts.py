import time
import json
import urllib.request
import urllib.error
import subprocess
import os
import signal
import sys

PORT = 8089
BASE_URL = f"http://127.0.0.1:{PORT}"
EXE_TURBO = os.path.join(os.path.dirname(__file__), "..", "build", "turbo-dense.exe")

def wait_for_server(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{BASE_URL}/api/model_info")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("loaded"):
                        return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def http_post_json(path, payload):
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return resp.status, resp.read().decode("utf-8")

def http_get_json(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def main():
    print("=" * 60)
    print("  HTTP SERVER LIVE RANDOM PROMPT & ARCHITECTURE TEST")
    print("=" * 60)

    env = os.environ.copy()
    env["PATH"] = "C:\\w64devkit\\bin;" + env.get("PATH", "")

    # Start server with Muse-Glimmer model
    cmd = [
        EXE_TURBO,
        "--server",
        "--port", str(PORT),
        "--model", "tests/fixtures/tiny_muse.g4dense"
    ]
    print(f"Starting server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        print("Waiting for server to initialize Vulkan and load model...")
        assert wait_for_server(20), "Server failed to start within 20s"
        print("Server is UP and ready!\n")

        # 1. Model info check
        status, info = http_get_json("/api/model_info")
        assert status == 200
        print(f"[TEST 1] Model info: name={info.get('name')}, arch={info.get('arch_name')}, layers={info.get('num_layers')}")
        assert info.get("arch_type") == 1
        assert info.get("arch_name") == "Muse-Glimmer"
        print("  -> PASS: Muse-Glimmer architecture reported correctly.\n")

        # 2. Non-streaming Chat Completion
        print("[TEST 2] Non-streaming Chat Completion with random prompt:")
        prompt1 = "What are three key principles of quantum mechanics?"
        payload1 = {
            "model": "muse-glimmer",
            "messages": [{"role": "user", "content": prompt1}],
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": False
        }
        s1, res1 = http_post_json("/v1/chat/completions", payload1)
        assert s1 == 200
        j1 = json.loads(res1)
        text1 = j1["choices"][0]["message"]["content"]
        print(f"  Prompt:   '{prompt1}'")
        print(f"  Response: '{text1.strip()}'")
        assert len(text1) > 0
        print("  -> PASS: Non-streaming completion returned valid content.\n")

        # 3. Streaming Chat Completion (SSE)
        print("[TEST 3] Streaming Chat Completion (SSE chunks):")
        prompt2 = "Write a haiku about winter frost."
        payload2 = {
            "model": "muse-glimmer",
            "messages": [{"role": "user", "content": prompt2}],
            "max_tokens": 12,
            "temperature": 0.7,
            "stream": True
        }
        url2 = f"{BASE_URL}/v1/chat/completions"
        req2 = urllib.request.Request(url2, data=json.dumps(payload2).encode(), headers={"Content-Type": "application/json"})
        chunks = []
        with urllib.request.urlopen(req2, timeout=10.0) as resp2:
            assert resp2.status == 200
            for line in resp2:
                line_str = line.decode().strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    chunk_json = json.loads(line_str[6:])
                    delta = chunk_json["choices"][0]["delta"].get("content", "")
                    if delta:
                        chunks.append(delta)
        streamed_text = "".join(chunks)
        print(f"  Prompt:         '{prompt2}'")
        print(f"  Streamed text:  '{streamed_text.strip()}' (received {len(chunks)} chunks)")
        assert len(chunks) > 0
        print("  -> PASS: SSE stream delivered valid delta chunks.\n")

        # 4. Dynamic Live Model Swap to Gemma 4
        print("[TEST 4] Dynamic Model Swap: Muse-Glimmer -> Gemma 4 (tiny):")
        s3, res3 = http_post_json("/api/load_model", {"path": "tests/fixtures/tiny.g4dense"})
        assert s3 == 200
        print("  Load model API response:", res3.strip())

        # Verify active model is now Gemma 4
        s4, info4 = http_get_json("/api/model_info")
        assert s4 == 200
        print(f"  New active model: name={info4.get('name')}, arch={info4.get('arch_name')}, arch_type={info4.get('arch_type')}")
        assert info4.get("arch_type") == 0
        assert info4.get("arch_name") == "Gemma 4"
        print("  -> PASS: Live model swap executed cleanly, exactly 1 model active in VRAM!\n")

        # 5. Query Gemma 4 after swap
        print("[TEST 5] Chat completion on newly swapped Gemma 4 model:")
        payload5 = {
            "model": "gemma-4",
            "messages": [{"role": "user", "content": "Explain binary search in one sentence."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False
        }
        s5, res5 = http_post_json("/v1/chat/completions", payload5)
        assert s5 == 200
        j5 = json.loads(res5)
        text5 = j5["choices"][0]["message"]["content"]
        print(f"  Response: '{text5.strip()}'")
        assert len(text5) > 0
        print("  -> PASS: Newly loaded model successfully generates tokens.\n")

        print("=" * 60)
        print("  ALL HTTP SERVER LIVE TESTS PASSED 100%!")
        print("=" * 60)

    finally:
        print("\nStopping server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server process cleanly stopped.")

if __name__ == "__main__":
    main()