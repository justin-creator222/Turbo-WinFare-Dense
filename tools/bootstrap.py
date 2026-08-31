"""
Bootstrap and verification script for Turbo-WinFare Dense toolchain.

Manages and verifies:
  * w64devkit       -> C:\\w64devkit          (g++, as, ld, etc.)
  * cmake / ninja   -> PATH (or $TURBO_SIBLING_ROOT/.venv/Scripts, if set)
  * DXC             -> build/dxc.exe, build/dxcompiler.dll, build/dxil.dll
  * Vulkan-Headers  -> third_party/vulkan/include (vulkan/vulkan.h, etc.)
  * vulkan-1.dll    -> C:\\Windows\\System32\\vulkan-1.dll

Usage:
    python tools/bootstrap.py           # Fetch missing tools and headers
    python tools/bootstrap.py --verify  # Verify all tools and compile test shader to SPIR-V
    python tools/bootstrap.py --force   # Re-download / refresh all dependencies
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
W64DEVKIT_DIR = Path("C:/w64devkit")
# Optional local fallback for cmake/ninja/DXC.
#
# This was a hardcoded path to the author's machine, which is useless to anyone else and
# leaks a private repo's layout. It is now opt-in: set TURBO_SIBLING_ROOT to a checkout that
# already has these tools built, or leave it unset and everything is fetched from upstream.
_SIBLING_ROOT = os.environ.get("TURBO_SIBLING_ROOT", "").strip()
SIBLING_VENV_SCRIPTS = Path(_SIBLING_ROOT) / ".venv" / "Scripts" if _SIBLING_ROOT else None
SIBLING_BUILD = Path(_SIBLING_ROOT) / "build" if _SIBLING_ROOT else None
BUILD_DIR = REPO_ROOT / "build"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
VULKAN_INCLUDE_DIR = THIRD_PARTY_DIR / "vulkan" / "include"

DXC_FILES = ("dxc.exe", "dxcompiler.dll", "dxil.dll")
UA = {"User-Agent": "turbo-dense-toolchain"}


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _download(url):
    print(f"    downloading {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _latest_release_asset(repo, predicate):
    """Returns (asset_name, download_url) for the first asset matching predicate."""
    data = _get_json(f"https://api.github.com/repos/{repo}/releases/latest")
    for asset in data.get("assets", []):
        name = asset.get("name", "")
        if predicate(name):
            return name, asset["browser_download_url"]
    raise RuntimeError(f"No matching asset in latest release of {repo}")


def _w64devkit_asset(name):
    """Picks the x64 build out of a release's asset list.

    w64devkit stopped publishing .zip assets after v1.23.0: v2.0.0 and v2.1.0 ship a
    plain self-extracting .exe, and v2.2.0 onward ship <name>.7z.exe. Matching only
    ".zip" therefore selected nothing and every fresh install died with "No matching
    asset in latest release". That went unnoticed for a long time because this
    function returns early whenever C:\\w64devkit already exists -- it only ever
    failed on a clean machine, which is exactly where CI runs.
    """
    n = name.lower()
    if n.endswith(".sig"):
        return False
    if not n.startswith("w64devkit-x64-"):
        return False
    return n.endswith(".exe") or n.endswith(".zip")


def install_w64devkit(force=False):
    gxx = W64DEVKIT_DIR / "bin" / "g++.exe"
    if gxx.exists() and not force:
        print(f"[skip] w64devkit already present at {W64DEVKIT_DIR}")
        return

    print("[w64devkit] resolving latest release...")
    name, url = _latest_release_asset("skeeto/w64devkit", _w64devkit_asset)
    print(f"[w64devkit] {name}")
    blob = _download(url)

    parent = W64DEVKIT_DIR.parent
    print(f"[w64devkit] extracting to {W64DEVKIT_DIR}...")

    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(parent)
    else:
        installer = Path(tempfile.gettempdir()) / name
        installer.write_bytes(blob)
        try:
            subprocess.run([str(installer), "-y", f"-o{parent}"],
                           check=False,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        finally:
            installer.unlink(missing_ok=True)

    if not gxx.exists():
        raise RuntimeError(
            f"Extraction finished but {gxx} is missing.\n"
            f"    If {parent} is not writable by this account, re-run from an\n"
            f"    elevated shell, or extract {name} by hand so that {gxx} exists."
        )

    probe = subprocess.run([str(gxx), "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"{gxx} exists but failed to run:\n{probe.stderr.strip()}")
    version = probe.stdout.splitlines()[0] if probe.stdout else "unknown version"

    print(f"[w64devkit] OK -> {gxx} ({version})")
    print("           NOTE: add C:\\w64devkit\\bin to PATH, or g++ cannot find 'as'.")


def install_dxc(force=False):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    all_present = all((BUILD_DIR / f).exists() for f in DXC_FILES)
    if all_present and not force:
        print(f"[skip] DXC already present in {BUILD_DIR}")
        return

    # Check if sibling build has dxcompiler.dll and dxil.dll that we can copy if offline
    # but we also need dxc.exe
    print("[dxc] resolving latest DirectXShaderCompiler release...")
    try:
        name, url = _latest_release_asset(
            "microsoft/DirectXShaderCompiler",
            lambda n: n.lower().startswith("dxc_") and n.lower().endswith(".zip"),
        )
        print(f"[dxc] {name}")
        blob = _download(url)

        wanted = {}
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for entry in zf.namelist():
                base = os.path.basename(entry).lower()
                if base in [f.lower() for f in DXC_FILES] and "/x64/" in entry.replace("\\", "/").lower():
                    wanted[base] = zf.read(entry)

        for fname in DXC_FILES:
            if fname.lower() in wanted:
                out = BUILD_DIR / fname
                out.write_bytes(wanted[fname.lower()])
                print(f"[dxc] wrote {out} ({len(wanted[fname.lower()]):,} bytes)")
            else:
                print(f"[dxc] warning: {fname} not found in zip bin/x64/")
    except Exception as e:
        print(f"[dxc] GitHub fetch failed: {e}")
        # Fallback copy from sibling build if available
        if SIBLING_BUILD is not None and SIBLING_BUILD.exists():
            for fname in ("dxcompiler.dll", "dxil.dll"):
                src = SIBLING_BUILD / fname
                if src.exists():
                    shutil.copy2(src, BUILD_DIR / fname)
                    print(f"[dxc] copied {fname} from $TURBO_SIBLING_ROOT/build")
        if not (BUILD_DIR / "dxc.exe").exists():
            raise RuntimeError(f"Could not install dxc.exe: {e}")

    print("[dxc] OK -- DXC is ready.")


def install_vulkan_headers(force=False):
    vk_header = VULKAN_INCLUDE_DIR / "vulkan" / "vulkan.h"
    if vk_header.exists() and not force:
        print(f"[skip] Vulkan headers already present at {VULKAN_INCLUDE_DIR}")
        return

    print("[vulkan-headers] resolving latest Vulkan-Headers release...")
    VULKAN_INCLUDE_DIR.mkdir(parents=True, exist_ok=True)

    # Download repository zip archive from KhronosGroup/Vulkan-Headers (main branch archive)
    url = "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/heads/main.zip"
    try:
        blob = _download(url)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            # Extract everything under include/ to VULKAN_INCLUDE_DIR
            prefix = "Vulkan-Headers-main/include/"
            count = 0
            for entry in zf.namelist():
                if entry.startswith(prefix) and not entry.endswith("/"):
                    rel_path = entry[len(prefix):]
                    out_path = VULKAN_INCLUDE_DIR / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(zf.read(entry))
                    count += 1
            print(f"[vulkan-headers] extracted {count} header files to {VULKAN_INCLUDE_DIR}")
    except Exception as e:
        print(f"[vulkan-headers] GitHub archive download failed: {e}")
        raise

    if not vk_header.exists():
        raise RuntimeError(f"Vulkan headers extraction finished but {vk_header} is missing.")
    print(f"[vulkan-headers] OK -> {vk_header}")


def verify_toolchain():
    print("=" * 60)
    print("Turbo-WinFare Dense — Toolchain Verification")
    print("=" * 60)
    errors = []

    # 1. PATH setup recommendation check
    w64_bin = W64DEVKIT_DIR / "bin"
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if str(w64_bin) not in path_dirs:
        os.environ["PATH"] = f"{w64_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    # 2. Check g++
    gxx = w64_bin / "g++.exe"
    if not gxx.exists():
        errors.append(f"g++.exe missing at {gxx}")
    else:
        res = subprocess.run([str(gxx), "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"C++ Compiler:      {res.stdout.splitlines()[0]}")
        else:
            errors.append(f"g++ failed to run: {res.stderr}")

    # 3. Check cmake
    cmake = shutil.which("cmake")
    if not cmake and SIBLING_VENV_SCRIPTS is not None and SIBLING_VENV_SCRIPTS.exists():
        candidate = SIBLING_VENV_SCRIPTS / "cmake.exe"
        if candidate.exists():
            cmake = str(candidate)
            os.environ["PATH"] = f"{SIBLING_VENV_SCRIPTS}{os.pathsep}{os.environ.get('PATH', '')}"

    if not cmake:
        errors.append("cmake.exe not found on PATH (see TURBO_SIBLING_ROOT)")
    else:
        res = subprocess.run([cmake, "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"CMake:             {res.stdout.splitlines()[0]}")
        else:
            errors.append(f"cmake failed to run: {res.stderr}")

    # 4. Check ninja
    ninja = shutil.which("ninja")
    if not ninja and SIBLING_VENV_SCRIPTS is not None and SIBLING_VENV_SCRIPTS.exists():
        candidate = SIBLING_VENV_SCRIPTS / "ninja.exe"
        if candidate.exists():
            ninja = str(candidate)

    if not ninja:
        errors.append("ninja.exe not found on PATH (see TURBO_SIBLING_ROOT)")
    else:
        res = subprocess.run([ninja, "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"Ninja:             version {res.stdout.strip()}")
        else:
            errors.append(f"ninja failed to run: {res.stderr}")

    # 5. Check Python
    py_ver = sys.version.replace("\n", " ")
    print(f"Python:            {py_ver}")

    # 6. Check Vulkan loader
    vk_dll = Path("C:/Windows/System32/vulkan-1.dll")
    if not vk_dll.exists():
        errors.append("vulkan-1.dll missing in System32")
    else:
        print(f"Vulkan Loader:     {vk_dll} (present)")

    # 7. Check Vulkan Headers
    vk_h = VULKAN_INCLUDE_DIR / "vulkan" / "vulkan.h"
    if not vk_h.exists():
        errors.append(f"Vulkan headers missing at {vk_h}")
    else:
        print(f"Vulkan Headers:    {vk_h} (present)")

    # 8. Check DXC and test compile probe.hlsl to SPIR-V
    dxc_exe = BUILD_DIR / "dxc.exe"
    if not dxc_exe.exists():
        errors.append(f"dxc.exe missing at {dxc_exe}")
    else:
        res = subprocess.run([str(dxc_exe), "--version"], capture_output=True, text=True)
        dxc_ver = res.stdout.splitlines()[0] if res.stdout else "unknown"
        print(f"DXC Compiler:      {dxc_ver}")

        probe_hlsl = REPO_ROOT / "tests" / "fixtures" / "probe.hlsl"
        probe_spv = BUILD_DIR / "probe.spv"
        if probe_hlsl.exists():
            compile_cmd = [
                str(dxc_exe),
                "-T", "cs_6_6",
                "-spirv",
                "-fspv-target-env=vulkan1.3",
                "-E", "main",
                str(probe_hlsl),
                "-Fo", str(probe_spv)
            ]
            comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
            if comp_res.returncode != 0:
                errors.append(f"DXC SPIR-V compilation failed: {comp_res.stderr}\nCommand: {' '.join(compile_cmd)}")
            elif not probe_spv.exists() or probe_spv.stat().st_size == 0:
                errors.append("DXC SPIR-V compilation produced empty output")
            else:
                print(f"SPIR-V Test Probe: OK -> {probe_spv} ({probe_spv.stat().st_size} bytes)")
        else:
            errors.append(f"Probe shader missing at {probe_hlsl}")

    print("-" * 60)
    if errors:
        print(f"VERIFICATION FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  * {err}")
        return False
    else:
        print("VERIFICATION PASSED: All required tools and dependencies are operational.")
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w64devkit", action="store_true", help="install only w64devkit")
    ap.add_argument("--dxc", action="store_true", help="install only DXC")
    ap.add_argument("--vulkan-headers", action="store_true", help="install only Vulkan-Headers")
    ap.add_argument("--verify", action="store_true", help="verify toolchain readiness")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if args.verify:
        ok = verify_toolchain()
        return 0 if ok else 1

    do_all = not (args.w64devkit or args.dxc or args.vulkan_headers)

    try:
        if do_all or args.w64devkit:
            install_w64devkit(force=args.force)
        if do_all or args.dxc:
            install_dxc(force=args.force)
        if do_all or args.vulkan_headers:
            install_vulkan_headers(force=args.force)

        print("\nRunning verification...")
        if not verify_toolchain():
            return 1
    except Exception as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
