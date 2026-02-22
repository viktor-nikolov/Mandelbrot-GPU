"""Mandelbrot set browsing desktop application with interactive zoom and pan.

https://github.com/viktor-nikolov/Mandelbrot-GPU

Prerequisites (Windows):
    pip install numpy Pillow mpmath gmpy2 numba cupy-cuda13x pyopencl

Prerequisites (Linux):
    sudo apt install python3-tk python3-numpy python3-pil python3-pil.imagetk python3-mpmath python3-gmpy2
    Or in a virtual environment (tkinter still requires apt):
        sudo apt install python3-tk
        pip install numpy Pillow mpmath gmpy2 numba

cupy-cuda13x is optional and provides GPU-accelerated computation via
CUDA when an NVIDIA GPU is available.
IMPORTANT: cupy-cuda13x also requires CUDA 13 libraries.
On Linux:   pip install nvidia-cuda-runtime nvidia-cuda-nvrtc
On Windows: Install the CUDA 13 Runtime from CUDA 13 Toolkit:
            https://developer.nvidia.com/cuda-downloads

numba is optional and accelerates CPU computation via JIT compilation
when no GPU is available.

gmpy2 is optional and accelerates arbitrary-precision arithmetic for
perturbation theory (deep zoom). When not installed, mpmath is used.

pyopencl is optional and enables GPU computation on Intel integrated GPUs
via OpenCL. If no GPU is found, the app falls back to CPU multiprocessing.
On Linux, Intel OpenCL support also requires: sudo apt install intel-opencl-icd

mpmath provides arbitrary-precision arithmetic for perturbation theory, enabling
sharp rendering at deep zoom levels beyond float64 precision (~10^13).

Note: Intel integrated GPUs typically lack float64 (double precision)
support, so the OpenCL kernels use float32 (single precision). This
comes with a penalty of lower precision -- perturbation calculation
kicks in much earlier (~10^5 zoom vs ~10^13 with CUDA/CPU) and overall
zoom-in capability is more limited compared to the float64 backends.

Command-line options:
    --forcecpu     Disable all GPU acceleration. Forces CPU computation
                   via Numba JIT (if installed) or multiprocessing.
    --forceintel   Disable CUDA but keep Intel OpenCL GPU if available.
    -h, --help     Show help message and exit.

Copyright (c) 2026 Viktor Nikolov
MIT License
"""

import argparse
import math
import os
import threading
from concurrent.futures import ProcessPoolExecutor

_MISSING = []
try:
    import tkinter as tk
except ImportError:
    _MISSING.append(("tkinter", None, "python3-tk"))
try:
    import mpmath
except ImportError:
    _MISSING.append(("mpmath", "mpmath", "python3-mpmath"))
try:
    import numpy as np
except ImportError:
    _MISSING.append(("numpy", "numpy", "python3-numpy"))
try:
    from PIL import Image, ImageTk
except ImportError:
    _MISSING.append(("Pillow", "Pillow", "python3-pil python3-pil.imagetk"))
if _MISSING:
    import sys
    names = ", ".join(n for n, _, _ in _MISSING)
    pip_pkgs = " ".join(p for _, p, _ in _MISSING if p)
    msg = f"Error: missing required package(s): {names}\n"
    if sys.platform == "linux":
        apt_pkgs = " ".join(a for _, _, a in _MISSING)
        msg += (
            f"\n"
            f"Install with:\n"
            f"    sudo apt install {apt_pkgs}\n"
        )
        if pip_pkgs:
            msg += (
                f"\n"
                f"Or in a virtual environment (tkinter still requires apt):\n"
                f"    sudo apt install python3-tk\n"
                f"    pip install {pip_pkgs}\n"
            )
    else:
        if pip_pkgs:
            msg += f"\nInstall with:  pip install {pip_pkgs}\n"
    msg += (
        f"\n"
        f"On a machine with NVIDIA GPU install also CuPy with:\n"
        f"    pip install cupy-cuda13x\n"
        f"\n"
    )
    if sys.platform == "linux":
        msg += (
            f"IMPORTANT: NVIDIA GPU calculation requires CUDA 13 libraries:\n"
            f"           pip install nvidia-cuda-runtime nvidia-cuda-nvrtc\n"
        )
    else:
        msg += (
            f"IMPORTANT: NVIDIA GPU calculation requires CUDA 13 Runtime from CUDA 13 Toolkit:\n"
            f"           https://developer.nvidia.com/cuda-downloads\n"
        )
    msg += (
        f"\n"
        f"On a machine with Intel integrated GPU install also PyOpenCL with:\n"
        f"    pip install pyopencl"
    )
    if sys.platform == "linux":
        msg += (
            f"\n\n"
            f"On Linux, Intel OpenCL support also requires:\n"
            f"    sudo apt install intel-opencl-icd"
        )
    sys.exit(msg)

# GPU acceleration via CuPy CUDA (optional)
import ctypes as _ctypes
import sys as _sys

def _has_nvidia_gpu() -> bool:
    """Detect NVIDIA GPU by loading the CUDA driver library."""
    for name in ("nvcuda.dll", "nvcuda", "libcuda.so", "libcuda.so.1"):
        try:
            _ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False

def _has_intel_opencl_gpu() -> bool:
    """Detect Intel integrated GPU with OpenCL support (Windows only).

    Loads OpenCL.dll via ctypes and queries platforms for an Intel vendor
    that exposes a GPU device.
    """
    if _sys.platform != "win32":
        return False
    try:
        opencl = _ctypes.windll.LoadLibrary("OpenCL.dll")
    except OSError:
        return False

    # OpenCL constants
    CL_PLATFORM_VENDOR = 0x0903
    CL_DEVICE_TYPE_GPU = 0x4
    CL_SUCCESS = 0

    # clGetPlatformIDs(num_entries, *platforms, *num_platforms)
    num_platforms = _ctypes.c_uint32()
    if opencl.clGetPlatformIDs(0, None, _ctypes.byref(num_platforms)) != CL_SUCCESS:
        return False
    if num_platforms.value == 0:
        return False

    platform_ids = (_ctypes.c_void_p * num_platforms.value)()
    if opencl.clGetPlatformIDs(
        num_platforms.value, platform_ids, None,
    ) != CL_SUCCESS:
        return False

    for plat in platform_ids:
        # Re-wrap as c_void_p; iterating the array yields plain ints
        # which overflow on 64-bit Windows without explicit wrapping.
        plat = _ctypes.c_void_p(plat)
        # clGetPlatformInfo(platform, param, size, *value, *size_ret)
        vendor_buf = _ctypes.create_string_buffer(256)
        if opencl.clGetPlatformInfo(
            plat, CL_PLATFORM_VENDOR, 256, vendor_buf, None,
        ) != CL_SUCCESS:
            continue
        vendor = vendor_buf.value.decode("utf-8", errors="ignore").lower()
        if "intel" not in vendor:
            continue

        # Check if this Intel platform has a GPU device
        num_devices = _ctypes.c_uint32()
        ret = opencl.clGetDeviceIDs(
            plat, CL_DEVICE_TYPE_GPU, 0, None, _ctypes.byref(num_devices),
        )
        if ret == CL_SUCCESS and num_devices.value > 0:
            return True

    return False

_HAS_NVIDIA = _has_nvidia_gpu()
_HAS_INTEL_GPU = _has_intel_opencl_gpu()
try:
    if not _HAS_NVIDIA:
        raise ImportError("No NVIDIA GPU detected")
    import cupy as _cp
    _cp.array([0])  # probe that the GPU actually works
    _mandelbrot_kernel = _cp.RawKernel(r'''
    extern "C" __global__
    void mandelbrot(const double* re, const double* im,
                    int* counts, int width, int height, int max_iter) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= width || y >= height) return;
        double cr = re[x];
        double ci = im[y];
        double zr = 0.0, zi = 0.0;
        int i;
        for (i = 0; i < max_iter; i++) {
            double zr2 = zr * zr;
            double zi2 = zi * zi;
            if (zr2 + zi2 > 4.0) break;
            zi = 2.0 * zr * zi + ci;
            zr = zr2 - zi2 + cr;
        }
        counts[y * width + x] = i;
    }
    ''', 'mandelbrot')
    _perturbation_kernel = _cp.RawKernel(r'''
    extern "C" __global__
    void perturbation(const double* ref_re, const double* ref_im,
                      int ref_len,
                      const double* dc_re, const double* dc_im,
                      int* counts, int width, int height, int max_iter) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x >= width || y >= height) return;

        double dcr = dc_re[x];
        double dci = dc_im[y];
        double dr = 0.0, di = 0.0;

        int limit = ref_len < max_iter ? ref_len : max_iter;
        int i;
        for (i = 0; i < limit; i++) {
            double Zr = ref_re[i];
            double Zi = ref_im[i];
            /* delta_{n+1} = 2 * Z_n * delta_n + delta_n^2 + dc */
            double new_dr = 2.0 * (Zr * dr - Zi * di) + dr * dr - di * di + dcr;
            double new_di = 2.0 * (Zr * di + Zi * dr) + 2.0 * dr * di + dci;
            dr = new_dr;
            di = new_di;
            /* escape check: |Z_{n+1} + delta_{n+1}|^2 > 4 */
            if (i + 1 < ref_len) {
                double full_r = ref_re[i + 1] + dr;
                double full_i = ref_im[i + 1] + di;
                if (full_r * full_r + full_i * full_i > 4.0) {
                    counts[y * width + x] = i;
                    return;
                }
            }
        }
        counts[y * width + x] = max_iter;
    }
    ''', 'perturbation')
    # Force kernel compilation to catch NVRTC/driver issues at startup
    _mandelbrot_kernel.compile()
    _HAS_CUDA = True
    _CUPY_MISSING = False
    _CUDA_ERROR = None
except ImportError:
    _HAS_CUDA = False
    _CUPY_MISSING = _HAS_NVIDIA
    _CUDA_ERROR = None
except Exception as _e:
    _HAS_CUDA = False
    _CUPY_MISSING = False
    _CUDA_ERROR = str(_e)

# GPU acceleration via PyOpenCL for Intel integrated GPUs (optional)
_HAS_OPENCL = False
_PYOPENCL_MISSING = False
_OPENCL_ERROR = None
try:
    import pyopencl as _cl

    # Find an Intel GPU device
    _cl_device = None
    for _plat in _cl.get_platforms():
        if "intel" in _plat.vendor.lower():
            try:
                _devs = _plat.get_devices(device_type=_cl.device_type.GPU)
            except _cl.LogicError:
                _devs = []
            if _devs:
                _cl_device = _devs[0]
                break

    if _cl_device is None:
        raise RuntimeError("No Intel GPU device found via PyOpenCL")

    _cl_ctx = _cl.Context(devices=[_cl_device])
    _cl_queue = _cl.CommandQueue(_cl_ctx)

    _CL_MANDELBROT_SRC = r"""
    __kernel void mandelbrot(__global const float* re,
                             __global const float* im,
                             __global int* counts,
                             int width, int height, int max_iter) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x >= width || y >= height) return;
        float cr = re[x];
        float ci = im[y];
        float zr = 0.0f, zi = 0.0f;
        int i;
        for (i = 0; i < max_iter; i++) {
            float zr2 = zr * zr;
            float zi2 = zi * zi;
            if (zr2 + zi2 > 4.0f) break;
            zi = 2.0f * zr * zi + ci;
            zr = zr2 - zi2 + cr;
        }
        counts[y * width + x] = i;
    }
    """

    _CL_PERTURBATION_SRC = r"""
    __kernel void perturbation(__global const float* ref_re,
                               __global const float* ref_im,
                               int ref_len,
                               __global const float* dc_re,
                               __global const float* dc_im,
                               __global int* counts,
                               int width, int height, int max_iter) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        if (x >= width || y >= height) return;

        float dcr = dc_re[x];
        float dci = dc_im[y];
        float dr = 0.0f, di = 0.0f;

        int limit = ref_len < max_iter ? ref_len : max_iter;
        int i;
        for (i = 0; i < limit; i++) {
            float Zr = ref_re[i];
            float Zi = ref_im[i];
            float new_dr = 2.0f * (Zr * dr - Zi * di) + dr * dr - di * di + dcr;
            float new_di = 2.0f * (Zr * di + Zi * dr) + 2.0f * dr * di + dci;
            dr = new_dr;
            di = new_di;
            if (i + 1 < ref_len) {
                float full_r = ref_re[i + 1] + dr;
                float full_i = ref_im[i + 1] + di;
                if (full_r * full_r + full_i * full_i > 4.0f) {
                    counts[y * width + x] = i;
                    return;
                }
            }
        }
        counts[y * width + x] = max_iter;
    }
    """

    _cl_mandelbrot_knl = _cl.Kernel(
        _cl.Program(_cl_ctx, _CL_MANDELBROT_SRC).build(), "mandelbrot")
    _cl_perturbation_knl = _cl.Kernel(
        _cl.Program(_cl_ctx, _CL_PERTURBATION_SRC).build(), "perturbation")
    _HAS_OPENCL = True
except ImportError:
    _PYOPENCL_MISSING = True
except Exception as _e:
    _HAS_OPENCL = False
    _OPENCL_ERROR = str(_e)

# Arbitrary-precision acceleration via gmpy2 (optional, falls back to mpmath)
_HAS_GMPY2 = False
try:
    import gmpy2
    _HAS_GMPY2 = True
except ImportError:
    pass

if _HAS_GMPY2:
    _gmpy2_precision = 53  # module-level; survives cross-thread _set_dps

    def _ensure_precision():
        """Set the current thread's gmpy2 context to the global precision."""
        ctx = gmpy2.get_context()
        if ctx.precision != _gmpy2_precision:
            ctx.precision = _gmpy2_precision

    def _mpf(value):
        _ensure_precision()
        return gmpy2.mpfr(str(value))

    def _mpc(re, im):
        _ensure_precision()
        return gmpy2.mpc(re, im)

    def _set_dps(dps):
        global _gmpy2_precision
        # gmpy2 uses binary precision; convert from decimal digits
        _gmpy2_precision = int(dps * 3.322) + 10
        gmpy2.get_context().precision = _gmpy2_precision

    def _fabs(z):
        _ensure_precision()
        return abs(z)
else:
    def _mpf(value):
        return mpmath.mpf(str(value))

    def _mpc(re, im):
        return mpmath.mpc(re, im)

    def _set_dps(dps):
        mpmath.mp.dps = dps

    def _fabs(z):
        return mpmath.fabs(z)

# CPU acceleration via Numba JIT compilation (optional)
_HAS_NUMBA = False
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    pass

if _HAS_NUMBA:
    @njit(parallel=True)
    def _compute_strip_numba(re, im, max_iter):
        h, w = len(im), len(re)
        counts = np.full((h, w), max_iter, dtype=np.int32)
        for y in prange(h):
            ci = im[y]
            for x in range(w):
                cr = re[x]
                zr = 0.0
                zi = 0.0
                for i in range(max_iter):
                    zr2 = zr * zr
                    zi2 = zi * zi
                    if zr2 + zi2 > 4.0:
                        counts[y, x] = i
                        break
                    zi = 2.0 * zr * zi + ci
                    zr = zr2 - zi2 + cr
        return counts

    @njit(parallel=True)
    def _compute_perturbation_strip_numba(ref_re, ref_im, ref_len, dc_re, dc_im, max_iter):
        h, w = len(dc_im), len(dc_re)
        counts = np.full((h, w), max_iter, dtype=np.int32)
        limit = min(ref_len, max_iter)
        for y in prange(h):
            dci = dc_im[y]
            for x in range(w):
                dcr = dc_re[x]
                dr = 0.0
                di = 0.0
                for i in range(limit):
                    Zr = ref_re[i]
                    Zi = ref_im[i]
                    new_dr = 2.0 * (Zr * dr - Zi * di) + dr * dr - di * di + dcr
                    new_di = 2.0 * (Zr * di + Zi * dr) + 2.0 * dr * di + dci
                    dr = new_dr
                    di = new_di
                    if i + 1 < ref_len:
                        full_r = ref_re[i + 1] + dr
                        full_i = ref_im[i + 1] + di
                        if full_r * full_r + full_i * full_i > 4.0:
                            counts[y, x] = i
                            break
        return counts


def _compute_strip(
    re: np.ndarray,
    im_strip: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Compute iteration counts for a horizontal strip (vectorized).

    Module-level function so it can be pickled for multiprocessing.
    """
    c = re[np.newaxis, :] + 1j * im_strip[:, np.newaxis]
    z = np.zeros_like(c)
    counts = np.full(c.shape, max_iter, dtype=np.int32)
    mask = np.ones(c.shape, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask] * z[mask] + c[mask]
        escaped = mask & (np.abs(z) > 2.0)
        counts[escaped] = i
        mask &= ~escaped
        if not mask.any():
            break

    return counts


def _compute_perturbation_strip(
    ref_re: np.ndarray,
    ref_im: np.ndarray,
    ref_len: int,
    dc_re: np.ndarray,
    dc_im_strip: np.ndarray,
    max_iter: int,
) -> np.ndarray:
    """Compute perturbation iteration counts for a horizontal strip.

    Module-level function so it can be pickled for multiprocessing.
    """
    height, width = len(dc_im_strip), len(dc_re)

    dc = dc_re[np.newaxis, :] + 1j * dc_im_strip[:, np.newaxis]
    delta = np.zeros((height, width), dtype=np.complex128)
    counts = np.full((height, width), max_iter, dtype=np.int32)
    mask = np.ones((height, width), dtype=bool)

    for i in range(ref_len):
        Z = complex(ref_re[i], ref_im[i])
        delta[mask] = 2 * Z * delta[mask] + delta[mask] ** 2 + dc[mask]
        # Escape check uses Z_{i+1} + delta_{i+1} = z_{i+1}
        if i + 1 < ref_len:
            Z_next = complex(ref_re[i + 1], ref_im[i + 1])
            full = Z_next + delta
            escaped = mask & (np.abs(full) > 2.0)
            counts[escaped] = i
            mask &= ~escaped
            if not mask.any():
                break

    return counts


class MandelbrotApp:
    """Desktop application that renders the Mandelbrot set with zoom and pan."""

    # Default complex-plane bounds showing the full Mandelbrot set
    DEFAULT_BOUNDS = (-2.5, 1.0, -1.25, 1.25)
    BASE_MAX_ITER = 256
    ZOOM_FACTOR = 5

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        if _HAS_CUDA:
            self.root.title("Mandelbrot (CUDA)")
        elif _HAS_OPENCL:
            self.root.title("Mandelbrot (Intel GPU)")
        elif _HAS_NUMBA:
            self.root.title("Mandelbrot (Numba JIT)")
        else:
            self.root.title("Mandelbrot (CPU)")

        # Window size: 1/4 screen area, same aspect ratio, centered
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = screen_w // 2
        win_h = screen_h // 2
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # Canvas fills the window
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Complex-plane bounds (float64 for standard mode)
        self.x_min, self.x_max, self.y_min, self.y_max = self.DEFAULT_BOUNDS
        self._startup_view = True  # True until first zoom/pan

        # Arbitrary-precision center for deep zoom perturbation
        self._center_re_mp = _mpf(
            (self.DEFAULT_BOUNDS[0] + self.DEFAULT_BOUNDS[1]) / 2)
        self._center_im_mp = _mpf(
            (self.DEFAULT_BOUNDS[2] + self.DEFAULT_BOUNDS[3]) / 2)
        self._half_re = (self.DEFAULT_BOUNDS[1] - self.DEFAULT_BOUNDS[0]) / 2
        self._half_im = (self.DEFAULT_BOUNDS[3] - self.DEFAULT_BOUNDS[2]) / 2

        # GPU handles computation when available; otherwise use process pool.
        self.num_workers = os.cpu_count() or 1
        if _HAS_CUDA or _HAS_OPENCL or _HAS_NUMBA or self.num_workers <= 1:
            self.executor = None
        else:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)

        # Keep a reference to the displayed image so it isn't garbage-collected
        self._photo_image = None
        self._last_rgb = None  # Last rendered RGB array for pan reuse
        self._last_ref = None              # (ref_re, ref_im, ref_len) tuple
        self._last_ref_center_re_mp = None  # mpmath position of reference point
        self._last_ref_center_im_mp = None

        # Drag state
        self._drag_start = None
        self._drag_offset = (0, 0)
        self._dragging = False
        _DRAG_THRESHOLD = 5  # pixels before press becomes a drag
        self._drag_threshold = _DRAG_THRESHOLD

        # Background computation state
        self._computing = False

        # Resize debounce
        self._resize_after_id = None

        # Help overlay visibility
        self._show_help = True

        # Bind events
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        root.bind("<Key-h>", self._toggle_help)
        root.bind("<Key-H>", self._toggle_help)

        # Build the color palette once
        self._palette = self._build_palette(2048)

    # ------------------------------------------------------------------
    # Color palette
    # ------------------------------------------------------------------

    @staticmethod
    def _build_palette(size: int) -> np.ndarray:
        """Create an HSV-based RGB palette for smooth Mandelbrot coloring."""
        t = np.linspace(0, 1, size, dtype=np.float64)
        # Classic cyclic HSV mapping
        h = np.mod(t * 5.0 + 0.6, 1.0)
        s = np.full_like(t, 0.8)
        v = np.where(t < 1.0, 1.0, 0.0)  # last entry = black (inside set)

        # HSV to RGB conversion (vectorized)
        i = (h * 6.0).astype(int)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        u = v * (1.0 - (1.0 - f) * s)

        i_mod = i % 6
        r = np.choose(i_mod, [v, q, p, p, u, v])
        g = np.choose(i_mod, [u, v, v, q, p, p])
        b = np.choose(i_mod, [p, p, u, v, v, q])

        palette = np.stack([r, g, b], axis=-1)
        return (palette * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Mandelbrot computation
    # ------------------------------------------------------------------

    def _max_iterations(self) -> int:
        """Increase max iterations with zoom level for better detail."""
        default_half = (self.DEFAULT_BOUNDS[1] - self.DEFAULT_BOUNDS[0]) / 2
        zoom_level = default_half / self._half_re
        # Scale: more iterations at deeper zoom
        return int(self.BASE_MAX_ITER + 100 * np.log2(max(zoom_level, 1)))

    def _counts_to_rgb(self, counts: np.ndarray, max_iter: int) -> np.ndarray:
        """Map iteration counts to RGB colors using the palette."""
        palette_size = len(self._palette)
        indices = np.mod(counts, palette_size).astype(int)
        inside = counts == max_iter
        rgb = self._palette[indices]
        rgb[inside] = 0
        return rgb

    def _needs_perturbation(self) -> bool:
        """Return True if pixel spacing is too small for direct iteration.

        float64 loses precision around 1e-13; float32 (Intel GPU OpenCL)
        around 1e-5, so we switch to perturbation earlier in that case.
        """
        width = self.canvas.winfo_width()
        if width < 2:
            return False
        pixel_size = 2 * self._half_re / (width - 1)
        # OpenCL kernels use float32 (Intel iGPUs lack float64 support),
        # which loses precision around 1e-5; CUDA and CPU use float64 (~1e-13).
        threshold = 1e-5 if (_HAS_OPENCL and not _HAS_CUDA) else 1e-13
        return pixel_size < threshold

    def _compute_region(
        self, re: np.ndarray, im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute Mandelbrot RGB for an arbitrary rectangular region (standard mode)."""
        if len(re) == 0 or len(im) == 0:
            return np.zeros((len(im), len(re), 3), dtype=np.uint8)

        if _HAS_CUDA:
            counts = self._compute_gpu(re, im, max_iter)
        elif _HAS_OPENCL:
            counts = self._compute_opencl(re, im, max_iter)
        elif _HAS_NUMBA:
            counts = _compute_strip_numba(re, im, max_iter)
        elif self.executor is None:
            counts = _compute_strip(re, im, max_iter)
        else:
            n = min(self.num_workers * 4, max(1, len(im)))
            strips = np.array_split(im, n)
            futures = [
                self.executor.submit(_compute_strip, re, s, max_iter)
                for s in strips
            ]
            counts = np.vstack([f.result() for f in futures])
        return self._counts_to_rgb(counts, max_iter)

    @staticmethod
    def _compute_gpu(
        re: np.ndarray, im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute iteration counts on the GPU via CuPy CUDA kernel."""
        height, width = len(im), len(re)

        d_re = _cp.asarray(re, dtype=_cp.float64)
        d_im = _cp.asarray(im, dtype=_cp.float64)
        d_counts = _cp.empty((height, width), dtype=_cp.int32)

        block = (32, 8)  # x=32 aligns with warp size for memory coalescing
        grid = ((width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1])

        _mandelbrot_kernel(
            grid, block,
            (d_re, d_im, d_counts, np.int32(width), np.int32(height),
             np.int32(max_iter)),
        )
        return _cp.asnumpy(d_counts)

    @staticmethod
    def _compute_opencl(
        re: np.ndarray, im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute iteration counts on Intel GPU via OpenCL kernel."""
        height, width = len(im), len(re)
        mf = _cl.mem_flags

        re_buf = _cl.Buffer(_cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(re, dtype=np.float32))
        im_buf = _cl.Buffer(_cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=np.ascontiguousarray(im, dtype=np.float32))
        counts = np.empty((height, width), dtype=np.int32)
        counts_buf = _cl.Buffer(_cl_ctx, mf.WRITE_ONLY, counts.nbytes)

        # Round global_size up to multiple of 16 (local_size)
        gw = ((width + 15) // 16) * 16
        gh = ((height + 15) // 16) * 16

        _cl_mandelbrot_knl.set_args(
            re_buf, im_buf, counts_buf,
            np.int32(width), np.int32(height), np.int32(max_iter),
        )
        _cl.enqueue_nd_range_kernel(_cl_queue, _cl_mandelbrot_knl,
                                    (gw, gh), (16, 16))
        _cl.enqueue_copy(_cl_queue, counts, counts_buf).wait()
        return counts

    # ------------------------------------------------------------------
    # Perturbation theory (deep zoom beyond float64 precision)
    # ------------------------------------------------------------------

    @staticmethod
    def _iterate_orbit(
        c_re_mp: mpmath.mpf, c_im_mp: mpmath.mpf, max_iter: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Iterate a single orbit at arbitrary precision.

        Returns (ref_re, ref_im, orbit_len).
        """
        C = _mpc(c_re_mp, c_im_mp)
        Z = _mpc(0, 0)

        ref_re = np.empty(max_iter, dtype=np.float64)
        ref_im = np.empty(max_iter, dtype=np.float64)
        orbit_len = max_iter

        for i in range(max_iter):
            ref_re[i] = float(Z.real)
            ref_im[i] = float(Z.imag)
            Z = Z * Z + C
            if float(_fabs(Z)) > 1e6:
                orbit_len = i + 1
                break

        return ref_re[:orbit_len], ref_im[:orbit_len], orbit_len

    @staticmethod
    def _compute_reference_orbit(
        center_re_mp: mpmath.mpf,
        center_im_mp: mpmath.mpf,
        max_iter: int,
        pixel_spacing: float,
        half_re: float = 0.0,
        half_im: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, int, float, float]:
        """Compute a reference orbit at arbitrary precision using mpmath.

        If the center escapes before max_iter, candidate points across the
        view are tested to find a longer-lived (ideally non-escaping) reference.

        Returns (ref_re, ref_im, orbit_len, ref_offset_re, ref_offset_im)
        where ref_re/ref_im are float64 arrays of Z_0..Z_N and
        ref_offset_re/ref_offset_im are the float64 offset of the chosen
        reference point from center (0.0 when center itself is used).
        """
        dps = max(50, int(-math.log10(pixel_spacing)) + 20)
        _set_dps(dps)

        ref_re, ref_im, orbit_len = MandelbrotApp._iterate_orbit(
            center_re_mp, center_im_mp, max_iter,
        )
        if orbit_len == max_iter:
            return ref_re, ref_im, orbit_len, 0.0, 0.0

        # Center escaped — search for a better reference across the view.
        best = (ref_re, ref_im, orbit_len, 0.0, 0.0)
        hr = _mpf(half_re) if half_re else _mpf(pixel_spacing * 100)
        hi = _mpf(half_im) if half_im else hr
        # Try points at ~25% and 50% of the view span in each direction
        offsets = []
        for frac in (_mpf('0.25'), _mpf('0.5')):
            dr, di = hr * frac, hi * frac
            offsets.extend([
                (dr, 0), (-dr, 0), (0, di), (0, -di),
                (dr, di), (-dr, di), (dr, -di), (-dr, -di),
            ])
        for dre, dim in offsets:
            r, i, length = MandelbrotApp._iterate_orbit(
                center_re_mp + dre, center_im_mp + dim, max_iter,
            )
            if length > best[2]:
                best = (r, i, length, float(dre), float(dim))
                if length == max_iter:
                    break
        return best

    @staticmethod
    def _perturbation_counts_gpu(
        ref_re: np.ndarray, ref_im: np.ndarray, ref_len: int,
        dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Launch the perturbation CUDA kernel with precomputed reference orbit."""
        height, width = len(dc_im), len(dc_re)

        d_ref_re = _cp.asarray(ref_re, dtype=_cp.float64)
        d_ref_im = _cp.asarray(ref_im, dtype=_cp.float64)
        d_dc_re = _cp.asarray(dc_re, dtype=_cp.float64)
        d_dc_im = _cp.asarray(dc_im, dtype=_cp.float64)
        d_counts = _cp.empty((height, width), dtype=_cp.int32)

        block = (32, 8)  # x=32 aligns with warp size for memory coalescing
        grid = ((width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1])

        _perturbation_kernel(
            grid, block,
            (d_ref_re, d_ref_im, np.int32(ref_len),
             d_dc_re, d_dc_im,
             d_counts, np.int32(width), np.int32(height),
             np.int32(max_iter)),
        )
        return _cp.asnumpy(d_counts)

    @staticmethod
    def _perturbation_counts_opencl(
        ref_re: np.ndarray, ref_im: np.ndarray, ref_len: int,
        dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Launch the perturbation OpenCL kernel with precomputed reference orbit."""
        height, width = len(dc_im), len(dc_re)
        mf = _cl.mem_flags

        ref_re_buf = _cl.Buffer(_cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=np.ascontiguousarray(ref_re, dtype=np.float32))
        ref_im_buf = _cl.Buffer(_cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                hostbuf=np.ascontiguousarray(ref_im, dtype=np.float32))
        dc_re_buf = _cl.Buffer(_cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=np.ascontiguousarray(dc_re, dtype=np.float32))
        dc_im_buf = _cl.Buffer(_cl_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=np.ascontiguousarray(dc_im, dtype=np.float32))
        counts = np.empty((height, width), dtype=np.int32)
        counts_buf = _cl.Buffer(_cl_ctx, mf.WRITE_ONLY, counts.nbytes)

        gw = ((width + 15) // 16) * 16
        gh = ((height + 15) // 16) * 16

        _cl_perturbation_knl.set_args(
            ref_re_buf, ref_im_buf, np.int32(ref_len),
            dc_re_buf, dc_im_buf, counts_buf,
            np.int32(width), np.int32(height), np.int32(max_iter),
        )
        _cl.enqueue_nd_range_kernel(_cl_queue, _cl_perturbation_knl,
                                    (gw, gh), (16, 16))
        _cl.enqueue_copy(_cl_queue, counts, counts_buf).wait()
        return counts

    def _compute_perturbation_opencl(
        self, dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute iteration counts via perturbation theory on Intel GPU (OpenCL).

        dc_re / dc_im are pixel offsets from the arbitrary-precision center,
        NOT absolute coordinates.
        """
        height, width = len(dc_im), len(dc_re)
        pixel_spacing = (dc_re[-1] - dc_re[0]) / max(width - 1, 1)

        ref_re, ref_im, ref_len, ref_off_re, ref_off_im = (
            self._compute_reference_orbit(
                self._center_re_mp, self._center_im_mp, max_iter,
                pixel_spacing, self._half_re, self._half_im,
            ))

        if ref_off_re != 0.0 or ref_off_im != 0.0:
            dc_re = dc_re - ref_off_re
            dc_im = dc_im - ref_off_im

        self._last_ref = (ref_re, ref_im, ref_len)
        self._last_ref_center_re_mp = self._center_re_mp + _mpf(ref_off_re)
        self._last_ref_center_im_mp = self._center_im_mp + _mpf(ref_off_im)

        return self._perturbation_counts_opencl(
            ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
        )

    def _compute_perturbation_gpu(
        self, dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute iteration counts via perturbation theory on GPU.

        dc_re / dc_im are pixel offsets from the arbitrary-precision center,
        NOT absolute coordinates.
        """
        height, width = len(dc_im), len(dc_re)
        pixel_spacing = (dc_re[-1] - dc_re[0]) / max(width - 1, 1)

        ref_re, ref_im, ref_len, ref_off_re, ref_off_im = (
            self._compute_reference_orbit(
                self._center_re_mp, self._center_im_mp, max_iter,
                pixel_spacing, self._half_re, self._half_im,
            ))

        # Adjust dc offsets: perturbation needs dc = c - C_ref, but dc arrays
        # are relative to the view center.  When the reference is off-center,
        # subtract its offset so each pixel's dc is relative to the reference.
        if ref_off_re != 0.0 or ref_off_im != 0.0:
            dc_re = dc_re - ref_off_re
            dc_im = dc_im - ref_off_im

        # Store reference orbit for pan reuse
        self._last_ref = (ref_re, ref_im, ref_len)
        self._last_ref_center_re_mp = self._center_re_mp + _mpf(ref_off_re)
        self._last_ref_center_im_mp = self._center_im_mp + _mpf(ref_off_im)

        return self._perturbation_counts_gpu(
            ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
        )

    def _perturbation_counts_cpu(
        self, ref_re: np.ndarray, ref_im: np.ndarray, ref_len: int,
        dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Run the perturbation iteration on CPU, using multiprocessing when available."""
        if _HAS_NUMBA:
            return _compute_perturbation_strip_numba(
                ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
            )
        if self.executor is None:
            return _compute_perturbation_strip(
                ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
            )
        n = min(self.num_workers * 4, max(1, len(dc_im)))
        strips = np.array_split(dc_im, n)
        futures = [
            self.executor.submit(
                _compute_perturbation_strip,
                ref_re, ref_im, ref_len, dc_re, s, max_iter,
            )
            for s in strips
        ]
        return np.vstack([f.result() for f in futures])

    def _compute_perturbation_cpu(
        self, dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute iteration counts via perturbation theory on CPU (NumPy).

        dc_re / dc_im are pixel offsets from the arbitrary-precision center,
        NOT absolute coordinates.
        """
        height, width = len(dc_im), len(dc_re)
        pixel_spacing = (dc_re[-1] - dc_re[0]) / max(width - 1, 1)

        ref_re, ref_im, ref_len, ref_off_re, ref_off_im = (
            self._compute_reference_orbit(
                self._center_re_mp, self._center_im_mp, max_iter,
                pixel_spacing, self._half_re, self._half_im,
            ))

        # Adjust dc offsets for off-center reference (see GPU method comment)
        if ref_off_re != 0.0 or ref_off_im != 0.0:
            dc_re = dc_re - ref_off_re
            dc_im = dc_im - ref_off_im

        # Store reference orbit for pan reuse
        self._last_ref = (ref_re, ref_im, ref_len)
        self._last_ref_center_re_mp = self._center_re_mp + _mpf(ref_off_re)
        self._last_ref_center_im_mp = self._center_im_mp + _mpf(ref_off_im)

        return self._perturbation_counts_cpu(
            ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
        )

    def _compute_perturbation_region(
        self, dc_re: np.ndarray, dc_im: np.ndarray, max_iter: int,
    ) -> np.ndarray:
        """Compute perturbation RGB for a region using the stored reference orbit.

        dc_re / dc_im must already be relative to the stored reference point.
        """
        if len(dc_re) == 0 or len(dc_im) == 0:
            return np.zeros((len(dc_im), len(dc_re), 3), dtype=np.uint8)

        ref_re, ref_im, ref_len = self._last_ref
        if _HAS_CUDA:
            counts = self._perturbation_counts_gpu(
                ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
            )
        elif _HAS_OPENCL:
            counts = self._perturbation_counts_opencl(
                ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
            )
        else:
            counts = self._perturbation_counts_cpu(
                ref_re, ref_im, ref_len, dc_re, dc_im, max_iter,
            )
        return self._counts_to_rgb(counts, max_iter)

    def compute_mandelbrot(self, width: int, height: int) -> np.ndarray:
        """Compute the full Mandelbrot image, using processes when available."""
        max_iter = self._max_iterations()

        if self._needs_perturbation():
            # Generate dc offsets from half-spans (float64 is fine for offsets)
            dc_re = np.linspace(-self._half_re, self._half_re, width)
            dc_im = np.linspace(-self._half_im, self._half_im, height)
            if _HAS_CUDA:
                counts = self._compute_perturbation_gpu(dc_re, dc_im, max_iter)
            elif _HAS_OPENCL:
                counts = self._compute_perturbation_opencl(dc_re, dc_im, max_iter)
            else:
                counts = self._compute_perturbation_cpu(dc_re, dc_im, max_iter)
            return self._counts_to_rgb(counts, max_iter)

        re = np.linspace(self.x_min, self.x_max, width)
        im = np.linspace(self.y_min, self.y_max, height)
        return self._compute_region(re, im, max_iter)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _sync_float64_bounds(self) -> None:
        """Derive float64 bounds from arbitrary-precision center + half-spans."""
        center_re = float(self._center_re_mp)
        center_im = float(self._center_im_mp)
        self.x_min = center_re - self._half_re
        self.x_max = center_re + self._half_re
        self.y_min = center_im - self._half_im
        self.y_max = center_im + self._half_im

    def _adjust_aspect(self, width: int, height: int) -> None:
        """Adjust complex-plane bounds to match window aspect ratio.

        Preserves the view center and expands one axis as needed so the
        Mandelbrot set is never distorted.
        """
        window_aspect = width / height
        region_aspect = self._half_re / self._half_im

        if window_aspect > region_aspect:
            self._half_re = self._half_im * window_aspect
        else:
            self._half_im = self._half_re / window_aspect

        self._sync_float64_bounds()

    def _show_calculating(self, perturbation: bool = False) -> None:
        """Show CALCULATING label at top center of the canvas."""
        cx = self.canvas.winfo_width() // 2
        pad_x, pad_y = 8, 4
        tag = "calc_label"
        label = "CALCULATING PERTURBATION" if perturbation else "CALCULATING"
        text_id = self.canvas.create_text(
            cx, 10 + pad_y, anchor=tk.N, text=label,
            fill="red", font=("Consolas", 14, "bold"), tags=tag,
        )
        bbox = self.canvas.bbox(text_id)
        if bbox:
            self.canvas.create_rectangle(
                bbox[0] - pad_x, bbox[1] - pad_y,
                bbox[2] + pad_x, bbox[3] + pad_y,
                fill="yellow", outline="red", width=2, tags=tag,
            )
            self.canvas.tag_raise(text_id)
        self.canvas.update_idletasks()

    def _toggle_help(self, event=None) -> None:
        """Toggle visibility of the help overlay."""
        self._show_help = not self._show_help
        if self._last_rgb is not None:
            self._display_image(self._last_rgb)

    def _display_image(self, rgb: np.ndarray) -> None:
        """Display an RGB array on the canvas with help overlay."""
        self._last_rgb = rgb
        img = Image.fromarray(rgb, "RGB")
        self._photo_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image)
        if self._show_help:
            self.canvas.create_text(
                8, 8, anchor=tk.NW,
                text="Click: zoom in 5x (with Shift: 2.5x)\nRight click: zoom out 5x (with Shift: 2.5x)\nDrag: pan\nH: toggle this help",
                fill="white", font=("Consolas", 10),
            )

    def render(self) -> None:
        """Compute and display the full Mandelbrot set."""
        self._resize_after_id = None
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 2 or height < 2:
            return

        self._adjust_aspect(width, height)
        self._show_calculating(perturbation=self._needs_perturbation())
        self._computing = True
        thread = threading.Thread(
            target=self._run_render, args=(width, height), daemon=True,
        )
        thread.start()
        self._poll_thread(thread)

    def _run_render(self, width: int, height: int) -> None:
        """Background thread target for full render."""
        try:
            self._pending_rgb = self.compute_mandelbrot(width, height)
        except Exception:
            self._pending_rgb = None

    def _poll_thread(self, thread: threading.Thread) -> None:
        """Poll until the background computation thread finishes."""
        if thread.is_alive():
            self.root.after(50, self._poll_thread, thread)
            return
        self._computing = False
        if self._pending_rgb is not None:
            self._display_image(self._pending_rgb)
            self._pending_rgb = None

    def _render_pan(self, dx: int, dy: int) -> None:
        """Optimized render after pan: reuse shifted pixels, compute only exposed strips."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        use_perturbation = self._needs_perturbation()

        # Fall back to full render when we can't reuse pixels.
        if (self._last_rgb is None
                or self._last_rgb.shape[0] != height
                or self._last_rgb.shape[1] != width
                or abs(dx) >= width or abs(dy) >= height
                or (use_perturbation and self._last_ref is None)):
            self.render()
            return

        self._show_calculating(perturbation=use_perturbation)
        self._computing = True
        thread = threading.Thread(
            target=self._run_render_pan,
            args=(dx, dy, width, height, use_perturbation),
            daemon=True,
        )
        thread.start()
        self._poll_thread(thread)

    def _run_render_pan(
        self, dx: int, dy: int, width: int, height: int,
        use_perturbation: bool,
    ) -> None:
        """Background thread target for pan render."""
        try:
            self._run_render_pan_inner(dx, dy, width, height, use_perturbation)
        except Exception:
            self._pending_rgb = None

    def _run_render_pan_inner(
        self, dx: int, dy: int, width: int, height: int,
        use_perturbation: bool,
    ) -> None:
        max_iter = self._max_iterations()

        rgb = np.zeros((height, width, 3), dtype=np.uint8)

        # Copy shifted old pixels into the new array
        sx0, sx1 = max(0, -dx), min(width, width - dx)
        sy0, sy1 = max(0, -dy), min(height, height - dy)
        tx0, tx1 = max(0, dx), min(width, width + dx)
        ty0, ty1 = max(0, dy), min(height, height + dy)
        rgb[ty0:ty1, tx0:tx1] = self._last_rgb[sy0:sy1, sx0:sx1]

        if use_perturbation:
            # Build full-view dc arrays relative to the stored reference point
            shift_re = float(self._center_re_mp - self._last_ref_center_re_mp)
            shift_im = float(self._center_im_mp - self._last_ref_center_im_mp)
            dc_re = np.linspace(-self._half_re, self._half_re, width) + shift_re
            dc_im = np.linspace(-self._half_im, self._half_im, height) + shift_im

            # Compute exposed horizontal strip (full width)
            if dy > 0:
                rgb[:dy, :] = self._compute_perturbation_region(
                    dc_re, dc_im[:dy], max_iter,
                )
            elif dy < 0:
                rgb[height + dy:, :] = self._compute_perturbation_region(
                    dc_re, dc_im[height + dy:], max_iter,
                )

            # Compute exposed vertical strip (excluding rows already computed above)
            vy0 = max(0, dy)
            vy1 = height + min(0, dy)
            if dx > 0 and vy1 > vy0:
                rgb[vy0:vy1, :dx] = self._compute_perturbation_region(
                    dc_re[:dx], dc_im[vy0:vy1], max_iter,
                )
            elif dx < 0 and vy1 > vy0:
                rgb[vy0:vy1, width + dx:] = self._compute_perturbation_region(
                    dc_re[width + dx:], dc_im[vy0:vy1], max_iter,
                )
        else:
            re = np.linspace(self.x_min, self.x_max, width)
            im = np.linspace(self.y_min, self.y_max, height)

            # Compute exposed horizontal strip (full width)
            if dy > 0:
                rgb[:dy, :] = self._compute_region(re, im[:dy], max_iter)
            elif dy < 0:
                rgb[height + dy:, :] = self._compute_region(
                    re, im[height + dy:], max_iter,
                )

            # Compute exposed vertical strip (excluding rows already computed above)
            vy0 = max(0, dy)
            vy1 = height + min(0, dy)
            if dx > 0 and vy1 > vy0:
                rgb[vy0:vy1, :dx] = self._compute_region(
                    re[:dx], im[vy0:vy1], max_iter,
                )
            elif dx < 0 and vy1 > vy0:
                rgb[vy0:vy1, width + dx:] = self._compute_region(
                    re[width + dx:], im[vy0:vy1], max_iter,
                )

        self._pending_rgb = rgb

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_configure(self, event: tk.Event) -> None:
        """Handle window/canvas resize with debounce."""
        if self._computing:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(200, self.render)

    def _zoom(self, event: tk.Event, factor: float) -> None:
        """Zoom in or out centered on the mouse position."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width < 2 or height < 2:
            return

        # Click offset from view center (float64 is fine for the offset)
        frac_x = event.x / width - 0.5   # -0.5 .. 0.5
        frac_y = event.y / height - 0.5
        click_offset_re = frac_x * 2 * self._half_re
        click_offset_im = frac_y * 2 * self._half_im

        # New center = old center + click offset (arbitrary precision)
        self._center_re_mp += _mpf(click_offset_re)
        self._center_im_mp += _mpf(click_offset_im)

        # Scale half-spans
        self._half_re /= factor
        self._half_im /= factor

        self._sync_float64_bounds()
        self._startup_view = False
        self._last_ref = None  # Invalidate stored reference orbit on zoom

        self.render()

    def _on_left_press(self, event: tk.Event) -> None:
        if self._computing:
            return
        self._drag_start = (event.x, event.y)
        self._drag_offset = (0, 0)
        self._dragging = False

    def _on_left_motion(self, event: tk.Event) -> None:
        if self._computing or self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        if not self._dragging:
            if abs(dx) < self._drag_threshold and abs(dy) < self._drag_threshold:
                return
            self._dragging = True
        self._drag_offset = (dx, dy)
        # Move the existing image visually for immediate feedback
        self.canvas.delete("all")
        if self._photo_image:
            self.canvas.create_image(dx, dy, anchor=tk.NW, image=self._photo_image)

    def _on_left_release(self, event: tk.Event) -> None:
        if self._computing or self._drag_start is None:
            return
        # If a resize is pending (debounce timer active), this click is a
        # side effect of a title-bar double-click or other window state
        # change — ignore it to avoid an unintended zoom/pan.
        if self._resize_after_id is not None:
            self._drag_start = None
            return
        if not self._dragging:
            # No significant movement — treat as a click (zoom in)
            self._drag_start = None
            factor = self.ZOOM_FACTOR / 2 if event.state & 0x1 else self.ZOOM_FACTOR
            self._zoom(event, factor)
            return
        # Finish pan
        dx, dy = self._drag_offset
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        # Pan shift in complex-plane units
        shift_re = -dx / width * 2 * self._half_re
        shift_im = -dy / height * 2 * self._half_im

        # Update arbitrary-precision center
        self._center_re_mp += _mpf(shift_re)
        self._center_im_mp += _mpf(shift_im)
        self._sync_float64_bounds()

        self._drag_start = None
        self._drag_offset = (0, 0)
        self._dragging = False
        self._startup_view = False
        self._render_pan(dx, dy)

    def _on_right_click(self, event: tk.Event) -> None:
        """Zoom out on right click (5x, or 2.5x with Shift)."""
        if self._computing:
            return
        factor = self.ZOOM_FACTOR / 2 if event.state & 0x1 else self.ZOOM_FACTOR
        self._zoom(event, 1.0 / factor)


def main() -> None:
    global _HAS_CUDA, _HAS_OPENCL
    parser = argparse.ArgumentParser(
        description="Mandelbrot set explorer with interactive zoom and pan.",
    )
    parser.add_argument(
        "--forcecpu", action="store_true",
        help="Disable all GPU acceleration. Forces CPU computation via Numba JIT (if installed) or multiprocessing.",
    )
    parser.add_argument(
        "--forceintel", action="store_true",
        help="Disable CUDA but keep Intel OpenCL GPU if available.",
    )
    args = parser.parse_args()
    if args.forcecpu:
        _HAS_CUDA = False
        _HAS_OPENCL = False
    elif args.forceintel:
        _HAS_CUDA = False
        if not _HAS_OPENCL:
            if _PYOPENCL_MISSING:
                msg = (
                    "NOTE: --forceintel was specified but PyOpenCL is not installed.\n"
                    "      The app will fall back to CPU computation.\n"
                    "      Install PyOpenCL:  pip install pyopencl"
                )
                if _sys.platform == "linux":
                    msg += "\n      On Linux also:     sudo apt install intel-opencl-icd"
                print(msg)
            elif _OPENCL_ERROR:
                msg = (
                    f"NOTE: --forceintel was specified but OpenCL failed to initialize:\n"
                    f"      {_OPENCL_ERROR}\n"
                    f"      The app will fall back to CPU computation."
                )
                if _sys.platform == "linux":
                    msg += "\n      On Linux, ensure Intel OpenCL is installed:  sudo apt install intel-opencl-icd"
                print(msg)
            else:
                print(
                    "NOTE: --forceintel was specified but no Intel GPU device was found via OpenCL.\n"
                    "      The app will fall back to CPU computation."
                )

    if _CUDA_ERROR:
        msg = (
            f"NOTE: NVIDIA GPU detected but CUDA failed to initialize:\n"
            f"      {_CUDA_ERROR}\n"
            f"      \n"
            f"      The app will fall back to Intel GPU or CPU computation.\n"
        )
        if _sys.platform == "linux":
            msg += (
                f"      Install CUDA 13 libraries:\n"
                f"          pip install nvidia-cuda-runtime nvidia-cuda-nvrtc"
            )
        else:
            msg += (
                f"      Install the CUDA 13 Runtime from CUDA 13 Toolkit:\n"
                f"          https://developer.nvidia.com/cuda-downloads"
            )
        print(msg)
    elif _CUPY_MISSING:
        msg = (
            "NOTE: NVIDIA GPU detected but CuPy is not installed.\n"
            "      The app will fall back to Intel GPU or CPU computation.\n"
            "      For NVIDIA GPU acceleration, install CuPy:\n"
            "          pip install cupy-cuda13x\n"
            "\n"
        )
        if _sys.platform == "linux":
            msg += (
                "IMPORTANT: NVIDIA GPU calculation requires CUDA 13 libraries:\n"
                "           pip install nvidia-cuda-runtime nvidia-cuda-nvrtc"
            )
        else:
            msg += (
                "IMPORTANT: NVIDIA GPU calculation requires CUDA 13 Runtime from CUDA 13 Toolkit:\n"
                "           https://developer.nvidia.com/cuda-downloads"
            )
        print(msg)

    if _PYOPENCL_MISSING:
        msg = (
            "NOTE: Intel integrated GPU detected but PyOpenCL is not installed.\n"
            "      The app will fall back to CPU computation.\n"
            "      For Intel GPU acceleration, install PyOpenCL:\n"
            "          pip install pyopencl"
        )
        if _sys.platform == "linux":
            msg += (
                "\n      On Linux also:\n"
                "          sudo apt install intel-opencl-icd"
            )
        print(msg)

    if not _HAS_CUDA and not _HAS_OPENCL and not _HAS_NUMBA:
        print("TIP: Install Numba for faster CPU computation:\n"
              "         pip install numba")

    if not _HAS_GMPY2:
        msg = ("TIP: Install gmpy2 for faster deep-zoom rendering:\n"
               "         sudo apt install python3-gmpy2"
               if _sys.platform == "linux" else
               "TIP: Install gmpy2 for faster deep-zoom rendering:\n"
               "         pip install gmpy2")
        msg += ("\n     gmpy2 accelerates the arbitrary-precision reference orbit\n"
                "     computation used by perturbation theory at deep zoom levels.")
        print(msg)

    # Enable DPI awareness on Windows so canvas pixels match monitor pixels
    if _sys.platform == "win32":
        try:
            _ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    app = MandelbrotApp(root)
    root.mainloop()
    if app.executor is not None:
        app.executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
