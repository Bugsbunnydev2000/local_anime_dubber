"""
Small helpers for managing GPU memory across pipeline stages.

Because Whisper-large-v3-turbo, M2M100-418M, and VoxCPM2 are used one at a
time (never simultaneously), we explicitly free CUDA memory between stages.
This is what makes the pipeline fit on an 8GB card: each stage gets close to
the full VRAM budget instead of three models fighting over it at once.

Usage pattern in calling code:
    model = load_something()
    ... use model ...
    del model
    free_gpu_memory()
"""

import gc

import torch


def free_gpu_memory() -> None:
    """Run garbage collection and clear the CUDA cache.

    Call this AFTER deleting your last reference to a model
    (`del model`) so the underlying tensors are actually eligible
    for collection.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_vram(tag: str = "") -> None:
    """Debug helper: print current VRAM usage. Safe to call even on CPU."""
    if not torch.cuda.is_available():
        print(f"[VRAM{' ' + tag if tag else ''}] CUDA not available (running on CPU)")
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[VRAM{' ' + tag if tag else ''}] allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
