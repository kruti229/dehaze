#!/usr/bin/env python3
"""
RUN_ALL.PY — Execute all pipeline steps in order
==================================================
Run this file to verify the COMPLETE pipeline from
raw image to dehazed output in one shot.

Usage:
    python run_all.py
"""
import subprocess, sys, os, time

BASE = os.path.dirname(os.path.abspath(__file__))

steps = [
    ("step_00_verify_env.py",        "Environment verification"),
    ("step_01_load_and_normalize.py","Load image & normalize"),
    ("step_02_convnext_encoder.py",  "ConvNext encoder (implicit space)"),
    ("step_03_dcu_cee.py",           "DCU + CEE (latent projection)"),
    ("step_04_fft_decompose.py",     "FFT frequency decomposition ★"),
    ("step_05_fdaa.py",              "FDAA cross-frequency attention ★"),
    ("step_06_to_11_pipeline.py",    "HAG + SDE + DDU + Decoder + Full view ★"),
]

print("=" * 70)
print("AILD-FREQ FULL PIPELINE VERIFICATION")
print("Running all steps in sequence...")
print("=" * 70)

results = []
for script, description in steps:
    script_path = os.path.join(BASE, script)
    print(f"\n{'─'*70}")
    print(f"▶  {description}")
    print(f"   {script}")
    print(f"{'─'*70}")

    t_start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # show output live
        text=True
    )
    elapsed = time.time() - t_start

    status = "✓ PASSED" if result.returncode == 0 else "✗ FAILED"
    results.append((description, status, f"{elapsed:.1f}s"))

    if result.returncode != 0:
        print(f"\n❌ Step failed: {script}")
        print("Stopping pipeline.")
        break

# Summary
print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
all_passed = True
for desc, status, t in results:
    all_passed = all_passed and "PASSED" in status
    icon = "✓" if "PASSED" in status else "✗"
    print(f"  {icon}  {desc:<45} {status}  ({t})")

print("\n" + "=" * 70)
if all_passed:
    print("ALL STEPS PASSED ✓")
    print(f"\nOutput visualizations saved in:")
    print(f"  {os.path.join(BASE, 'outputs/step_outputs/')}")
    print("\nFiles generated:")
    out_dir = os.path.join(BASE, "outputs/step_outputs")
    if os.path.exists(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.endswith('.png'):
                print(f"  📊 {f}")
    print("\nNext step: Start coding the full training pipeline")
    print("  → train_stage1.py  (encoder/decoder pretraining)")
    print("  → train_stage2.py  (diffusion model training)")
else:
    print("PIPELINE FAILED — fix errors above before proceeding")
print("=" * 70)