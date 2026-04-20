CONFIGS = {
    "M0": {"use_fdaa": False, "use_hag": False, "use_lfreq": False},
    "M1": {"use_fdaa": False, "use_hag": True,  "use_lfreq": True },
    "M2": {"use_fdaa": True,  "use_hag": False, "use_lfreq": True },
    "M3": {"use_fdaa": True,  "use_hag": True,  "use_lfreq": False},
    "M4": {"use_fdaa": True,  "use_hag": True,  "use_lfreq": True },
}
 
for name, flags in CONFIGS.items():
    print(f"\n=== Ablation: {name} ===")
    # Pass flags to train_stage2 to conditionally enable modules
    # Then run test() and log PSNR/SSIM
