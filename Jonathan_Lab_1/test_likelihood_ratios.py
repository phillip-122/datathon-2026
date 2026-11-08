import numpy as np

REGIME_RATIOS = {
    'steady': (0.8352732428176273, 3.064865326422233),
    'balanced': (0.2723717096144427, 0.9994126064420332),
    'jackpot': (0.13738046756868819, 0.504089691845762),
}

for mt, (r0, r1) in REGIME_RATIOS.items():
    lr_1 = (r1 / (1 + r1)) / (r0 / (1 + r0))
    lr_high = (1 / (1 + r1)) / (1 / (1 + r0))
    log_lr_1 = np.log(lr_1)
    log_lr_high = np.log(lr_high)
    print(f"Machine Type: {mt:8s}")
    print(f"  If payout == 1: LR(S=1 / S=0) = {lr_1:.4f} (log-LR = {log_lr_1:+.4f})")
    print(f"  If payout >= 2: LR(S=1 / S=0) = {lr_high:.4f} (log-LR = {log_lr_high:+.4f})")
    print(f"  Ratio between the two: {lr_1 / lr_high:.4f} (log diff = {log_lr_1 - log_lr_high:+.4f})")
