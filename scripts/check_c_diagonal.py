"""Check C (diagonal): Query SDF toward cube corner (1,1,1) to detect OOD oscillations."""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from evaluate import load_model_and_config

device = torch.device('cpu')
model_pe,   lc_pe,   _ = load_model_and_config('experiments/EXP-09/seed42', device)
model_nope, lc_nope, _ = load_model_and_config('experiments/EXP-02/seed42', device)

n = 500
t = np.linspace(0.0, 1.8, n)  # 0=origin, 1.73=cube corner
direction = np.array([1, 1, 1]) / (3 ** 0.5)
pts = np.outer(t, direction).astype(np.float32)
pts_t = torch.from_numpy(pts)

for label, model, lc in [('No-PE', model_nope, lc_nope), ('PE L=6', model_pe, lc_pe)]:
    z = lc(torch.tensor([0])).expand(n, -1)
    with torch.no_grad():
        sdf = model(z, pts_t).squeeze(-1).numpy()

    in_mask  = t <= 1.0
    ood_mask = t > 1.0
    print(f'{label}:')
    print(f'  Inside  r<=1.0: mean={sdf[in_mask].mean():.4f}  std={sdf[in_mask].std():.4f}  '
          f'min={sdf[in_mask].min():.4f}  max={sdf[in_mask].max():.4f}')
    print(f'  Outside r>1.0:  mean={sdf[ood_mask].mean():.4f}  std={sdf[ood_mask].std():.4f}  '
          f'min={sdf[ood_mask].min():.4f}  max={sdf[ood_mask].max():.4f}')
    print(f'  At r=1.73 (corner): SDF={sdf[-1]:.4f}  (expected: large positive ~0.7)')
    sign_changes = int(np.sum(np.diff(np.sign(sdf[ood_mask])) != 0))
    print(f'  Sign changes outside r>1.0: {sign_changes}  (>0 = oscillating)')
    print()
