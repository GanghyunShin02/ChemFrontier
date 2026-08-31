"""
FEM 파라미터 sweep -> 학습 데이터 생성.
femml.py와 같은 디렉토리에 두고 실행:  python3 gen_data.py
"""
import os, sys, time, json
import numpy as np
from scipy.stats import qmc

# femml.py 안의 함수/상수를 그대로 재사용한다.
# (femml.py 하단의 드라이버 코드가 import 시점에 실행되지 않도록,
#  femml.py의 3640줄 이후 드라이버 부분은 미리 잘라내거나 주석 처리해 두어야 한다)
from femml import DEFAULT_PARAMS, adaptive_polarization_curve

OUT = "/home/ss/ChemFrontier/ml_dataset"
os.makedirs(OUT, exist_ok=True)

# ---- 샘플링할 파라미터와 범위 -------------------------------------------
# DEFAULT_PARAMS의 키 이름과 정확히 일치해야 한다.
SWEEP = {
    "T_cell":   (303.0,  353.0),    # K
    "p_in":     (1.007e5, 2.5e5),   # Pa (p_out은 1.0e5 고정이므로 이게 곧 압력차)
    "RH_in":    (0.0,     0.95),    # 상대습도
    "eps_p":    (0.30,    0.70),    # 공극률
    "tau_brug": (1.1,     2.5),     # Bruggeman 지수
}
KEYS = list(SWEEP)
LO = np.array([SWEEP[k][0] for k in KEYS])
HI = np.array([SWEEP[k][1] for k in KEYS])

N_CASES = 200
ETA_TARGET = 0.8

# Latin Hypercube: 격자보다 훨씬 적은 점으로 고차원 공간을 고르게 덮는다.
# seed 고정 -> 중단 후 재시작해도 같은 파라미터 집합을 재현.
sampler = qmc.LatinHypercube(d=len(KEYS), seed=0)
params = qmc.scale(sampler.random(n=N_CASES), LO, HI)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


for i, p in enumerate(params):
    out_path = f"{OUT}/case_{i:04d}.npz"

    # 이미 끝난 케이스는 건너뛴다 -> 중단 후 재시작 가능.
    if os.path.exists(out_path):
        continue

    ov = {k: float(v) for k, v in zip(KEYS, p)}
    log(f"case {i}/{N_CASES}  " + "  ".join(f"{k}={v:.4g}" for k, v in ov.items()))

    t0 = time.time()
    try:
        eta, I_avg, s_max, iters = adaptive_polarization_curve(
            ov, ETA_TARGET, checkpoint_path=None)
    except Exception as e:
        # 한 케이스가 실패해도 전체 sweep은 계속 진행.
        log(f"  FAILED: {type(e).__name__}: {e}")
        np.savez(out_path, failed=True, **{f"p_{k}": v for k, v in ov.items()})
        continue

    np.savez(out_path, failed=False, eta=eta, I_avg=I_avg,
             s_max=s_max, iters=iters,
             **{f"p_{k}": v for k, v in ov.items()})
    log(f"  done in {(time.time()-t0)/60:.1f} min, "
        f"{len(eta)} pts, eta max {eta.max():.3f}, s_max max {s_max.max():.4f}")

log("ALL DONE")