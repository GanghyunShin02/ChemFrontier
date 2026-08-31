"""
PEMFC flooding 진단 신경망.
학습:  python3 pemfc_ml.py train
진단:  python3 pemfc_ml.py diagnose
효율:  python3 pemfc_ml.py efficiency
"""
import sys, glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/ss/ChemFrontier/ml_dataset"
CKPT = "/home/ss/ChemFrontier/flooding_twin.pt"
FIGDIR = "/home/ss/ChemFrontier"
IN_COLS = ["T_cell", "p_in", "RH_in", "eps_p", "tau_brug", "eta"]
CLASS_NAMES = ["near-single-phase", "light", "moderate", "severe", "near-saturated"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FloodingTwin(nn.Module):
    def __init__(self, n_in, n_hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
        )
        self.head_s = nn.Linear(n_hidden, 1)
        self.head_c = nn.Linear(n_hidden, 5)
        self.head_I = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        s = torch.sigmoid(self.head_s(h)).squeeze(-1)
        return s, self.head_c(h), self.head_I(h).squeeze(-1)


MONO = {IN_COLS.index("T_cell"): -1, IN_COLS.index("RH_in"): +1,
        IN_COLS.index("eta"): +1, IN_COLS.index("tau_brug"): +1}


def monotonicity_penalty(model, x):
    x = x.clone().requires_grad_(True)
    s, _, _ = model(x)
    g = torch.autograd.grad(s.sum(), x, create_graph=True)[0]
    pen = 0.0
    for i, sign in MONO.items():
        pen = pen + torch.relu(-sign * g[:, i]).pow(2).mean()
    return pen


def load_dataset():
    rows_X, rows_s, rows_I = [], [], []
    for fp in sorted(glob.glob(f"{DATA}/case_*.npz")):
        d = np.load(fp)
        if bool(d["failed"]):
            continue
        eta, s_max, I_avg = d["eta"], d["s_max"], d["I_avg"]
        cond = [float(d[f"p_{k}"]) for k in IN_COLS[:-1]]
        for j in range(len(eta)):
            rows_X.append(cond + [float(eta[j])])
            rows_s.append(float(s_max[j]))
            rows_I.append(float(I_avg[j]))
    return (np.array(rows_X, dtype=np.float32),
            np.array(rows_s, dtype=np.float32),
            np.array(rows_I, dtype=np.float32))


def _split(X_raw, y_s, y_c, y_I=None):
    X_mean, X_std = X_raw.mean(0), X_raw.std(0) + 1e-8
    X = (X_raw - X_mean) / X_std
    perm = np.random.permutation(len(X))
    n_tr = int(0.8 * len(X))
    i_tr, i_va = perm[:n_tr], perm[n_tr:]
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    out = dict(X_tr=t(X[i_tr]), X_va=t(X[i_va]),
               s_tr=t(y_s[i_tr]), s_va=t(y_s[i_va]),
               c_tr=t(y_c[i_tr], torch.long), c_va=t(y_c[i_va], torch.long),
               X_mean=X_mean, X_std=X_std)
    if y_I is not None:
        out["I_tr"] = t(y_I[i_tr]); out["I_va"] = t(y_I[i_va])
    return out


def train():
    X_raw, y_s, y_I = load_dataset()
    print(f"총 {len(X_raw)} 샘플")
    y_c = np.digitize(y_s, [0.05, 0.3, 0.6, 0.9]).astype(np.int64)
    print("클래스 분포:", np.bincount(y_c, minlength=5))

    logI = np.log10(np.clip(y_I, 1.0, None)).astype(np.float32)
    logI_mean, logI_std = logI.mean(), logI.std() + 1e-8
    logI_n = (logI - logI_mean) / logI_std

    d = _split(X_raw, y_s, y_c, logI_n)
    X_tr, X_va = d["X_tr"], d["X_va"]
    s_tr, s_va = d["s_tr"], d["s_va"]
    c_tr, c_va = d["c_tr"], d["c_va"]
    I_tr, I_va = d["I_tr"], d["I_va"]
    X_mean, X_std = d["X_mean"], d["X_std"]

    model = FloodingTwin(n_in=len(IN_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4000)
    W_CLS, W_I, W_MONO = 0.3, 0.5, 0.1
    BATCH = min(512, len(X_tr))

    history = {"ep": [], "loss": [], "s_mae": [], "acc": []}

    for ep in range(4000):
        bi = torch.randint(0, len(X_tr), (BATCH,), device=device)
        xb = X_tr[bi]
        s_p, lg_p, I_p = model(xb)
        loss = (nn.functional.mse_loss(s_p, s_tr[bi])
                + W_CLS * nn.functional.cross_entropy(lg_p, c_tr[bi])
                + W_I * nn.functional.mse_loss(I_p, I_tr[bi])
                + W_MONO * monotonicity_penalty(model, xb))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if ep % 400 == 0:
            model.eval()
            with torch.no_grad():
                s_pred_va = model(X_va)[0].cpu().numpy()
            s_true_va = s_va.cpu().numpy()

            mae = np.abs(s_pred_va - s_true_va).mean()
            ss_res = np.sum((s_true_va - s_pred_va) ** 2)
            ss_tot = np.sum((s_true_va - s_true_va.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot

            plt.figure(figsize=(5.5, 5.5))
            plt.scatter(s_true_va, s_pred_va, s=8, alpha=0.4,
                        label=f"validation samples (n={len(s_true_va)})")
            lims = [0, max(s_true_va.max(), s_pred_va.max()) * 1.05]
            plt.plot(lims, lims, "r--", linewidth=1.5, label="y = x (perfect prediction)")
            plt.xlim(lims); plt.ylim(lims)
            plt.xlabel("FEM ground-truth s_max")
            plt.ylabel("Neural network predicted s_max")
            plt.title("Validation set: predicted vs. true s_max")
            plt.legend(loc="upper left")
            plt.text(0.98, 0.05, f"MAE = {mae:.4f}\nR$^2$ = {r2:.4f}",
                    transform=plt.gca().transAxes, ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            plt.tight_layout()
            plt.savefig(f"{FIGDIR}/fig_pred_vs_true.png", dpi=150)
            print("saved ->", f"{FIGDIR}/fig_pred_vs_true.png")

    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean, "X_std": X_std,
                "logI_mean": logI_mean, "logI_std": logI_std}, CKPT)
    print("saved ->", CKPT)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["ep"], history["loss"])
    axes[0].set_title("Total training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[1].plot(history["ep"], history["s_mae"])
    axes[1].set_title("Validation s_max MAE"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MAE")
    axes[2].plot(history["ep"], history["acc"])
    axes[2].set_title("Flooding-grade classification accuracy"); axes[2].set_xlabel("epoch"); axes[2].set_ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_training_curve.png", dpi=150)
    print("saved ->", f"{FIGDIR}/fig_training_curve.png")


def diagnose(T_cell, p_in, RH_in, eps_p, tau_brug, eta):
    ck = torch.load(CKPT, map_location=device)
    model = FloodingTwin(n_in=len(IN_COLS)).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    x_raw = np.array([[T_cell, p_in, RH_in, eps_p, tau_brug, eta]], dtype=np.float32)
    x = torch.tensor((x_raw - ck["X_mean"]) / ck["X_std"], device=device)
    with torch.no_grad():
        s, logits, I_n = model(x)
        prob = torch.softmax(logits, 1)[0]
    I = 10.0 ** (float(I_n) * ck["logI_std"] + ck["logI_mean"])
    return dict(s_max=float(s), grade=CLASS_NAMES[int(prob.argmax())],
                confidence=float(prob.max()), I_A_cm2=I / 1e4)


def data_efficiency():
    X_raw, y_s, _ = load_dataset()
    y_c = np.digitize(y_s, [0.05, 0.3, 0.6, 0.9]).astype(np.int64)
    d = _split(X_raw, y_s, y_c)
    X_tr, X_va = d["X_tr"], d["X_va"]
    s_tr, s_va = d["s_tr"], d["s_va"]
    c_tr, c_va = d["c_tr"], d["c_va"]

    results = {}
    print("=== data efficiency sweep ===")
    for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
        for use_mono in [False, True]:
            sub = torch.randperm(len(X_tr))[:max(64, int(frac * len(X_tr)))]
            m = FloodingTwin(n_in=len(IN_COLS)).to(device)
            o = torch.optim.Adam(m.parameters(), lr=2e-3)
            for _ in range(1500):
                bi = sub[torch.randint(0, len(sub), (min(512, len(sub)),))]
                s_p, lg_p, I_p = m(X_tr[bi])
                l = (nn.functional.mse_loss(s_p, s_tr[bi])
                     + 0.3 * nn.functional.cross_entropy(lg_p, c_tr[bi]))
                if use_mono:
                    l = l + 0.1 * monotonicity_penalty(m, X_tr[bi])
                o.zero_grad(); l.backward(); o.step()
            with torch.no_grad():
                mae = (m(X_va)[0] - s_va).abs().mean().item()
            results[(frac, use_mono)] = mae
            print(f"  train_frac={frac:5.0%}  monotonicity_penalty={use_mono!s:5}  "
                  f"valid_s_MAE={mae:.4f}")

    fracs = [0.05, 0.1, 0.25, 0.5, 1.0]
    mae_no_mono = [results[(f, False)] for f in fracs]
    mae_mono = [results[(f, True)] for f in fracs]

    plt.figure(figsize=(6.5, 5))
    plt.plot(fracs, mae_no_mono, "o-", label="baseline (no physics constraint)")
    plt.plot(fracs, mae_mono, "s-", label="with monotonicity constraint")
    plt.xlabel("fraction of training data used")
    plt.ylabel("validation s_max MAE (lower is better)")
    plt.xscale("log")
    plt.title("Effect of physics-informed monotonicity constraint\non data efficiency")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_data_efficiency.png", dpi=150)
    print("saved ->", f"{FIGDIR}/fig_data_efficiency.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "diagnose":
        print(diagnose(353, 1.5e5, 0.1, 0.5, 1.5, 0.5))
        print(diagnose(303, 1.5e5, 0.9, 0.5, 1.5, 0.5))
    elif cmd == "efficiency":
        data_efficiency()