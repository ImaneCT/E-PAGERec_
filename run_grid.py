import argparse
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# k-means variants of Table 3 (clustering loss weight 1e-3)
KMEANS = "--cluster_algo kmeans --lambda_proto 1e-3"
MODES = {
    "none": "--comm none",
    "random": f"--comm random {KMEANS}",
    "epagec": "",
}


def phase_runs(phase, modes, fracs):
    runs = []
    if phase == "sparse_seeds":
        # Tables 1 and 2
        for s in (1, 2, 3, 4, 5):
            for frac in fracs:
                for tag in modes:
                    runs.append((f"frac{frac}/{tag}_s{s}",
                                 f"{MODES[tag]} --train_frac {frac} --seed {s}"))
    elif phase == "ablation20":
        # Table 3, 20% of the training data, 5 seeds
        variants = [
            ("kmeans_raw", f"{KMEANS} --pagec_p 0"),     # k-means on embeddings
            ("pagec_kmeans", KMEANS),                    # diffusion + k-means
            ("static_epagec", "--comm static"),          # frozen E-PAGEC
            ("wo_cluloss", "--lambda_proto 0"),          # w/o E-PAGEC objective
            ("wo_conv", "--beta 0"),                     # w/o community convolution
            ("wo_balance", "--lambda_bal 0"),            # w/o balance term
        ]
        for s in (1, 2, 3, 4, 5):
            for tag, flags in variants:
                runs.append((f"{tag}_s{s}", f"{flags} --train_frac 0.2 --seed {s}"))
    elif phase == "sensitivity":
        # number of communities, 20% of the training data, 3 seeds 
        for s in (1, 2, 3):
            for k in (50, 100, 500):
                runs.append((f"K{k}_s{s}", f"--k_user {k} --k_item {k} --train_frac 0.2 --seed {s}"))
    elif phase == "umap":
        # Fig. 3: item embeddings and communities at epochs 3, 7, 15 and at the end
        runs.append(("epagec_frac0.2", "--train_frac 0.2 --seed 1 --save_emb_epochs 3,7,15,final"))
    else:
        raise SystemExit(f"unknown phase {phase}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["sparse_seeds", "ablation20", "sensitivity", "umap"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True, help="dataset short name for the output folder")
    ap.add_argument("--modes", default="none,epagec",
                    help="sparse_seeds only: variants to run (none, random, epagec)")
    ap.add_argument("--fracs", default="1.0,0.2",
                    help="sparse_seeds only: training-data fractions to run")
    ap.add_argument("--extra", default="", help="flags added to every run")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    fracs = [float(f) for f in args.fracs.split(",")]

    for tag, flags in phase_runs(args.phase, modes, fracs):
        out = os.path.join("runs", args.name, args.phase, tag)
        if os.path.exists(os.path.join(out, "result.json")):
            print(f"[skip] {out}")
            continue
        cmd = [sys.executable, os.path.join(HERE, "epagerec.py"), "--data", args.data,
               "--out", out] + shlex.split(f"{flags} {args.extra}")
        print("[run]", " ".join(cmd), flush=True)
        if not args.dry:
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
