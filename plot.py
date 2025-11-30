import os
import json
import argparse
import csv
import matplotlib.pyplot as plt

def read_metrics_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # convert numeric fields
            for k in ["epoch","train_loss","train_acc","val_loss","val_acc","lr","secs"]:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except Exception:
                        pass
            rows.append(r)
    return rows

def read_test_json(run_dir):
    p = os.path.join(run_dir, "test.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True, help="Path to a run dir under ./plot_data")
    ap.add_argument("--out", type=str, default="./plots", help="Directory to save figures")
    args = ap.parse_args()

    metrics_csv = os.path.join(args.run, "metrics.csv")
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"metrics.csv not found under {args.run}")

    os.makedirs(args.out, exist_ok=True)
    rows = read_metrics_csv(metrics_csv)
    test_info = read_test_json(args.run)

    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) for r in rows]
    va_acc = [float(r["val_acc"]) for r in rows]

    plt.figure(figsize=(7,5))
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, va_loss, label="Val Loss")
    if len(va_loss) > 0:
        best_ep = epochs[min(range(len(va_loss)), key=lambda i: va_loss[i])]
        best_val = min(va_loss)
        plt.scatter([best_ep], [best_val], marker="o")
        plt.text(best_ep, best_val, f"best@{best_ep}={best_val:.3f}", fontsize=9)
    # title = f"Loss vs Epoch  |  Test: {test_info.get('test_loss','-'):.3f}/{test_info.get('test_acc','-'):.3f}"
    # plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    out1 = os.path.join(args.out, "loss_curve.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(epochs, tr_acc, label="Train Acc")
    plt.plot(epochs, va_acc, label="Val Acc")
    if len(va_acc) > 0:
        best_ep_acc = epochs[max(range(len(va_acc)), key=lambda i: va_acc[i])]
        best_val_acc = max(va_acc)
        plt.scatter([best_ep_acc], [best_val_acc], marker="o")
        plt.text(best_ep_acc, best_val_acc, f"best@{best_ep_acc}={best_val_acc:.3f}", fontsize=9)
    # title = f"Accuracy vs Epoch  |  Test: acc={test_info.get('test_acc','-'):.3f}"
    # plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    out2 = os.path.join(args.out, "acc_curve.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved plots:\n - {out1}\n - {out2}")

if __name__ == "__main__":
    main()