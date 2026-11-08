import argparse
import h5py
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from models.bilstm import BiLSTMClassifier, ATISDataset


def load_test_from_h5(h5_path, group_name):
    with h5py.File(h5_path, "r") as f:
        grp = f[group_name]

        X_test = grp["X_test"][:]
        y_test = grp["y_test"][:]
        attn_test = grp["attention_mask_test"][:]

        intents = grp["intents"][:] 
        intent_ids = grp["intent_ids"][:]

        # id -> intent name 
        id2intent = {}
        for name_b, iid in zip(intents, intent_ids):
            id2intent[int(iid)] = name_b.decode("utf-8")

        num_classes = len(id2intent)
        intent_names = [id2intent[i] for i in range(num_classes)]

    return X_test, y_test, attn_test, intent_names



# load model and evaluate
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str,
                        default="data/preprocessed_data/atis_preprocessed.h5")
    parser.add_argument("--group_name", type=str,
                        default="len40_lower_noaug")
    parser.add_argument("--model_path", type=str,
                        default="models/BiLSTM_library/bilstm_len40_lower_noaug.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--minority_threshold", type=int, default=20,
                        help="classes with <= this many test examples are treated as minority")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu or cuda; default: auto")

    parser.add_argument("--cm_fig_path", type=str,
                        default="confusion_matrix.png",
                        help="where to save confusion matrix figure")

    args = parser.parse_args()

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Using device:", device)

    # read test data & intent 
    X_test, y_test, attn_test, intent_names = load_test_from_h5(
        args.h5_path, args.group_name
    )
    num_classes = len(intent_names)
    print("Test size:", X_test.shape[0])
    print("Num intents:", num_classes)

    test_dataset = ATISDataset(X_test, y_test, attn_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # load model checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    vocab_size = ckpt["vocab_size"]
    num_intents = ckpt["num_classes"]
    embed_dim = ckpt["embed_dim"]
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]
    pad_id = ckpt["pad_id"]

    print("Loaded checkpoint from:", args.model_path)
    print(f"vocab_size={vocab_size}, num_intents={num_intents}, embed_dim={embed_dim}, hidden_dim={hidden_dim}")

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_intents,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_id=pad_id,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # run in test
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # calculate indicators
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n=== Overall Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    # confusion matrix: only include possible intent id
    unique_labels = np.unique(y_true)
    used_intent_names = [intent_names[i] for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=used_intent_names)
    disp.plot(include_values=False, xticks_rotation=90, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix (Intents)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.cm_fig_path) or ".", exist_ok=True)
    plt.savefig(args.cm_fig_path, dpi=300)
    print(f"Confusion matrix saved to: {args.cm_fig_path}")

    # classification_report: only include possible intent id
    report = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=used_intent_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    print("\n=== Per-class metrics (first few shown) ===")
    for j, lbl in enumerate(unique_labels[:10]):
        name = intent_names[lbl]
        stats = report[name]
        print(
            f"[{lbl:02d}] {name:<25} "
            f"support={int(stats['support']):3d}  "
            f"recall={stats['recall']:.4f}  "
            f"f1={stats['f1-score']:.4f}"
        )

    # minority class: only include possible intent id
    counts = np.bincount(y_true, minlength=num_classes)
    minority_ids = [i for i in unique_labels if counts[i] <= args.minority_threshold]

    print(f"\n=== Minority intents (support <= {args.minority_threshold}) ===")
    if not minority_ids:
        print("No minority classes under this threshold in the test set.")
    else:
        for i in minority_ids:
            name = intent_names[i]
            stats = report[name]
            print(
                f"[{i:02d}] {name:<25} "
                f"support={int(stats['support']):3d}  "
                f"recall={stats['recall']:.4f}  "
                f"f1={stats['f1-score']:.4f}"
            )



if __name__ == "__main__":
    main()
