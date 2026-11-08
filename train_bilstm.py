import os
import argparse
import h5py
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.bilstm import BiLSTMClassifier, ATISDataset
import matplotlib.pyplot as plt

# load data
def load_h5_group(h5_path: str, group_name: str):
    """
    return:
        X_train, y_train, X_test, y_test,
        attn_train, attn_test,
        vocab_size, num_intents, pad_id
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[group_name]

        X_train = grp["X_train"][:]
        y_train = grp["y_train"][:]
        X_test = grp["X_test"][:]
        y_test = grp["y_test"][:]

        attn_train = grp["attention_mask_train"][:]
        attn_test = grp["attention_mask_test"][:]

        vocab_words = grp["vocab_words"][:] 
        vocab_ids = grp["vocab_ids"][:]
        vocab_size = len(vocab_words)

        intent_ids = grp["intent_ids"][:]
        num_intents = len(intent_ids)
        pad_id = 0

    return (
        X_train, y_train,
        X_test, y_test,
        attn_train, attn_test,
        vocab_size, num_intents, pad_id
    )


# train / val
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += input_ids.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += input_ids.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


# compute class weights
def compute_class_weights(y_train_np, device):
    """
    weight[c] = N / (K * n_c)
    """
    num_classes = int(np.max(y_train_np)) + 1
    class_counts = np.bincount(y_train_np, minlength=num_classes)
    total_samples = len(y_train_np)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = total_samples / (num_classes * class_counts.astype(np.float32))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights


def main():
    parser = argparse.ArgumentParser()

    # path & data
    parser.add_argument("--h5_path", type=str,
                        default="data/preprocessed_data/atis_preprocessed.h5")
    parser.add_argument("--group_name", type=str,
                        default="len40_lower_noaug",
                        help="HDF5 group name, e.g. len30_lower_noaug / len40_lower_aug")

    # model hyperparameters
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # experimental options
    parser.add_argument("--use_class_weights", action="store_true",
                        help="If set, use class-weighted cross entropy")

    # save model
    parser.add_argument("--save_dir", type=str, default="models/BiLSTM_library")
    parser.add_argument("--save_name", type=str,
                        default=None,
                        help="If None, will be bilstm_{group_name}.pt")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load data
    (
        X_train,
        y_train,
        X_test,
        y_test,
        attn_train,
        attn_test,
        vocab_size,
        num_intents,
        pad_id,
    ) = load_h5_group(args.h5_path, args.group_name)

    print(f"Loaded group '{args.group_name}' from {args.h5_path}")
    print("Train size:", X_train.shape[0])
    print("Test size :", X_test.shape[0])
    print("Vocab size:", vocab_size)
    print("Num intents:", num_intents)
    print("PAD_ID:", pad_id)

    # divide train / val
    num_train = X_train.shape[0]
    indices = np.arange(num_train)
    np.random.shuffle(indices)

    split = int(num_train * (1.0 - args.val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_tr, y_tr, attn_tr = X_train[train_idx], y_train[train_idx], attn_train[train_idx]
    X_val, y_val, attn_val = X_train[val_idx], y_train[val_idx], attn_train[val_idx]

    print(f"Train subset: {X_tr.shape[0]}, Val subset: {X_val.shape[0]}")

    # construct Dataset / DataLoader
    train_dataset = ATISDataset(X_tr, y_tr, attn_tr)
    val_dataset = ATISDataset(X_val, y_val, attn_val)
    test_dataset = ATISDataset(X_test, y_test, attn_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # define model
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        num_classes=num_intents,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_id=pad_id,
        dropout=args.dropout,
    ).to(device)

    # loss function & optimizer
    if args.use_class_weights:
        print("Using class-weighted CrossEntropyLoss")
        class_weights = compute_class_weights(y_tr, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train loop
    best_val_acc = 0.0
    best_state_dict = None
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # evaluate
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # plot training curves
    epochs = range(1, args.epochs + 1)
    os.makedirs("figures", exist_ok=True)

    # loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"BiLSTM Loss Curves ({args.group_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"loss_curve_{args.group_name}.png"))
    plt.close()

    # accuracy curve
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs,   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"BiLSTM Accuracy Curves ({args.group_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("figures", f"acc_curve_{args.group_name}.png"))
    plt.close()

    print("Saved training curves to 'figures/' directory.")


    # save model
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_name is None:
        args.save_name = f"bilstm_{args.group_name}"
        if args.use_class_weights:
            args.save_name += "_cw"
        args.save_name += ".pt"

    save_path = os.path.join(args.save_dir, args.save_name)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "num_classes": num_intents,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "pad_id": pad_id,
            "group_name": args.group_name,
            "use_class_weights": args.use_class_weights,
        },
        save_path,
    )

    print("Saved best BiLSTM model to:", save_path)


if __name__ == "__main__":
    main()
