import os
import pickle
import numpy as np
import h5py

from augmentation import augment_tokens

DATA_DIR = "data/raw_data"
MAX_LEN = 40
SAVE_PATH = "data/preprocessed_data/atis_preprocessed.h5"
use_augmentation = False

def load_ds(fname):
    with open(fname, 'rb') as f:
        ds, dicts = pickle.load(f)
    return ds, dicts

train_ds, dicts = load_ds(os.path.join(DATA_DIR, "atis.train.pkl"))
test_ds, dicts = load_ds(os.path.join(DATA_DIR, "atis.test.pkl"))

# original token_id / intent_id dict
t2i_orig = dicts["token_ids"]                   # word → id
i2t_orig = {i: w for w, i in t2i_orig.items()}  # id → word
in2i_orig = dicts["intent_ids"]                 # intent → id
i2in = {i: w for w, i in in2i_orig.items()}     # id → intent

# lowercase vocabulary
vocab_set = set()
for tid in range(len(i2t_orig)):
    word = i2t_orig[tid].lower()
    vocab_set.add(word)

special_tokens = ["<pad>", "<unk>"]
final_vocab = special_tokens + sorted(list(vocab_set))

new_t2i = {w: i for i, w in enumerate(final_vocab)}
new_i2t = {i: w for w, i in new_t2i.items()}
PAD_ID = new_t2i["<pad>"]
UNK_ID = new_t2i["<unk>"]

def ids_to_lower_tokens(token_ids):
    return [i2t_orig[tid].lower() for tid in token_ids]

# lower tokens to new id
def remap_tokens(tokens):
    new_ids = []
    for t in tokens:
        new_tid = new_t2i.get(t, UNK_ID)
        new_ids.append(new_tid)
    return new_ids

# fixed length MAX_LEN
def pad_or_truncate(ids):
    if len(ids) > MAX_LEN:
        return ids[:MAX_LEN]
    else:
        return ids + [PAD_ID] * (MAX_LEN - len(ids))

# complete pretreatment
def preprocess_query(token_ids):
    tokens = ids_to_lower_tokens(token_ids)
    ids = remap_tokens(tokens)
    ids = pad_or_truncate(ids)
    return ids

def preprocess_query_with_aug(token_ids):
    tokens = ids_to_lower_tokens(token_ids)
    aug_tokens = augment_tokens(tokens)
    ids = remap_tokens(aug_tokens)
    ids = pad_or_truncate(ids)
    return ids

X_train = []
y_train = []
for q, y in zip(train_ds["query"], train_ds["intent_labels"]):
    X_train.append(preprocess_query(q))
    y_train.append(y[0])
    if use_augmentation:
        X_train.append(preprocess_query_with_aug(q))
        y_train.append(y[0])

X_test = [preprocess_query(q) for q in test_ds["query"]]
y_test = [y[0] for y in test_ds["intent_labels"]]

print("Example:", X_train[0])
print("Length:", len(X_train[0]))
print("Intent:", y_train[0])
print("Decoded:", [new_i2t[i] for i in X_train[0]])

X_train = np.asarray(X_train, dtype="int32")
X_test  = np.asarray(X_test,  dtype="int32")

attention_mask_train = (X_train != PAD_ID).astype("int32")
attention_mask_test  = (X_test  != PAD_ID).astype("int32")

# save in hdf5 file
def save_version(version_name, X_train, y_train, X_test, y_test,
                 attn_train=None, attn_test=None, config=None):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with h5py.File(SAVE_PATH, "a") as f:
        # grp = f.require_group(version_name)
        if version_name in f:
            del f[version_name]
        grp = f.create_group(version_name)

        grp.create_dataset("X_train", data=np.asarray(X_train, dtype="int32"))
        grp.create_dataset("y_train", data=np.asarray(y_train, dtype="int32"))
        grp.create_dataset("X_test",  data=np.asarray(X_test,  dtype="int32"))
        grp.create_dataset("y_test",  data=np.asarray(y_test,  dtype="int32"))

        vocab_words = np.array(list(new_t2i.keys()), dtype="S")
        vocab_ids   = np.array(list(new_t2i.values()), dtype="int32")
        grp.create_dataset("vocab_words", data=vocab_words)
        grp.create_dataset("vocab_ids", data=vocab_ids)

        intents = np.array(list(in2i_orig.keys()), dtype="S")
        intent_ids   = np.array(list(in2i_orig.values()), dtype="int32")
        grp.create_dataset("intents", data=intents)
        grp.create_dataset("intent_ids", data=intent_ids)

        if attn_train is not None:
            grp.create_dataset("attention_mask_train", data=np.asarray(attn_train, dtype="int32"))
        if attn_test is not None:
            grp.create_dataset("attention_mask_test", data=np.asarray(attn_test, dtype="int32"))
        if config is not None:
            for k, v in config.items():
                grp.attrs[k] = v

version_name = (
    f"len{MAX_LEN}_lower_"
    f"{'aug' if use_augmentation else 'noaug'}"
)

config = {
    "max_len": MAX_LEN,
    "lowercase": True,
    "augmentation": bool(use_augmentation),
    "version_note": version_name
}

save_version(
    version_name,
    X_train, y_train, X_test, y_test,
    attention_mask_train, attention_mask_test,
    config=config
)

print("Saved to", SAVE_PATH)