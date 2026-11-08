import h5py
import numpy as np

# read test
SAVE_PATH = "data/preprocessed_data/atis_preprocessed.h5"

def print_h5_structure(filename):
    def print_group(name, obj):
        print(name)
    with h5py.File(filename, "r") as f:
        f.visititems(print_group)

print_h5_structure(SAVE_PATH)

with h5py.File(SAVE_PATH, "r") as f:
    grp = f["len40_lower_noaug"]
    X_train = np.array(grp["X_train"])
    y_train = np.array(grp["y_train"])
    X_test  = np.array(grp["X_test"])
    y_test  = np.array(grp["y_test"])
    attn_train = np.array(grp["attention_mask_train"])
    attn_test  = np.array(grp["attention_mask_test"])

    words = [w.decode("utf-8") for w in grp["vocab_words"][:]]
    words_ids = grp["vocab_ids"][:]
    i2t = dict(zip(words_ids, words)) 

    intents = [w.decode("utf-8") for w in grp["intents"][:]]
    intent_ids   = grp["intent_ids"][:]
    i2in = dict(zip(intent_ids, intents))
    
    print("config:", dict(grp.attrs))

    print("Example:", X_train[0])
    print("Length:", len(X_train[0]))
    print("Tokens Number: ", len(X_train))
    print("Intent:", y_train[0], " (", i2in[y_train[0]], ")")
    print("Decoded:", [i2t[i] for i in X_train[0]])
    print("attention_mask:", attn_train[0])
