import nltk
import random
from typing import List
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
nltk.download("omw-1.4")

# keep these words "to/and/the" unchanged
STOPWORDS = {
    "a", "an", "the", "to", "of", "and", "or", "is", "are", "am", "in", "on",
    "at", "for", "from", "with", "by", "this", "that", "these", "those", "it"
}

def get_synonyms(word: str) -> List[str]:
    """use WordNet to find synonyms"""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ").lower()
            # only keep normal words
            if w != word and w.isalpha():
                synonyms.add(w)
    return list(synonyms)

# synonym replacement
def synonym_replacement(tokens: List[str], n: int = 1) -> List[str]:
    """
    randomly pick n words, replace them with their synonyms
    - tokens: orginal words list
    - n: max replaced words
    """
    new_tokens = tokens.copy()
    candidates = [i for i, w in enumerate(new_tokens)
                  if w not in STOPWORDS and w.isalpha() and len(w) > 1]
    random.shuffle(candidates)

    num_replaced = 0
    for idx in candidates:
        word = new_tokens[idx]
        syns = get_synonyms(word)
        if len(syns) == 0:
            continue
        new_word = random.choice(syns)
        new_tokens[idx] = new_word
        num_replaced += 1
        if num_replaced >= n:
            break
    return new_tokens

# random deletion
def random_deletion(tokens: List[str], p: float = 0.1) -> List[str]:
    """
    delete each word with probability p (retain at least one)
    """
    if len(tokens) == 1:
        return tokens
    new_tokens = []
    for w in tokens:
        if random.random() > p:
            new_tokens.append(w)
    if len(new_tokens) == 0:
        new_tokens.append(random.choice(tokens))
    return new_tokens

# random insertion
def random_insertion(tokens: List[str], n: int = 1) -> List[str]:
    """
    randomly do n times: find a word -> get its synonyms -> insert into a random position
    """
    new_tokens = tokens.copy()
    for _ in range(n):
        candidates = [w for w in new_tokens
                      if w not in STOPWORDS and w.isalpha()]
        if not candidates:
            break
        base_word = random.choice(candidates)
        syns = get_synonyms(base_word)
        if not syns:
            continue
        new_word = random.choice(syns)
        insert_pos = random.randint(0, len(new_tokens))
        new_tokens.insert(insert_pos, new_word)
    return new_tokens



# multiple augments
def augment_tokens(tokens: List[str],
                   prob_sr: float = 0.34,
                   prob_rd: float = 0.33,
                   prob_ri: float = 0.33,
                   n_sr: int = 1,
                   p_rd: float = 0.1,
                   n_ri: int = 1) -> List[str]:
    """
    perform lightweight combination enhancement on a token list
    - perform synonym replacement with probability: prob_sr
    - perform random deletion with probability: prob_rd
    - perform random insertion with probability: prob_ri
    """
    aug = tokens.copy()

    # synonym replacement
    if random.random() < prob_sr:
        aug = synonym_replacement(aug, n=n_sr)

    # random deletion
    if random.random() < prob_rd:
        aug = random_deletion(aug, p=p_rd)

    # random insertion
    if random.random() < prob_ri:
        aug = random_insertion(aug, n=n_ri)

    return aug
