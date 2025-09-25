from __future__ import annotations
import os
import re
import math
import random
import statistics
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import requests
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, sent_tokenize

# ---- Configs ----
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
OUT_DIR = "outputs"

# Gutenberg id for "The Call of the Wild" (plain text, no images)
# Multiple mirrors exist; we try a robust plaintext URL.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/215/pg215.txt"
RAW_TXT = os.path.join(RAW_DIR, "the_call_of_the_wild.txt")
CLEAN_TXT = os.path.join(CLEAN_DIR, "the_call_of_the_wild_clean.txt")

# Pruning rules
MIN_LEN = 2
MAX_LEN = 20
MIN_FREQ = 4
TOP_PCT_REMOVE = 0.01           # remove top 1% most frequent words
WHITELIST = {"dog"}             # never prune these (case-insensitive)

# Collocation params
WINDOW = 5                      # symmetric window size
MIN_PAIRS = 3                   # min occurrences to consider a pair
MAX_REPORT = 20                 # Top-k to report
RANDOM_BASELINE_SAMPLES = 50000 # number of random pairs for baseline stats
SEED = 42


# ---- Utils ----
def ensure_dirs():
    for d in (RAW_DIR, CLEAN_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


def maybe_download_book() -> None:
    if os.path.exists(RAW_TXT):
        return
    print("[INFO] Downloading raw book from Project Gutenberg ...")
    r = requests.get(GUTENBERG_URL, timeout=30)
    r.raise_for_status()
    with open(RAW_TXT, "wb") as f:
        f.write(r.content)
    print(f"[INFO] Saved -> {RAW_TXT}")


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """
    Remove Gutenberg header/footer using common markers.
    Falls back to naive trimming if markers are not found.
    """
    start_pat = re.compile(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*", re.I)
    end_pat   = re.compile(r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*", re.I)

    start = start_pat.search(raw_text)
    end = end_pat.search(raw_text)

    if start and end:
        core = raw_text[start.end(): end.start()]
    else:
        # Fallback: keep all
        core = raw_text
    return core.strip()


def read_or_clean_text() -> str:
    """Load the cleaned body text; create it if missing."""
    if os.path.exists(CLEAN_TXT):
        return open(CLEAN_TXT, "r", encoding="utf-8", errors="ignore").read()

    raw = open(RAW_TXT, "r", encoding="utf-8", errors="ignore").read()
    clean = strip_gutenberg_boilerplate(raw)
    with open(CLEAN_TXT, "w", encoding="utf-8") as f:
        f.write(clean)
    print(f"[INFO] Saved cleaned text -> {CLEAN_TXT}")
    return clean


def nltk_bootstrap():
    """
    Ensure required NLTK resources are available.
    We'll attempt lazy download to be friendly to fresh environments.
    """
    pkgs = [
        "punkt", "punkt_tab",  # sentence & word tokenizers (punkt_tab for newer NLTK)
        "stopwords",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet", "omw-1.4",
    ]
    for p in pkgs:
        try:
            nltk.data.find(p if "/" in p else f"tokenizers/{p}")
        except LookupError:
            try:
                nltk.download(p, quiet=True)
            except Exception:
                pass


def penn_to_universal(tag: str) -> str:
    """Map Penn Treebank tags to coarse Universal POS tags we need."""
    if tag.startswith("JJ"):
        return "ADJ"
    if tag.startswith("NN"):
        return "NOUN"
    # everything else is irrelevant here
    return "OTHER"


def penn_to_wordnet_pos(tag: str):
    """Map Penn tag to WordNet POS for lemmatization."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def build_tokens_with_pos(text: str) -> List[Tuple[str, str, str]]:
    """
    Tokenize, POS-tag, and lemmatize.
    Returns list of tuples: (token_lower, lemma_lower, universal_pos)
    """
    lemm = WordNetLemmatizer()
    out = []
    # Sentence-level tokenization for better tagging quality
    for sent in sent_tokenize(text):
        words = word_tokenize(sent)
        # keep only alphabetic strings
        words = [w for w in words if re.fullmatch(r"[A-Za-z]+", w)]
        if not words:
            continue
        tagged = pos_tag(words)  # Penn tags
        for w, penn in tagged:
            uni = penn_to_universal(penn)
            wn_pos = penn_to_wordnet_pos(penn)
            lemma = lemm.lemmatize(w.lower(), pos=wn_pos)
            out.append((w.lower(), lemma.lower(), uni))
    return out


def prune_vocabulary(tokens: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Apply pruning rules to lemmas:
      - remove stopwords
      - keep only alphabetic lemmas with length in [MIN_LEN, MAX_LEN]
      - remove lemmas with total freq < MIN_FREQ
      - remove top 1% most frequent lemmas
      - whitelist words (e.g., 'dog') to never prune
    Returns the filtered token triples with the same ordering.
    """
    sw = set(stopwords.words("english"))
    lemmas = [lem for _, lem, _ in tokens]
    counts = Counter(lemmas)

    # frequency threshold
    keep_by_freq = {lem for lem, c in counts.items() if c >= MIN_FREQ}

    # remove top 1% most frequent (but keep whitelisted)
    total_vocab = len(counts)
    k_top = max(1, int(total_vocab * TOP_PCT_REMOVE))
    top_lemmas = {lem for lem, _ in counts.most_common(k_top)}
    top_lemmas -= {w.lower() for w in WHITELIST}

    def ok(lemma: str) -> bool:
        if lemma in WHITELIST:
            return True
        if lemma in sw:
            return False
        if not (MIN_LEN <= len(lemma) <= MAX_LEN):
            return False
        if lemma not in keep_by_freq:
            return False
        if lemma in top_lemmas:
            return False
        if not re.fullmatch(r"[a-z]+", lemma):
            return False
        return True

    filtered = [(tok, lem, pos) for (tok, lem, pos) in tokens if ok(lem)]
    return filtered


def collect_pair_distances(
    seq: List[Tuple[str, str, str]], window: int
) -> Dict[Tuple[str, str, str], List[int]]:
    """
    Slide a symmetric window and collect absolute distances for
    (ADJ–NOUN) and (NOUN–NOUN) pairs.

    seq is a list of (token, lemma, universal_pos).
    We use lemmas to define pairs.
    The pair 'type' is a string "ADJ-NOUN" or "NOUN-NOUN".
    """
    index_by_pos = [i for i, (_, _, pos) in enumerate(seq)]
    # direct O(n*window) scan
    dist_map: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

    n = len(seq)
    for i in range(n):
        w1, l1, p1 = seq[i]
        if p1 not in ("ADJ", "NOUN"):
            continue
        j_lo = max(0, i - window)
        j_hi = min(n, i + window + 1)
        for j in range(j_lo, j_hi):
            if j == i:
                continue
            w2, l2, p2 = seq[j]
            if p2 not in ("ADJ", "NOUN"):
                continue

            # keep only the two types we care about
            if p1 == "ADJ" and p2 == "NOUN":
                key = (l1, l2, "ADJ-NOUN")
            elif p1 == "NOUN" and p2 == "NOUN":
                # order matters a bit less; we store lexicographically to dedup
                if l1 <= l2:
                    key = (l1, l2, "NOUN-NOUN")
                else:
                    key = (l2, l1, "NOUN-NOUN")
            else:
                continue

            dist = abs(j - i)
            if dist == 0:
                continue
            dist_map[key].append(dist)

    return dist_map


def random_baseline_distances(seq: List[Tuple[str, str, str]], window: int, samples: int) -> List[int]:
    """
    Draw random token pairs and collect their absolute index distances
    (limited to <= window to match the collocation window).
    """
    n = len(seq)
    rng = random.Random(SEED)
    dists = []
    for _ in range(samples):
        i = rng.randrange(n)
        # choose a j within window (but not equal)
        offset = rng.randint(1, window)
        j = i + offset if rng.random() < 0.5 else i - offset
        if 0 <= j < n:
            dists.append(abs(j - i))
    return dists


def z_p_value(sample_mean: float, pop_mean: float, pop_std: float, n: int) -> float:
    """
    Two-sided z-test p-value using baseline mean/std as population stats.
    """
    if pop_std <= 0 or n <= 1:
        return 1.0
    z = (sample_mean - pop_mean) / (pop_std / math.sqrt(n))
    # two-sided
    # p = 2 * (1 - Phi(|z|)) ~ using erfc
    p = math.erfc(abs(z) / math.sqrt(2))
    return max(min(p, 1.0), 0.0)


def summarize_pairs(
    dist_map: Dict[Tuple[str, str, str], List[int]],
    baseline: List[int],
    min_pairs: int,
    top_k: int
) -> pd.DataFrame:
    """
    Turn raw distance lists into a sorted dataframe with mean distance and p-values.
    """
    base_mean = statistics.mean(baseline)
    base_std = statistics.pstdev(baseline) if len(baseline) > 1 else 0.0

    rows = []
    for (l1, l2, typ), dlist in dist_map.items():
        if len(dlist) < min_pairs:
            continue
        m = statistics.mean(dlist)
        p = z_p_value(m, base_mean, base_std, len(dlist))
        rows.append((l1, l2, typ, m, len(dlist), p))

    df = pd.DataFrame(rows, columns=["w1", "w2", "type", "mean_dist", "pairs", "p_value"])
    df = df.sort_values(["mean_dist", "pairs", "p_value"], ascending=[True, False, True]).head(top_k)
    return df


def save_outputs(df_all: pd.DataFrame, df_dog: pd.DataFrame) -> None:
    df_all.to_csv(os.path.join(OUT_DIR, "colloc_overall.csv"), index=False)
    df_dog.to_csv(os.path.join(OUT_DIR, "colloc_dog.csv"), index=False)

    # preview markdown
    lines = ["# Collocations in *The Call of the Wild*",
             "",
             f"Top-{len(df_all)} overall (adj–noun / noun–noun), by mean distance (smaller=tighter):",
             ""]
    for _, r in df_all.iterrows():
        lines.append(f"- {r.w1} – {r.w2}  ({r.type}), mean={r.mean_dist:.2f}, n={int(r.pairs)}, p={r.p_value:.3f}")
    lines.append("")
    lines.append(f"Top-{len(df_dog)} around **dog**:")
    lines.append("")
    if df_dog.empty:
        lines.append("_No pairs under current filters (try larger WINDOW or lower MIN_PAIRS)._")
    else:
        for _, r in df_dog.iterrows():
            lines.append(f"- {r.w1} – {r.w2}  ({r.type}), mean={r.mean_dist:.2f}, n={int(r.pairs)}, p={r.p_value:.3f}")

    with open(os.path.join(OUT_DIR, "preview.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Saved outputs -> {OUT_DIR}/colloc_overall.csv, colloc_dog.csv, preview.md")


def main():
    random.seed(SEED)
    ensure_dirs()
    nltk_bootstrap()
    maybe_download_book()
    text = read_or_clean_text()

    print("[INFO] Tokenizing + POS-tagging + lemmatizing (NLTK) ...")
    triples = build_tokens_with_pos(text)
    print(f"[INFO] Raw tokens (alpha only after filtering): {len(triples):,}")

    print("[INFO] Pruning vocabulary ...")
    triples = prune_vocabulary(triples)
    print(f"[INFO] Tokens after pruning: {len(triples):,}")

    print("[INFO] Collecting pair distances ...")
    dist_map = collect_pair_distances(triples, WINDOW)

    print("[INFO] Building random baseline ...")
    baseline = random_baseline_distances(triples, WINDOW, RANDOM_BASELINE_SAMPLES)
    if not baseline:
        raise RuntimeError("Baseline is empty; check tokenization/window settings.")

    print("[INFO] Summarizing overall top pairs ...")
    df_all = summarize_pairs(dist_map, baseline, MIN_PAIRS, MAX_REPORT)

    print("[INFO] Summarizing 'dog' pairs ...")
    dog_keys = {k: v for k, v in dist_map.items() if "dog" in (k[0], k[1])}
    df_dog = summarize_pairs(dog_keys, baseline, MIN_PAIRS, MAX_REPORT)

    save_outputs(df_all, df_dog)


if __name__ == "__main__":
    main()
