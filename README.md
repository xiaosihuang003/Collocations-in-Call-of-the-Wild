# Collocations in *The Call of the Wild*

A NLTK pipeline that downloads Jack London’s **The Call of the Wild** (Project Gutenberg #215), cleans the text, prunes the vocabulary, and computes **collocations** using mean absolute token distance within a ±5 word window. We report the strongest (smallest mean distance) **NOUN–NOUN** and **ADJ–NOUN** pairs, plus pairs around the word **“dog”**.

---

## How to run

    # (optional) create & activate a virtual env
    python3 -m venv .venv
    source .venv/bin/activate

    # install deps
    pip install -r requirements.txt

    # run
    python collocations.py

Outputs will appear in the `outputs/` directory.

---

## Method (brief)

- **Tokenization & POS tagging**: NLTK `sent_tokenize`, `word_tokenize`, and `pos_tag`.
- **Lemmas**: WordNet lemmatizer; pairs are formed on **lemmas**.
- **Pruning** (Exercise 2.3 rules):
  - remove NLTK English stopwords;
  - keep only alphabetic lemmas with length ∈ [2, 20];
  - drop lemmas with total count < 4;
  - drop top 1% most‑frequent lemmas (to reduce function‑word noise);
  - whitelist `"dog"` so it’s never pruned.
- **Window**: ±5 tokens; collect absolute distances for **NOUN–NOUN** and **ADJ–NOUN** pairs.
- **Significance**: Compare each pair’s mean distance to a random‑pair baseline via a two‑sided z‑test; report **p‑value**.

---

## Results (preview)

Full tables are in the CSVs; below are the first items for quick reading.

### Top‑20 overall (smallest mean distance)

- club – fang *(NOUN–NOUN)*, mean≈1.00, n=14, p≈1.16e‑07  
- fore – leg *(NOUN–NOUN)*, mean≈1.00, n=14, p≈1.16e‑07  
- red – sweater *(ADJ–NOUN)*, mean≈1.00, n=12, p≈9.28e‑07  
- country – law *(NOUN–NOUN)*, mean≈1.00, n=8,  p≈6.17e‑05  
- trace – trail *(NOUN–NOUN)*, mean≈1.00, n=8,  p≈6.17e‑05  
- clara – santa *(NOUN–NOUN)*, mean≈1.00, n=8,  p≈6.17e‑05  
- clara – valley *(NOUN–NOUN)*, mean≈1.00, n=8,  p≈6.17e‑05  
- rest – team *(NOUN–NOUN)*, mean≈1.00, n=8,  p≈6.17e‑05  
- salt – water *(NOUN–NOUN)*, mean≈1.00, n=8,  p≈6.17e‑05  
- forth – paragraph *(NOUN–NOUN)*, mean≈1.00, n=8, p≈6.17e‑05  
- hundred – yard *(ADJ–NOUN)*, mean≈1.00, n=7, p≈1.79e‑04  
- anyone – united *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- toot – ysabel *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- end – rope *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- air – mid *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- hell – lak *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- forty – mile *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- mile – quarter *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- bank – yukon *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  
- head – pack *(NOUN–NOUN)*, mean≈1.00, n=6, p≈5.22e‑04  

**Read:** very tight pairs (mean distance ≈1) with tiny p‑values show fixed phrases and names from the novel’s sledding world (e.g., *club–fang*, *trace–trail*, *salt–water*).

### Top‑20 pairs around **“dog”**

- breath – dog *(NOUN–NOUN)*, mean≈1.00, n=4, p≈0.005  
- dog – manner *(NOUN–NOUN)*, mean≈1.33, n=6, p≈0.004  
- main – dog *(ADJ–NOUN)*, mean≈1.33, n=3, p≈0.041  
- dog – outside *(NOUN–NOUN)*, mean≈1.50, n=16, p≈1.2e‑04  
- cliff – dog *(NOUN–NOUN)*, mean≈1.50, n=4, p≈0.023  
- dog – toil *(NOUN–NOUN)*, mean≈1.50, n=4, p≈0.033  
- advance – dog *(NOUN–NOUN)*, mean≈1.67, n=6, p≈0.003  
- dog – half *(NOUN–NOUN)*, mean≈1.67, n=6, p≈0.003  
- ten – dog *(ADJ–NOUN)*, mean≈1.67, n=3, p≈0.102  
- strange – dog *(ADJ–NOUN)*, mean≈1.75, n=4, p≈0.076  
- fourteen – dog *(ADJ–NOUN)*, mean≈1.80, n=5, p≈0.057  
- dog – hour *(NOUN–NOUN)*, mean≈2.00, n=10, p≈0.025  
- dog – mother *(NOUN–NOUN)*, mean≈2.00, n=4, p≈0.156  
- dog – spring *(NOUN–NOUN)*, mean≈2.00, n=4, p≈0.156  
- dog – joe *(NOUN–NOUN)*, mean≈2.00, n=4, p≈0.156  
- wild – dog *(ADJ–NOUN)*, mean≈2.00, n=4, p≈0.156  
- dog – rabbit *(NOUN–NOUN)*, mean≈2.00, n=4, p≈0.156  
- course – dog *(NOUN–NOUN)*, mean≈2.00, n=4, p≈0.156  
- dog – mush *(NOUN–NOUN)*, mean≈2.00, n=4, p≈0.156  
- dead – dog *(ADJ–NOUN)*, mean≈2.00, n=3, p≈0.219  

**Read:** frequent tight pairings like *dog–outside*, *advance–dog*, *dog–toil* reflect sled work, harsh travel, and survival scenes that define the book’s setting.

---

## Notes

- The CSVs contain exact numeric values (mean distances, counts, p‑values).  
- You can tweak window size, frequency thresholds, or the random‑baseline sample size at the top of `collocations.py`.
