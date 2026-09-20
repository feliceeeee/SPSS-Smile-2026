# SMILE Competition 2026 — Decode the Market Sentiment 🏆 5th Place · Team BITin BYTE

A sentiment classification project built for **SMILE Competition 2026**, themed *"Profiting from Chaos: Finding Signal in a Reality."* The task: classify Indonesian economic, financial, and business news headlines into **POSITIF**, **NETRAL**, or **NEGATIF**, scored on Kaggle leaderboard (Macro F1) plus notebook quality.

**Result: 5th place**, competing as **Team BITin BYTE**.
<img width="1367" height="731" alt="image" src="https://github.com/user-attachments/assets/f15a6be0-41d5-4ccb-af7c-329eeb65db52" />


## The approach

**Task**: predict `sentiment` for 654 held-out Indonesian news headlines, given 1,526 labeled training headlines (`id`, `kalimat`, `sentiment`). Metric: **Macro F1**, computed via `sklearn.metrics.f1_score(y_true, y_pred, average="macro")`.

The decisive edge was reading the EDA before choosing a representation, then letting that drive the model comparison:

* **EDA-driven preprocessing decision**: unigram word frequencies were nearly identical across all three classes — the dataset is dominated by one shared news topic (BRI), so single keywords carry almost no sentiment signal. Bigram/trigram analysis is where clear per-class themes emerged (POSITIF → achievement/CSR language; NETRAL → factual market-report phrasing; NEGATIF → incident/complaint language). This is why preprocessing stayed minimal (lowercasing + whitespace cleanup only, no stemming/stopword removal) and why TF-IDF was configured with `ngram_range=(1,2)`.
* **Classical baseline sweep**: Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, and KNN, each tuned across several hyperparameter values on the same 80/20 stratified holdout. **Linear SVM (C=0.25)** won this round at Macro F1 = 0.8290.
* **IndoBERT fine-tuning**, staged to keep the search efficient: Stage 1 swept learning rate × epochs on a fixed base config; Stage 2 then probed max sequence length, class weighting, batch size, and weight decay/warmup around the Stage 1 winner. Best configuration (lr=2e-5, epochs=2, max_len=128) reached Macro F1 = 0.8540 — a clear +2.5pp jump over the best classical model, confirming that contextual representation mattered far more here than squeezing more out of bag-of-words models.
* **Weighted ensemble**: IndoBERT's softmax probabilities were blended with the SVM's normalized decision scores across a full weight sweep (0–100% IndoBERT) on the validation set. The optimal blend (60% IndoBERT / 40% SVM) matched IndoBERT's standalone score exactly — a useful reminder that ensembling gains can be fragile on a ~300-sample validation set, even when the two models are quite different in nature.
* Both final models were retrained on 100% of the labeled data before generating test predictions.
* **What I'd try next**: data augmentation strategies that preserve short-headline meaning, and a larger/held-out validation set to get a more reliable read on whether ensembling actually helps here.

📓 Notebook: `notebook/BITinBYTE_Decode-the-Market-Sentiment.ipynb`

## Tech stack

`Python` · `scikit-learn` (TF-IDF, Logistic Regression, Linear SVM, Naive Bayes, Random Forest, KNN) · `PyTorch` · `Hugging Face Transformers` (IndoBERT fine-tuning) · `pandas` · `matplotlib` / `seaborn` for EDA

## How to run

⚠️ The competition dataset (`train.csv`, `test.csv`, and `sample_submission.csv`) is not included. Place all three files under a `data/` folder and update the paths in the notebook accordingly.

```
pip install numpy pandas matplotlib seaborn scikit-learn jupyter torch transformers
jupyter notebook notebook/sentiment_classification.ipynb
```

IndoBERT fine-tuning runs meaningfully faster on a CUDA-capable GPU; the notebook falls back to CPU automatically if none is available, at the cost of significantly longer training time.

## Team

Built with my SMILE Competition 2026 teammates as **Team BITin BYTE**:
* Felisha Yang — GitHub [@handle](https://github.com/handle) 
