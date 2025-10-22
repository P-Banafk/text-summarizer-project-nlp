
---

## 1. Introduction / Project Overview 🎯

* **Goal:** Briefly state the project's aim (e.g., "To summarize long news articles using both extractive and abstractive methods and compare their performance").
* **Methods:** Mention the specific techniques used (TextRank for extractive, fine-tuning T5 for abstractive).
* **Dataset:** Name the dataset used (e.g., "News Summary dataset from Kaggle") and link to it. Mention the specific columns used (`ctext`/`full_text` for the article, `text`/`summary` for the reference).
* **Evaluation Metric:** State that **ROUGE** scores (specifically ROUGE-1, ROUGE-2, ROUGE-L F-measures) were used for evaluation.

---

## 2. Setup and Environment ⚙️

* **Environment:** Describe how you set up your environment (e.g., "Python 3.13 virtual environment", or "Conda environment with Python 3.10").
* **Key Libraries:** List the main libraries installed and their versions (especially important for reproducibility):
    * `pandas`
    * `spacy` (and the model like `en_core_web_sm`)
    * `pytextrank`
    * `rouge-score`
    * `datasets`
    * `transformers`
    * `torch`
    * `accelerate`
    * `evaluate`
    * *(If you used the older method first: `gensim==3.8.3`)*

---

## 3. Phase 1: Extractive Summarization (TextRank Baseline) 📊

* **Method:** Explain TextRank briefly (graph-based, ranks sentences by importance, selects top sentences). Mention using the `pytextrank` library integrated with `spaCy`.
* **Implementation:**
    * Describe the data loading and cleaning steps (using pandas, handling NaNs).
    * Explain how you sampled the data (e.g., "Used the first 500 articles for baseline calculation").
    * Show or describe the core code used to generate summaries (loading `spacy`, adding `pytextrank` pipe, using `doc._.textrank.summary()`).
    * Explain how ROUGE scores were calculated for each generated summary against the reference.
* **Results:** State the **final average ROUGE scores** achieved by TextRank (ROUGE-1: 0.3929, ROUGE-2: 0.1703, ROUGE-L: 0.2482). Clearly label this as the **baseline**.

---

## 4. Phase 2: Abstractive Summarization (T5 Fine-tuning) 🤖

* **Method:** Explain abstractive summarization briefly (model generates new text based on understanding). Mention using the pre-trained `t5-small` model from Hugging Face Transformers.
* **Implementation:**
    * Describe converting the pandas DataFrame to a Hugging Face `Dataset` object and why (`datasets` library efficiency).
    * Explain the data sampling (e.g., "Used 1000 examples for training, 200 for validation").
    * Detail the **preprocessing steps:** tokenization using `AutoTokenizer`, adding the `"summarize: "` prefix, setting max lengths (`MAX_INPUT_LENGTH=512`, `MAX_TARGET_LENGTH=150`), and tokenizing labels using `as_target_tokenizer` or `text_target`. Show the `preprocess_function`.
    * List the **hyperparameters** used in `Seq2SeqTrainingArguments` (learning rate, batch size, epochs, etc.).
    * Mention using `DataCollatorForSeq2Seq` for dynamic padding.
    * Explain the `compute_metrics` function for calculating ROUGE scores during evaluation.
    * Note the use of `Seq2SeqTrainer` to handle the training loop.
* **Results:**
    * State the **final average ROUGE scores** achieved by the fine-tuned T5 model after 3 epochs (ROUGE-1: ~0.2497, ROUGE-2: ~0.1306, ROUGE-L: ~0.2133).
    * Include the **qualitative analysis:** mention that generated summaries were coherent but often basic/incomplete, using examples if helpful.

---

## 5. Comparison and Conclusion 🏁

* **Direct Comparison:** Put the TextRank baseline scores and the T5 fine-tuned scores side-by-side (a table works well).
* **Analysis:** State clearly that, **under the tested conditions (limited data/epochs)**, the TextRank baseline outperformed the fine-tuned `t5-small` model based on ROUGE scores.
* **Discussion:** Explain *why* this likely happened (insufficient training data/time for the abstractive model).
* **Future Work:** Suggest next steps if the project were to continue (e.g., "Train T5 on a larger dataset," "Train for more epochs," "Experiment with `t5-base` model," "Perform more extensive hyperparameter tuning").

---

