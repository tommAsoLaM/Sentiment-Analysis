# 🌟 Amazon Fine Food Reviews: Sentiment Analysis with BERT

An end-to-end Natural Language Processing (NLP) pipeline that leverages a pre-trained transformer model to predict 1-to-5 star sentiment ratings on the Amazon Fine Food Reviews dataset. 

Rather than treating sentiment as a simple binary (Positive vs. Negative), this project predicts nuanced, multi-class ratings and evaluates the model's performance using **custom subjective metrics** designed to account for the ambiguity of human language.

## 📁 Repository Structure
* **`sentiment-analysis-fr.ipynb`**: The main Jupyter/Kaggle notebook. It contains the data loading, preprocessing, zero-shot Hugging Face inference pipeline, and confusion matrix visualizations.
* **`utils_task.py`**: A custom evaluation module. It contains advanced `accuracy_score`, `precision_score`, and `recall_score` functions that calculate both strict and relaxed metrics for 1-to-5 star classification.

## 🧠 The Model
This project utilizes the `nlptown/bert-base-multilingual-uncased-sentiment` model via the Hugging Face `pipeline`. It performs zero-shot sequence classification to predict exact star ratings directly from the raw review text.

## 📊 Custom Evaluation Metrics (`utils_task.py`)
Because the difference between a 4-star and 5-star review (or a 1-star and 2-star review) is highly subjective, standard evaluation metrics often fail to capture a model's true understanding of sentiment polarity. 

To solve this, the `utils_task.py` module evaluates the model using two distinct methodologies concurrently:
1. **Strict Evaluation:** The predicted star rating must exactly match the ground truth (e.g., Actual 5 = Predicted 5).
2. **Relaxed Evaluation:** The prediction is considered correct if it successfully identifies the correct sentiment polarity. It forgives "off-by-one" errors on the extremes by grouping 4s with 5s (Positive) and 1s with 2s (Negative).

## 🚀 Results

| Metric | Strict Score | Relaxed Score |
| :--- | :--- | :--- |
| **Accuracy** | 63.23% | 84.74% |
| **Precision** | 68.03% | 74.22% |
| **Recall** | 72.21% | 75.93% |

### Key Insights
* **Polarity vs. Granularity:** The massive **21.5% jump in accuracy** between the strict and relaxed evaluations proves the model has a highly accurate understanding of overall sentiment polarity. When the model "misses," it is predominantly confusing a 4-star review with a 5-star review, which aligns with standard human disagreement.
* **Real-World Viability:** An 84.7% relaxed accuracy indicates this zero-shot model is highly viable for a production environment to automatically route negative reviews (1-2 stars) to customer service and positive reviews (4-5 stars) to marketing.

## 🛠️ How to Run Locally

### Requirements
Ensure you have the following libraries installed:
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib
