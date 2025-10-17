# Assignment 6: Imputation via Regression for Missing Data  

**Name:** Priyanshu Verma  
**Roll no:** CH22B087  
**Date:** 17-10-2025  

---

## Project Overview  

This notebook explores the impact of **different missing data imputation strategies** on the performance of a **credit risk classification model** using the **UCI Credit Card Default Clients dataset**.  

The assignment introduces **Missing At Random (MAR)** values artificially in selected numerical features and compares multiple approaches to handle missing data:  
- **Median Imputation (Baseline)**  
- **Linear Regression Imputation**  
- **Non-Linear Regression Imputation (KNN)**  
- **Listwise Deletion (Row Removal)**  

Each imputed dataset is evaluated using a **Logistic Regression classifier**, with model performance compared across **Accuracy, Precision, Recall, and F1-score** to assess the efficacy of each imputation strategy.

---

## Files in this Repository  

- `DA5401_Assignment_6.ipynb` - Main Jupyter Notebook containing the full implementation.
- `UCI_Credit_Card.csv` - The Dataset
- `README.md` - This file.


---

## Notebook Structure  

The notebook is organized into the following logical parts:  

1. **Assignment Description** — Overview of objectives and dataset details.  
2. **Part A: Data Preprocessing and Imputation**  
   - Introduce MAR missingness (5–10%) in selected numerical columns.  
   - Perform Median, Linear Regression, and Non-Linear Regression imputations.  
3. **Part B: Model Training and Evaluation**  
   - Split into train/test sets and standardize features.  
   - Train Logistic Regression classifiers on all imputed datasets and Listwise Deletion dataset.  
   - Generate full classification reports.  
4. **Part C: Comparative Analysis**  
   - Compare all models (A–D) using key metrics (especially F1-score).  
   - Discuss trade-offs between deletion and imputation.  
   - Provide final recommendations on the most effective imputation strategy.
   - Summarize findings and conceptual implications.

---

### Key Observations  
- All three imputation strategies (A–C) achieved **identical performance**, showing that missingness was minimal and largely MAR.  
- **Listwise Deletion** significantly reduced accuracy and F1 due to information loss.  
- **Linear Regression Imputation** matched the baseline, suggesting primarily **linear dependencies** between features.  

---

## Final Recommendation  

> **Median or Linear Regression Imputation** are the most suitable strategies for this dataset.  
> They maintain full data integrity, achieve the highest weighted F1-scores (0.776), and are efficient and interpretable.  
> Non-linear imputation should be considered only for datasets with stronger non-linear feature interactions or higher missingness.  
> Listwise deletion is **not recommended** due to its substantial data loss and reduced performance.  

---

## Requirements  

Recommended Python environment:  

- Python 3.8+ (3.9/3.10 recommended)  
- JupyterLab or Jupyter Notebook  

### Required Python Packages  

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```
