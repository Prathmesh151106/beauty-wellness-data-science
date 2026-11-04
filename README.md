# Personalized Skincare Recommendation System: Mitigating Choice Paralysis & Adverse Reactions
> **Domain:** Beauty & Wellness Analytics | **Program:** Data Science Internship — Week 1 Task  
> **Author:** Prathmesh Jadhav  
> **Status:** Orientation & Problem Definition Complete  

---

## 📌 1. Project Overview & Business Problem
In the global beauty and wellness e-commerce market (projected to reach **$590 billion by 2030** with e-commerce driving **28% of sales** and AI commerce influencing up to **35% of digital transactions**), consumers face acute **digital choice paralysis**:

- **Search Failure:** Over 81% of brand search engines fail on simple 2-word queries, driving a **53% cart abandonment rate** among online shoppers.
- **Dermatological Mismatch:** Unweighted star ratings conflate incompatible biological skin phenotypes (Oily vs. Dry vs. Sensitive). Formulation mismatches cause **20% to 65% of online beauty returns**, resulting in high reverse logistics costs and customer churn.

### Objective
This project designs a **Hybrid Ingredient-Aware Personalized Skincare Recommendation System** combining:
1. **Chemical INCI Vectorization** (TF-IDF & Word2Vec embeddings with regulatory concentration position-weighting).
2. **Aspect-Based Sentiment Analysis (ABSA)** using XGBoost / DistilBERT across ~1M customer reviews to isolate clinical efficacy and skin tolerance from shipping or packaging noise.
3. **Bayesian Hybrid Recommender** fusing content-based similarity with collaborative matrix factorization (SVD / LightFM) and inferred skin-type classifiers via Tree-structured Parzen Estimator (`Optuna` TPE).

---

## 🗂️ 2. Repository Structure

```text
├── README.md               <- Project overview, problem definition, and analytical roadmap
├── docs/                   <- Formal deliverables, reports, and documentation
│   └── Week_1_Beauty_Wellness_Personalized_Skincare_Problem_Definition.docx
├── notebooks/              <- Jupyter notebooks for EDA and modeling (Upcoming Weeks 2-4)
├── src/                    <- Python production source code (modules, pipelines, utils)
└── requirements.txt        <- Python dependencies and libraries


🔬 3. Research Questions & Hypotheses (Week 1 Scope)

    RQ1 (Chemical Vectorization): Does combining INCI chemical embeddings with user skin-type encodings improve ranking precision (NDCG@10) and biocompatibility over popularity baselines?

    RQ2 (Aspect Sentiment): Does ABSA weighting on clinical tolerance versus non-clinical attributes reduce false-positive recommendations for sensitive cohorts?

    RQ3 (Hybrid Ensemble): Does a Bayesian TPE-optimized hybrid model outperform standalone collaborative filtering under cold-start conditions?

    RQ4 (Price vs. Efficacy): Do active ingredient concentrations predict consumer recommendations (is_recommended) more strongly than nominal price tiers?

📊 4. Primary Datasets

    Sephora Products and Skincare Reviews Dataset (Kaggle)

        Over 8,000 beauty products (product_info.csv)

        ~1,000,000 skincare customer reviews with self-reported skin type, skin tone, star rating, and text.

    Sephora Cosmetics Chemical Components Dataset (Kaggle)

        1,472 curated skincare products with chemical INCI ingredient lists and binary skin-type ground truth.

🛠️ 5. Technology Stack

    Language: Python 3.11+

    Data Engineering: Pandas, NumPy, PyArrow, Regex

    Machine Learning: Scikit-Learn, LightFM, Surprise, Optuna (TPE)

    Natural Language Processing (NLP): SpaCy, NLTK, VADER, XGBoost, HuggingFace Transformers

    Visualization: Matplotlib, Seaborn, Plotly, UMAP

    Deployment & Prototype: Streamlit, FastAPI

🚀 6. Next Steps (Week 2 Onwards)

    [x] Week 1: Data Science Orientation, Problem Definition & Formal DOCX Report

    [ ] Week 2: Data Ingestion, INCI Standardization & Exploratory Data Analysis (EDA)

    [ ] Week 3: Aspect-Based Sentiment Classification (ABSA) Pipeline

    [ ] Week 4: Hybrid Recommendation Modeling & Offline Evaluation (NDCG, Recall, Diversity)

    [ ] Week 5: Interactive Streamlit Decision Support Prototype & Final Report
