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
