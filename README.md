# Edu2Job – Predicting Job Roles from Educational Background

## Project Overview
Edu2Job is a machine learning project developed as part of the **Infosys Springboard Internship (Batch 3)**.  
The goal is to predict suitable **job roles** based on a candidate’s **educational qualification, experience, and skills**.

---

## Milestone 1 – Dataset & EDA
- **Dataset Source:** Kaggle – `candidate_job_role_dataset`
- **Tasks Completed:**
  - Data Cleaning (duplicates, missing values, normalization)
  - Feature extraction (`num_skills`, `qualification_course`)
  - Exploratory Data Analysis (EDA)
  - Visualization using Seaborn & Matplotlib

**Cleaned file:** `cleaned_job_roles.csv`  
**Notebook:** `milestone1_EDA.ipynb`

---

## Milestone 2 – Preprocessing
- Encoded categorical features (qualification, experience_level, etc.)
- Standardized numerical columns
- Feature Selection
- Split data into train and test sets

## Milestone 3 – Model Training and Evaluation

**Machine Learning Models Used**
-Logistic Regression
-Decision Tree Classifier
-Random Forest Classifier
-Support Vector Machine (SVM)
-XGBoost Classifier

**Techniques Applied**

**1️.Cross-Validation:**
To ensure reliability and reduce bias, k-Fold Cross Validation (k=5) was applied to assess each model’s performance consistency across different splits.

**2️.Regularization:**
Regularization techniques were applied to control overfitting by limiting model complexity
Logistic Regression: Used L2 regularization
Random Forest: Reduced max_depth, max_features, and increased min_samples_split
XGBoost: Tuned gamma, reg_alpha, and reg_lambda parameters

**3️.Hyperparameter Tuning (RandomizedSearchCV):**
To identify optimal hyperparameters efficiently, RandomizedSearchCV was applied to XGBoost and Random Forest models to achieve better generalization and balanced accuracy.


## License
This project is licensed under the **MIT License** – see the 'LICENSE' file for details.

## Contact
For any questions, reach out to **Milka Joseph**.
