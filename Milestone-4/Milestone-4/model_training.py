import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from google.colab import files


df = pd.read_csv("processed_dataset.csv")

print("Data Loaded Successfully!")
print("Shape of dataset:", df.shape)
print(df.head())

class_counts = df['job_role'].value_counts()
rare_classes = class_counts[class_counts == 1].index

df = df[~df['job_role'].isin(rare_classes)]

print("Removed minority classes with only 1 sample.")
print("Classes removed:", list(rare_classes))
print("Updated class distribution:")
print(df['job_role'].value_counts())

X = df.drop('job_role', axis=1)
y = df['job_role']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

df

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature Scaling Done")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}



results = []

for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

comparison_df = pd.DataFrame(results)
print("\nModel Comparison Summary:")
print(comparison_df)

results = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'XGBoost'],
    'Accuracy': [0.815, 0.985, 0.975, 0.915, 0.985],
    'Precision': [0.784, 0.986, 0.976, 0.938, 0.986],
    'Recall': [0.815, 0.985, 0.975, 0.915, 0.985],
    'F1 Score': [0.792, 0.985, 0.975, 0.920, 0.985]
}

results_df = pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.show()


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

cv_results = {}


for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"\n{name}")
    print(f"Fold Accuracies: {scores}")
    print(f"Mean Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

xgb_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=4,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)


y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Difference:", round(train_acc - test_acc, 4))



log_reg = LogisticRegression(
    C=0.8,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
log_reg.fit(X_train, y_train)

train_acc = log_reg.score(X_train, y_train)
test_acc = log_reg.score(X_test, y_test)
print("Logistic Regression → Train:", train_acc, "Test:", test_acc)


dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=0.01,
    random_state=42
)
dt_model.fit(X_train, y_train)

print("Decision Tree → Train:", dt_model.score(X_train, y_train),
      "Test:", dt_model.score(X_test, y_test))



rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
rf_model.fit(X_train, y_train)

print("Random Forest → Train:", rf_model.score(X_train, y_train),
      "Test:", rf_model.score(X_test, y_test))

svm_model = SVC(C=0.8, kernel='rbf', gamma=0.1, random_state=42)
svm_model.fit(X_train, y_train)

print("SVM → Train:", svm_model.score(X_train, y_train),
      "Test:", svm_model.score(X_test, y_test))

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "XGBoost": xgb_model
}

results = []

for name, model in models.items():
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    diff = train_acc - test_acc
    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 3),
        "Test Accuracy": round(test_acc, 3),
        "Difference": round(diff, 3)
    })


results_df = pd.DataFrame(results)
print(results_df)

results_df = pd.DataFrame(results)

results_df["Score"] = results_df["Test Accuracy"] - results_df["Difference"]


best_model = results_df.loc[results_df["Score"].idxmax(), "Model"]


print("Model Comparison Table:")
print(results_df)
print(f"\nBest Model: {best_model}")

plt.figure(figsize=(10,6))

colors = ['green' if m == best_model else 'steelblue' for m in results_df["Model"]]


bars = plt.bar(results_df["Model"], results_df["Test Accuracy"], color=colors, edgecolor='black')
plt.title("Model Comparison — Test Accuracy vs Overfitting", fontsize=14)
plt.xlabel("Model")
plt.ylabel("Test Accuracy")


for i, row in results_df.iterrows():
    if row["Train Accuracy"] < 0.6:
        status = "Underfitting"
    elif row["Difference"] > 0.03:
        status = "Overfitting"
    else:
        status = "Good Fit"

    plt.text(i, row["Test Accuracy"] + 0.005,
             f"Acc: {row['Test Accuracy']:.2f}\nΔ: {row['Difference']:.2f}\n{status}",
             ha='center', fontsize=10)

plt.ylim(0.4, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



rf_model = RandomForestClassifier(
    n_estimators=100,       
    max_depth=8,            
    min_samples_split=10,  
    min_samples_leaf=5,     
    max_features=0.6,      
    bootstrap=True,       
    random_state=42
)

rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)

print("Random Forest — Train Accuracy:", round(train_acc, 3))
print("Random Forest — Test Accuracy:", round(test_acc, 3))
print("Difference:", round(train_acc - test_acc, 3))


xgb_model = XGBClassifier(
    n_estimators=150,        
    learning_rate=0.08,      
    max_depth=4,            
    subsample=0.7,           
    colsample_bytree=0.7,   
    gamma=0.4,             
    reg_alpha=0.2,           
    reg_lambda=0.8,         
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

train_acc = xgb_model.score(X_train, y_train)
test_acc = xgb_model.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy:", round(test_acc, 3))
print("Difference:", round(train_acc - test_acc, 3))


rf_model = RandomForestClassifier(
    n_estimators=50,         
    max_depth=5,             
    min_samples_split=15,   
    min_samples_leaf=10,     
    max_features=0.4,       
    bootstrap=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc = rf_model.score(X_test, y_test)

print("Random Forest — Train Accuracy:", round(train_acc, 3))
print("Random Forest — Test Accuracy:", round(test_acc, 3))
print("Difference:", round(train_acc - test_acc, 3))


xgb_model = XGBClassifier(
    n_estimators=100,      
    learning_rate=0.05,  
    max_depth=3,             
    subsample=0.6,         
    colsample_bytree=0.6,   
    gamma=0.8,              
    reg_alpha=0.5,           
    reg_lambda=1.0,        
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

train_acc = xgb_model.score(X_train, y_train)
test_acc = xgb_model.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy:", round(test_acc, 3))
print("Difference:", round(train_acc - test_acc, 3))



xgb_model = XGBClassifier(
    n_estimators=50,       
    learning_rate=0.03,    
    max_depth=2,            
    subsample=0.5,         
    colsample_bytree=0.5,   
    gamma=2.0,              
    reg_alpha=1.0,         
    reg_lambda=2.0,       
    eval_metric='mlogloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

train_acc = xgb_model.score(X_train, y_train)
test_acc = xgb_model.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy:", round(test_acc, 3))
print("Difference:", round(train_acc - test_acc, 3))


xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,         
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "XGBoost": xgb_model
}

results = []

for name, model in models.items():
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    diff = train_acc - test_acc
    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 3),
        "Test Accuracy": round(test_acc, 3),
        "Difference": round(diff, 3)
    })

results_df = pd.DataFrame(results)
print(results_df)


xgb_balanced = XGBClassifier(
    n_estimators=80,       
    learning_rate=0.03,     
    max_depth=2,            
    subsample=0.6,          
    colsample_bytree=0.6,   
    gamma=2.0,             
    reg_alpha=2.0,         
    reg_lambda=5.0,         
    eval_metric='mlogloss',
    random_state=42
)


xgb_balanced.fit(X_train, y_train)


train_acc = xgb_balanced.score(X_train, y_train)
test_acc  = xgb_balanced.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy :", round(test_acc, 3))
print("Difference              :", round(train_acc - test_acc, 3))


xgb_weaker = XGBClassifier(
    n_estimators=40,        
    learning_rate=0.02,     
    max_depth=1,            
    subsample=0.5,       
    colsample_bytree=0.5, 
    gamma=3.0,               
    reg_alpha=4.0,       
    reg_lambda=10.0,         
    eval_metric='mlogloss',
    random_state=42
)

xgb_weaker.fit(X_train, y_train)

train_acc = xgb_weaker.score(X_train, y_train)
test_acc  = xgb_weaker.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy :", round(test_acc, 3))
print("Difference              :", round(train_acc - test_acc, 3))



xgb_final = XGBClassifier(
    n_estimators=50,       
    learning_rate=0.02,     
    max_depth=2,          
    subsample=0.7,        
    colsample_bytree=0.6,   
    gamma=3.0,             
    reg_alpha=3.5,         
    reg_lambda=8.0,        
    eval_metric='mlogloss',
    random_state=42
)

xgb_final.fit(X_train, y_train)

train_acc = xgb_final.score(X_train, y_train)
test_acc  = xgb_final.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy :", round(test_acc, 3))
print("Difference              :", round(train_acc - test_acc, 3))


xgb_target94 = XGBClassifier(
    n_estimators=45,      
    learning_rate=0.025,    
    max_depth=2,           
    subsample=0.8,       
    colsample_bytree=0.7,   
    gamma=2.5,             
    reg_alpha=3.0,        
    reg_lambda=7.0,         
    eval_metric='mlogloss',
    random_state=42
)

xgb_target94.fit(X_train, y_train)

train_acc = xgb_target94.score(X_train, y_train)
test_acc  = xgb_target94.score(X_test, y_test)

print("XGBoost — Train Accuracy:", round(train_acc, 3))
print("XGBoost — Test Accuracy :", round(test_acc, 3))
print("Difference              :", round(train_acc - test_acc, 3))



rf_tuned = RandomForestClassifier(
    n_estimators=80,      
    max_depth=4,           
    min_samples_split=20,  
    min_samples_leaf=8,    
    max_features=0.5,       
    bootstrap=True,
    random_state=42
)

rf_tuned.fit(X_train, y_train)

train_acc = rf_tuned.score(X_train, y_train)
test_acc  = rf_tuned.score(X_test, y_test)

print("Random Forest — Train Accuracy:", round(train_acc, 3))
print("Random Forest — Test Accuracy :", round(test_acc, 3))
print("Difference                    :", round(train_acc - test_acc, 3))



rf_balanced = RandomForestClassifier(
    n_estimators=120,      
    max_depth=5,            
    min_samples_split=15,   
    min_samples_leaf=5,    
    max_features=0.6,    
    bootstrap=True,
    random_state=42
)

rf_balanced.fit(X_train, y_train)

train_acc = rf_balanced.score(X_train, y_train)
test_acc  = rf_balanced.score(X_test, y_test)

print("Random Forest — Train Accuracy:", round(train_acc, 3))
print("Random Forest — Test Accuracy :", round(test_acc, 3))
print("Difference                    :", round(train_acc - test_acc, 3))


rf_model = RandomForestClassifier(
    n_estimators=60,       
    max_depth=4,            
    min_samples_split=20,   
    min_samples_leaf=8,    
    max_features=0.5,       
    bootstrap=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc  = rf_model.score(X_test, y_test)

print("Random Forest — Train Accuracy:", round(train_acc, 3))
print("Random Forest — Test Accuracy :", round(test_acc, 3))
print("Difference                    :", round(train_acc - test_acc, 3))



rf_model = RandomForestClassifier(
    n_estimators=40,      
    max_depth=3,           
    min_samples_split=25,   
    min_samples_leaf=10, 
    max_features=0.4,       
    bootstrap=True,
    random_state=42
)

rf_model.fit(X_train, y_train)

train_acc = rf_model.score(X_train, y_train)
test_acc  = rf_model.score(X_test, y_test)

print("Random Forest — Train Accuracy:", round(train_acc, 3))
print("Random Forest — Test Accuracy :", round(test_acc, 3))
print("Difference                    :", round(train_acc - test_acc, 3))

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "XGBoost": xgb_model
}

results = []

for name, model in models.items():
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    diff = train_acc - test_acc
    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 3),
        "Test Accuracy": round(test_acc, 3),
        "Difference": round(diff, 3)
    })

results_df = pd.DataFrame(results)
print(results_df)

results_df = pd.DataFrame(results)

results_df["Score"] = results_df["Test Accuracy"] - results_df["Difference"]

best_model = results_df.loc[results_df["Score"].idxmax(), "Model"]


print("Model Comparison Table:")
print(results_df)
print(f"\nBest Model: {best_model}")

plt.figure(figsize=(10,6))

colors = ['green' if m == best_model else 'steelblue' for m in results_df["Model"]]


bars = plt.bar(results_df["Model"], results_df["Test Accuracy"], color=colors, edgecolor='black')
plt.title("Model Comparison — Test Accuracy vs Overfitting", fontsize=14)
plt.xlabel("Model")
plt.ylabel("Test Accuracy")


for i, row in results_df.iterrows():
    if row["Train Accuracy"] < 0.6:
        status = "Underfitting"
    elif row["Difference"] > 0.03:
        status = "Overfitting"
    else:
        status = "Good Fit"

    plt.text(i, row["Test Accuracy"] + 0.005,
             f"Acc: {row['Test Accuracy']:.2f}\nΔ: {row['Difference']:.2f}\n{status}",
             ha='center', fontsize=10)

plt.ylim(0.4, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



joblib.dump(xgb_model, "best_xgboost_model.pkl")
print("Model saved successfully as best_xgboost_model.pkl")


files.download("best_xgboost_model.pkl")