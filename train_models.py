import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# الموديلز اللي هنقارن بينها
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Starting Model Training Phase...\n")

# 1. بنحمل الداتا النظيفة اللي جهزناها (30 عمود)
df = pd.read_csv('cleaned_blood_cell_data.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. بنفصل الـ Features عن العمود المراد توقعه (anomaly_label)
X = df.drop(columns=['anomaly_label'])
y = df['anomaly_label']

# 3. بنقسم الداتا لـ 80% تدريب و 20% اختبار مع الحفاظ على نسبة الفئات متوازنة
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. بنعمل Standard Scaling عشان الموديلات اللي بتعتمد على المسافات زي KNN و Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# بنحفظ الـ Scaler عشان نستخدمه بعدين في تطبيق Streamlit
joblib.dump(scaler, 'scaler.pkl')

# 5. بنحدد الموديلات الخمسة ونعرف مين فيهم محتاج Scaling ومين لأ(dectionary)
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),
    "KNN": (KNeighborsClassifier(n_neighbors=5), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
    "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), False)
}

# 6. بنمر على كل موديل، ندربه، ونحسب الـ Accuracy بتاعته
results = []
best_acc = 0.0
best_model_name = ""
best_model_obj = None

for name, (model, needs_scaling) in models.items():
    # بنختار الداتا المتظبطة بالـ Scaling للموديلات اللي محتاجاها بس
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test
    
    # تدريب الموديل
    model.fit(X_tr, y_train)
    
    # التوقع وحساب الدقة
    preds = model.predict(X_te)
    acc = accuracy_score(y_test, preds)

    # 64: حساب الدقة
    acc = accuracy_score(y_test, preds)
    
    # اضيفي السطرين هنا بالظبط:
    print(f"\n--- {name} Classification Report ---")
    print(classification_report(y_test, preds))
    
    # 66: حفظ النتائج
    results.append({"Model": name, "Accuracy": f"{acc * 100:.2f}%"})
    
    results.append({"Model": name, "Accuracy": f"{acc * 100:.2f}%"})
    
    # بنحفظ الموديل اللي بيجيب أعلى دقة
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model_obj = model

# 7. بنطبع جدول المقارنة بين الموديلات
results_df = pd.DataFrame(results)
print("\n--- Model Comparison Results ---")
print(results_df.to_string(index=False))

print(f"\nBest Model: {best_model_name} with Accuracy: {best_acc * 100:.2f}%")

# 8. بنحفظ أفضل موديل في ملف best_model.pkl عشان نربطه بـ Streamlit
joblib.dump(best_model_obj, 'best_model.pkl')
print(f"Saved '{best_model_name}' to 'best_model.pkl' successfully!")

print(f"Training Accuracy: {best_model_obj.score(X_train, y_train) * 100:.2f}%")
print(f"Testing Accuracy: {best_model_obj.score(X_test, y_test) * 100:.2f}%")