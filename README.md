# ml_asst_2
# Heart Disease Prediction Dashboard

## 1. Problem Statement
Predicting the likelihood of heart disease in patients based on 11 clinical features using 6 different Machine Learning classification models.

## 2. Dataset Description
- **Source:** Heart Failure Prediction Dataset (Kaggle/UCI)
- **Instances:** 918
- **Features:** 11 (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)
- **Target:** HeartDisease (Binary: 1 = Disease, 0 = Normal)

## 3. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8478260869565217| 0.9008374802767326 | 0.9072164948453608 | 0.822429906542056 | 0.8627450980392157 | 0.6971361089405994 |
| Decision Tree | 0.8152173913043478 | 0.82109479305741 | 0.8842105263157894 | 0.7850467289719626 | 0.8316831683168316 | 0.633933502355767 |
| kNN | 0.8478260869565217 | 0.9222599830076464 | 0.9072164948453608 |0.822429906542056 | 0.8627450980392157 | 0.6971361089405994 |
| Naive Bayes | 0.842391304347826 | 0.9089695351377595 | 0.8823529411764706 | 0.8411214953271028 | 0.861244019138756 | 0.6801373270763298 |
| Random Forest | 0.8804347826086957 | 0.9422259983007646 | 0.9207920792079208 | 0.8691588785046729 | 0.8942307692307693 | 0.7586616066244436 |
| XGBoost | 0.8695652173913043 | 0.9366427964558807 | 0.9191919191919192 | 0.8504672897196262 | 0.883495145631068 | 0.738722650503732 |



## 4. Observations
- **Best Model:** Random Forest achieved the highest Accuracy and MCC.
- **Robustness:** Logistic Regression and Naive Bayes provided very similar and stable results.
- **Ensemble Power:** Boosting (XGBoost) and Bagging (Random Forest) clearly outperformed single-tree models.

