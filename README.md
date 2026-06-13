# 🚀 Machine Learning Portfolio Projects  
### Toronto Traffic Collision Prediction + Amazon Sales Revenue Forecasting

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## 📌 Overview

This repository contains **two end-to-end machine learning projects**, both built entirely in **Python**, showcasing my skills in:

- Data cleaning & preprocessing  
- Exploratory data analysis (EDA)  
- Feature engineering  
- Model building & evaluation  
- Hyperparameter tuning  
- Deployment-ready ML workflows  
- Business-focused insights  

These projects demonstrate my ability to work with **real-world datasets**, apply **advanced ML techniques**, and deliver **actionable insights**.

---

# 1️⃣ 🚦 Toronto Traffic Collision Involvement Prediction

A machine learning model predicting the **number of individuals involved** in Toronto motor vehicle collisions using **18,957 police-reported incidents** (2006–2023).

### 🔍 ML Highlights
- Cleaned 50+ raw attributes  
- Engineered temporal, environmental, and geospatial features  
- Selected top 14 predictive features using Random Forest importance  
- Trained four models:  
  - **XGBoost (best)**  
  - Random Forest  
  - Neural Network (MLP)  
  - Linear Regression  
- Performed hyperparameter tuning (RandomizedSearchCV)

### 🧠 Best Model Performance (XGBoost)
| Metric | Score |
|-------|-------|
| **MAE** | **0.93** |
| **RMSE** | **1.49** |
| **R²** | **0.77** |

### 📊 Key Insights
- Passenger count, collision type, and temporal features are the strongest predictors  
- Collision involvement is **non-linear**, requiring tree-based models  
- Geospatial hotspots show concentrated high-risk corridors  

---

# 2️⃣ 🛒 Amazon Sales Revenue Prediction (BI Project)

A machine learning pipeline built on **Amazon sales transaction data** to predict **revenue** and uncover business insights.

### 🔍 ML Highlights
- Cleaned and transformed raw Amazon sales data  
- Engineered features:  
  - Seasonal indicators  
  - Lagged revenue variables  
  - Product category encodings  
- Trained multiple regression models:  
  - Linear Regression  
  - Random Forest  
  - XGBoost  
  - Neural Network (MLP)  
- Compared model performance and selected the best model for forecasting

### 📈 Business Insights
- Identified high-revenue product categories  
- Detected seasonal demand patterns  
- Modeled the impact of pricing and shipping cost  
- Built a revenue forecasting model suitable for BI dashboards  

---

## 🛠 Tech Stack (Both Projects)

- **Python 3**  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Matplotlib, Seaborn  
- GeoPandas, Folium (Collision Project)  
- Streamlit (for deployment)  
- Jupyter Notebook  

---

## 👤 Author

**Muluwerk Derebe**  
Business Intelligence Systems Infrastructure (BISI)  
Ottawa, Canada  

Former Assistant Professor of Statistics with expertise in:
- Machine Learning  
- Predictive Analytics  
- Geospatial Modeling  
- Longitudinal Data Analysis


  ---

## 🔬 Methodology & Pipeline for ML project 2

### Stage 1 — Exploratory Data Analysis (EDA)

- Loaded and previewed the dataset using `.head()`, `.tail()`, `.sample()`, `.info()`, `.describe()`
- Identified **18,957 rows × 50 columns** with a mix of 46 object, 3 int64, and 1 float64 columns
- Detected widespread `NaN` values across categorical and numerical fields
- Derived the target variable `NumberOfInvolvedPerson` by counting duplicate `ACCNUM` entries

**Key descriptive findings:**
- Most incidents involve **2–6 individuals** (mean ≈ 4.5, max = 22)
- `FATAL_NO` is dominated by zeros — fatalities are rare but extreme when they occur
- Spearman correlation between `FATAL_NO` and `NumberOfInvolvedPerson` is essentially **-0.02** (no meaningful relationship)

### Stage 2 — Data Cleaning & Preparation

| Step | Action |
|---|---|
| Fill categorical NaN | `fillna("No")` + `replace("None", "No")` |
| Fill numeric NaN | `FATAL_NO` → `fillna(0)` |
| Drop insignificant columns | Removed `_id`, `STREET1`, `STREET2`, `OFFSET`, `INITDIR`, `HOOD_158`, `HOOD_140`, etc. |
| Drop incomplete rows | `dropna(subset=['NumberOfInvolvedPerson'] + features)` |
| Final shape | **18,957 rows × 42 features** (zero nulls remaining) |

### Stage 3 — Visualizations & EDA Insights

- **Distribution of Target Variable:** Right-skewed histogram — most incidents involve 1–5 people with a long tail
- **By Collision Characteristics (ACCLASS, IMPACTYPE, INVTYPE):** Fatal and multi-occupant vehicle incidents show highest mean involvement
- **By Human Factors (INVAGE, INJURY):** Older age groups trend slightly higher; Major/Fatal injuries show extreme outliers
- **By Environmental Conditions (ROAD_CLASS, VISIBILITY, LIGHT, RDSFCOND):** Expressways, fog, dark conditions, and icy roads correlate with slightly higher involvement
- **Temporal Trends:**
  - Yearly: Low and stable (3–4) from 2006–2013, jumped to ~6 from 2014–2019, settled at ~5 post-2019
  - Monthly: Peak involvement in late spring/early summer (months 5–7)
  - Weekly: Slightly higher on mid-week and weekends; lowest on Monday/Wednesday/Thursday

### Stage 4 — Geospatial Analysis

- Parsed JSON geometry column into Shapely `Point` objects using GeoPandas
- Created a static geospatial scatter plot colored by `NumberOfInvolvedPerson` (RdBu_r colormap)
- Built an **interactive Folium risk map** (`risks_map.html`) with circle markers:
  - 🔴 **Red** = High risk (≥10 involved persons)
  - 🔵 **Blue** = Low risk (<10 involved persons)
  - Marker size scales with collision severity

> Geospatial analysis revealed clear **hotspot clustering** in specific Toronto corridors — risk is not evenly distributed across the city.

### Stage 5 — Feature Selection Pipeline

Used two complementary methods to identify the most predictive features:

**i) Random Forest Feature Importance (200 trees)**

| Rank | Feature | Importance |
|---|---|---|
| 1 | PASSENGER | 0.1836 |
| 2 | YEAR | 0.1584 |
| 3 | DAY | 0.1146 |
| 4 | MONTH | 0.0990 |
| 5 | DIVISION | 0.0581 |
| 6 | WEEKDAY | 0.0527 |
| 7 | IMPACTYPE | 0.0422 |
| 8 | DISTRICT | 0.0225 |
| 9 | TRSN_CITY_VEH | 0.0211 |
| 10 | LIGHT | 0.0209 |

**Final 14 selected features:**
`PASSENGER`, `YEAR`, `MONTH`, `WEEKDAY`, `DIVISION`, `INVAGE`, `ACCLOC`, `IMPACTYPE`, `LIGHT`, `DISTRICT`, `ROAD_CLASS`, `TRAFFCTL`, `INJURY`, `VEHTYPE`

### Stage 6 — Machine Learning Modeling

**Preprocessing for modeling:**
- Label encoding for all categorical features
- `StandardScaler` applied for Neural Network (MLP) only
- 80/20 train/test split (`random_state=42`)

**Four models trained and evaluated:**

| Model | Description |
|---|---|
| Linear Regression | Baseline — direct proportional relationships |
| Random Forest | 400 trees, max_depth=10 — captures non-linear interactions |
| XGBoost | 600 estimators, lr=0.05, max_depth=7 — gradient boosting |
| Neural Network (MLP) | (100,50) hidden layers, ReLU, Adam, early stopping |

### Stage 7 — Model Evaluation & Comparison

**Baseline results (before tuning):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ✅ **XGBoost** | **1.0110** | **1.5702** | **0.7458** |
| Random Forest | 1.3419 | 2.0045 | 0.5856 |
| Neural Network (MLP) | 1.4829 | 2.1258 | 0.5340 |
| Linear Regression | 1.8043 | 2.7282 | 0.2325 |

**After hyperparameter tuning (RandomizedSearchCV, 5-fold CV, 30 iterations × 150 fits):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| ✅ **XGBoost** | **0.9316** | **1.4927** | **0.7702** |
| Random Forest | 1.0414 | 1.6545 | 0.7177 |
| Neural Network (MLP) | 1.2637 | 1.9411 | 0.6115 |

**Best XGBoost hyperparameters:**
```python
{
  'n_estimators': 1200,
  'learning_rate': 0.1,
  'max_depth': 9,
  'subsample': 0.8,
  'colsample_bytree': 0.8,
  'gamma': 0.5,
  'reg_alpha': 0.01,
  'reg_lambda': 5
}
```

---

## 📈 Key Findings & Insights

1. **XGBoost is the best model** — lowest RMSE (1.4927) and highest R² (0.7702) after tuning, consistently outperforming all other algorithms
2. **Passenger count, year, and temporal features** (YEAR, MONTH, DAY, WEEKDAY) are the strongest predictors of multi-person involvement
3. **The target variable is right-skewed** — most collisions involve small groups, but rare high-involvement events (10–22 people) create a long tail that challenges all models
4. **Linear models are inadequate** — R² of only 0.23 confirms that collision involvement is driven by non-linear, multi-factor interactions
5. **Geospatial risk is concentrated** — specific Toronto corridors show consistently higher involvement, suggesting targeted interventions would be most effective
6. **Temporal shifts are significant** — a sharp increase in average involvement from 2013 to 2014 suggests a structural change in collision patterns (possibly related to urbanization or reporting changes)

---

## 🏆 Conclusion & Recommendation

> **XGBoost should be selected as the primary model** for operational use, reporting, or deployment in a predictive safety-analytics pipeline. Its gradient-boosting structure captures complex, layered relationships across demographic, spatial, and environmental features that simpler models miss. Random Forest serves as a reliable secondary model or benchmark. Linear Regression and MLP are best reserved for comparison or exploratory purposes.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Geospatial** | GeoPandas, Shapely, Folium |
| **Machine Learning** | Scikit-learn (LinearRegression, RandomForestRegressor, MLPRegressor), XGBoost |
| **Feature Selection** | mutual_info_regression, SelectKBest, RandomForest Importance |
| **Tuning** | RandomizedSearchCV, KFold Cross-Validation |
| **Environment** | Jupyter Notebook, Anaconda |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/toronto-collision-involvement-prediction.git
cd toronto-collision-involvement-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook BI2-project_MD.ipynb
```

**requirements.txt essentials:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
geopandas
shapely
fiona
pyproj
rtree
folium
jupyter
```

---

## 👤 Author

**Muluwerk Derebe**  
*Data Scientist and Business Intelligence Systems Infrastructure expert (BISI)*  
Ottawa, Ontario, Canada

> **Background:** Former Assistant Professor of Statistics with 12 peer-reviewed publications in science-indexed journals, specializing in advanced statistical modeling, longitudinal data analysis, and geospatial methods. Transitioning into Data Analytics and Business Intelligence roles.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muluwerk-derebe-b959243a6/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/muluwerkderebe-StatisticalAnalyst)
[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=flat&logo=vercel&logoColor=white)](https://muluwerkderebe-statisticalanalyst.github.io/Portifolio.io/)

---

## 📄 License

This project is open for **educational and portfolio purposes**. Feel free to explore the code and methodology for learning.

---

<div align="center">

Made with ❤️ to support data-driven insights for improving road safety in Toronto.

</div>
