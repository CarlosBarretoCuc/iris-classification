# 🌸 Iris Species Classification

**Data Mining Final Project – Universidad de la Costa**  
Department of Computer Science and Electronics  
Prof. José Escorcia-Gutierrez, Ph.D.

---

## 👥 Team Members

- Carlos Andres Barreto Castilla

---

## 📌 Purpose

This project applies a complete data mining pipeline to classify Iris flowers into three
species (*Iris setosa*, *Iris versicolor*, *Iris virginica*) based on four morphological
measurements: sepal length, sepal width, petal length, and petal width.

---

## 🔬 Methodology / Workflow

| Step | Description |
|------|-------------|
| 1. Data Understanding | Load and inspect the Iris CSV (150 samples, 4 features, 3 classes) |
| 2. Preprocessing | Remove the `Id` column; verify no missing values are present |
| 3. Modeling | Train a **Random Forest** classifier (100 estimators, stratified 80/20 split) |
| 4. Validation | Evaluate on the held-out test set (30 samples) |
| 5. Evaluation | Report Accuracy, Precision, Recall, and F1-Score (weighted average) |
| 6. Dashboard | Interactive Streamlit app for visualisation and live prediction |

### Why Random Forest?

- Handles multiclass problems natively without modification.  
- Robust and accurate on small, clean datasets like Iris.  
- Provides interpretable feature-importance scores.  
- Requires minimal hyperparameter tuning and avoids overfitting via ensemble averaging.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/iris-classification.git
cd iris-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure `Iris.csv` is in the same folder as `Proyect.py`

### 4. Launch the dashboard

```bash
streamlit run Proyect.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Dashboard Features

| Section | Description |
|---------|-------------|
| **Sidebar** | Sliders to input sepal/petal measurements and get an instant prediction with class probabilities |
| **Model Metrics** | Accuracy, Precision, Recall, F1 cards + interactive confusion matrix |
| **3D Visualization** | 3D scatter plot showing the dataset and the new sample's position |
| **Data Exploration** | Descriptive statistics, feature histograms, and scatter matrix |
| **Model Insights** | Feature importance bar chart and workflow summary |

---

## 📁 Repository Structure

```
iris-classification/
├── Proyect.py          # Main Streamlit application
├── Iris.csv            # Dataset
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 📄 Dataset

The classic **Iris dataset** (R.A. Fisher, 1936) – 150 samples, 3 balanced classes, 4 numerical features.  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)
