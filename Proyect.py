import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6c3fc5;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .prediction-box {
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin-top: 1rem;
    }
    .team-footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
        padding-top: 1rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #444;
        border-left: 4px solid #6c3fc5;
        padding-left: 0.6rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────
SPECIES_COLORS = {
    "Iris-setosa":     "#FF6B6B",
    "Iris-versicolor": "#4ECDC4",
    "Iris-virginica":  "#45B7D1",
}

@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    df = df.drop(columns=["Id"], errors="ignore")
    return df

@st.cache_resource
def train_model(df):
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall":    recall_score(y_test, y_pred, average="weighted"),
        "f1":        f1_score(y_test, y_pred, average="weighted"),
    }
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    importances = pd.Series(clf.feature_importances_,
                            index=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"])
    return clf, metrics, cm, importances, X_test, y_test, y_pred

df = load_data()
clf, metrics, cm, importances, X_test, y_test, y_pred = train_model(df)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🌸 Iris Species Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data Mining Final Project · Universidad de la Costa</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR – PREDICTION PANEL
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Predict a New Sample")
    st.markdown("Enter measurements to classify a flower:")

    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sw = st.slider("Sepal Width (cm)",  1.5, 4.5, 3.0, 0.1)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    sample = np.array([[sl, sw, pl, pw]])
    prediction = clf.predict(sample)[0]
    proba = clf.predict_proba(sample)[0]
    proba_df = pd.DataFrame({
        "Species":     clf.classes_,
        "Probability": proba
    }).sort_values("Probability", ascending=False)

    color = SPECIES_COLORS.get(prediction, "#6c3fc5")
    st.markdown(
        f'<div class="prediction-box" style="background:{color};">'
        f'🌺 {prediction}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Confidence per class:**")
    for _, row in proba_df.iterrows():
        st.progress(float(row["Probability"]),
                    text=f"{row['Species']}: {row['Probability']*100:.1f}%")

    st.markdown("---")
    st.markdown("**Team members:**")
    st.markdown("- Carlos Andres Barreto Castilla")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Metrics",
    "🔬 3D Visualization",
    "📈 Data Exploration",
    "🌳 Model Insights",
])

# ── TAB 1: METRICS ────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Model Performance (Random Forest · 80/20 split)</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metric_data = [
        ("Accuracy",  metrics["accuracy"],  c1),
        ("Precision", metrics["precision"], c2),
        ("Recall",    metrics["recall"],    c3),
        ("F1-Score",  metrics["f1"],        c4),
    ]
    for label, value, col in metric_data:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{value*100:.2f}%</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    fig_cm = px.imshow(
        cm,
        x=list(clf.classes_),
        y=list(clf.classes_),
        text_auto=True,
        color_continuous_scale="Purples",
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix",
    )
    fig_cm.update_layout(height=380)
    st.plotly_chart(fig_cm, use_container_width=True)

# ── TAB 2: 3D SCATTER ─────────────────────────
with tab2:
    st.markdown('<div class="section-title">3D Scatter Plot – Petal vs Sepal dimensions</div>',
                unsafe_allow_html=True)

    axis_opts = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    col_x, col_y, col_z = st.columns(3)
    axis_x = col_x.selectbox("X axis", axis_opts, index=2)
    axis_y = col_y.selectbox("Y axis", axis_opts, index=3)
    axis_z = col_z.selectbox("Z axis", axis_opts, index=0)

    plot_df = df.copy()
    fig3d = px.scatter_3d(
        plot_df, x=axis_x, y=axis_y, z=axis_z,
        color="Species",
        color_discrete_map=SPECIES_COLORS,
        opacity=0.75,
        title="Iris Dataset – 3D Scatter",
        height=560,
    )

    # Add the new sample as a distinct marker
    fig3d.add_trace(go.Scatter3d(
        x=[sample[0][axis_opts.index(axis_x)]],
        y=[sample[0][axis_opts.index(axis_y)]],
        z=[sample[0][axis_opts.index(axis_z)]],
        mode="markers",
        marker=dict(size=12, color="gold", symbol="diamond",
                    line=dict(color="black", width=2)),
        name=f"New sample → {prediction}",
    ))

    fig3d.update_layout(legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig3d, use_container_width=True)

# ── TAB 3: DATA EXPLORATION ───────────────────
with tab3:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_a.metric("Total samples", len(df))
    col_b.metric("Features", 4)
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown('<div class="section-title">Feature Distributions by Species</div>',
                unsafe_allow_html=True)
    feature = st.selectbox("Select feature", axis_opts)
    fig_hist = px.histogram(
        df, x=feature, color="Species",
        color_discrete_map=SPECIES_COLORS,
        barmode="overlay", opacity=0.7,
        nbins=25, title=f"Distribution of {feature}",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-title">Scatter Matrix (Pair Plot)</div>',
                unsafe_allow_html=True)
    fig_matrix = px.scatter_matrix(
        df,
        dimensions=axis_opts,
        color="Species",
        color_discrete_map=SPECIES_COLORS,
        opacity=0.6,
        height=600,
        title="Scatter Matrix – all feature pairs",
    )
    fig_matrix.update_traces(diagonal_visible=False)
    st.plotly_chart(fig_matrix, use_container_width=True)

# ── TAB 4: MODEL INSIGHTS ─────────────────────
with tab4:
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    fig_imp = px.bar(
        importances.sort_values(ascending=True).reset_index(),
        x=0, y="index",
        orientation="h",
        labels={"0": "Importance", "index": "Feature"},
        color=0,
        color_continuous_scale="Purples",
        title="Random Forest – Feature Importance",
    )
    fig_imp.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-title">Workflow Summary</div>', unsafe_allow_html=True)
    st.markdown("""
| Step | Action |
|------|--------|
| 1. Data Understanding | Load CSV, inspect shape, value counts, descriptive stats |
| 2. Preprocessing | Drop `Id` column; check for nulls (none found) |
| 3. Modeling | Random Forest (100 trees, random_state=42) |
| 4. Validation | 80/20 stratified train-test split |
| 5. Evaluation | Accuracy, Precision, Recall, F1 (weighted) |
| 6. Dashboard | Interactive Streamlit app with 3D scatter & prediction panel |
""")

    st.markdown('<div class="section-title">Why Random Forest?</div>', unsafe_allow_html=True)
    st.info(
        "Random Forest was chosen because it handles multiclass problems natively, "
        "is robust to small datasets, provides feature importance scores, requires "
        "minimal hyperparameter tuning, and consistently achieves >95 % accuracy on the "
        "Iris benchmark."
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="team-footer">'
    'Universidad de la Costa · Department of Computer Science and Electronics · '
    'Data Mining · Prof. José Escorcia-Gutierrez, Ph.D.'
    '</div>',
    unsafe_allow_html=True,
)
