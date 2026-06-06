"""
PCA Dimensionality Reduction — CC General Dataset
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json, os, warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA Analysis | CC General",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "CC_GENERAL.csv"))
    df.drop(columns=["CUST_ID"], inplace=True)
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())
    return df

@st.cache_data
def run_pca(n_components):
    df = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_scaled, X_pca, pca, pca_full

@st.cache_data
def load_metrics():
    mp = os.path.join(os.path.dirname(__file__), "models", "pca_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return {}

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/combo-chart--v1.png", width=60)
st.sidebar.title("⚙️ PCA Controls")
st.sidebar.markdown("---")

n_components = st.sidebar.slider("Number of PCA Components", 2, 17, 10)
n_clusters = st.sidebar.slider("KMeans Clusters (k)", 2, 8, 3)
color_scheme = st.sidebar.selectbox("Colour Palette", ["Set1", "Vivid", "Plasma", "Turbo"])
show_loadings = st.sidebar.checkbox("Show Feature Loadings (Biplot)", True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Dataset Info")
st.sidebar.info("**8,950** customers\n\n**17** numeric features\n\nCC General Credit Card")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("📊 PCA — Dimensionality Reduction")
st.markdown("**Credit Card Customer Dataset** · Principal Component Analysis + KMeans Clustering")
st.markdown("---")

df = load_data()
metrics = load_metrics()
X_scaled, X_pca, pca, pca_full = run_pca(n_components)

# KMeans
km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_pca)
sil = silhouette_score(X_pca, cluster_labels)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# ── KPI row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Original Features", "17")
c3.metric("PCA Components", n_components)
c4.metric("Variance Retained", f"{cumulative[n_components-1]*100:.1f}%")
c5.metric("Silhouette Score", f"{sil:.4f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Variance Analysis", "🗺️ 2D Projection", "🧊 3D Projection", "🔬 Loadings", "📋 Data Explorer"]
)

# ── Tab 1 · Variance ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Explained Variance Analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig_scree = go.Figure()
        fig_scree.add_bar(
            x=[f"PC{i+1}" for i in range(17)],
            y=[round(v*100,2) for v in explained],
            marker_color=["#6366f1" if i < n_components else "#d1d5db" for i in range(17)],
            name="Individual",
        )
        fig_scree.add_scatter(
            x=[f"PC{i+1}" for i in range(17)],
            y=[round(v*100,2) for v in cumulative],
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="#f59e0b", width=2),
            yaxis="y2",
        )
        fig_scree.update_layout(
            title="Scree Plot",
            xaxis_title="Principal Component",
            yaxis_title="Individual Variance (%)",
            yaxis2=dict(title="Cumulative Variance (%)", overlaying="y", side="right"),
            legend=dict(x=0.6, y=0.3),
            height=400,
        )
        st.plotly_chart(fig_scree, use_container_width=True)

    with col2:
        thresh_90 = metrics.get("n_components_90pct", 10)
        thresh_95 = metrics.get("n_components_95pct", 12)
        fig_cum = go.Figure()
        fig_cum.add_scatter(
            x=list(range(1, 18)),
            y=[round(v*100,2) for v in cumulative],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=8),
            name="Cumulative Variance",
        )
        fig_cum.add_hline(y=90, line_dash="dash", line_color="#10b981", annotation_text="90%")
        fig_cum.add_hline(y=95, line_dash="dash", line_color="#f59e0b", annotation_text="95%")
        fig_cum.add_vline(x=thresh_90, line_dash="dot", line_color="#10b981",
                          annotation_text=f"n={thresh_90}", annotation_position="top right")
        fig_cum.add_vline(x=thresh_95, line_dash="dot", line_color="#f59e0b",
                          annotation_text=f"n={thresh_95}", annotation_position="top right")
        fig_cum.update_layout(
            title="Cumulative Explained Variance",
            xaxis_title="Number of Components",
            yaxis_title="Cumulative Variance (%)",
            height=400,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

    st.info(f"✅ **{thresh_90} components** explain ≥90% of variance | **{thresh_95} components** explain ≥95% of variance")

    # Variance table
    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(17)],
        "Explained Variance (%)": [round(v*100,2) for v in explained],
        "Cumulative Variance (%)": [round(v*100,2) for v in cumulative],
        "Selected": ["✅" if i < n_components else "—" for i in range(17)],
    })
    st.dataframe(var_df, use_container_width=True, hide_index=True)

# ── Tab 2 · 2D ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("2D PCA Projection with KMeans Clusters")
    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
        "Cluster": [f"Cluster {c}" for c in cluster_labels],
    })
    fig2d = px.scatter(
        plot_df, x="PC1", y="PC2", color="Cluster",
        color_discrete_sequence=px.colors.qualitative.__dict__.get(color_scheme, px.colors.qualitative.Set1),
        title=f"PCA 2D Scatter (k={n_clusters}, Silhouette={sil:.4f})",
        opacity=0.65, height=550,
        labels={"PC1": f"PC1 ({explained[0]*100:.1f}%)", "PC2": f"PC2 ({explained[1]*100:.1f}%)"},
    )
    fig2d.update_traces(marker=dict(size=4))
    st.plotly_chart(fig2d, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        dist = pd.Series(cluster_labels).value_counts().sort_index()
        fig_bar = px.bar(
            x=[f"Cluster {i}" for i in dist.index], y=dist.values,
            color=[f"Cluster {i}" for i in dist.index],
            color_discrete_sequence=px.colors.qualitative.__dict__.get(color_scheme, px.colors.qualitative.Set1),
            title="Cluster Distribution", labels={"x": "Cluster", "y": "Count"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        sil_data = {}
        for k in range(2, 9):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km_tmp.fit_predict(X_pca)
            sil_data[k] = round(silhouette_score(X_pca, lbl), 4)
        fig_sil = px.line(
            x=list(sil_data.keys()), y=list(sil_data.values()),
            markers=True, title="Silhouette Score vs k",
            labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"},
        )
        fig_sil.add_vline(x=max(sil_data, key=sil_data.get), line_dash="dash", line_color="#6366f1")
        st.plotly_chart(fig_sil, use_container_width=True)

# ── Tab 3 · 3D ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("3D PCA Projection")
    if n_components >= 3:
        plot3d = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "PC3": X_pca[:, 2],
            "Cluster": [f"Cluster {c}" for c in cluster_labels],
        })
        fig3d = px.scatter_3d(
            plot3d, x="PC1", y="PC2", z="PC3", color="Cluster",
            color_discrete_sequence=px.colors.qualitative.__dict__.get(color_scheme, px.colors.qualitative.Set1),
            title="PCA 3D Scatter",
            opacity=0.6, height=650,
            labels={
                "PC1": f"PC1 ({explained[0]*100:.1f}%)",
                "PC2": f"PC2 ({explained[1]*100:.1f}%)",
                "PC3": f"PC3 ({explained[2]*100:.1f}%)",
            },
        )
        fig3d.update_traces(marker=dict(size=2.5))
        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.warning("Set PCA components ≥ 3 in the sidebar for 3D visualization.")

# ── Tab 4 · Loadings ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("Feature Loadings (Component Matrix)")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    fig_heat = px.imshow(
        loadings,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="PCA Loadings Heatmap",
        zmin=-1, zmax=1,
        height=500,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    if show_loadings:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PC1 Top Loadings**")
            pc1 = loadings["PC1"].abs().sort_values(ascending=False)
            fig_pc1 = px.bar(
                x=pc1.values[:8], y=pc1.index[:8], orientation="h",
                color=loadings.loc[pc1.index[:8], "PC1"],
                color_continuous_scale="RdBu_r", range_color=[-1, 1],
                title="PC1 Feature Importance",
            )
            st.plotly_chart(fig_pc1, use_container_width=True)
        with col2:
            st.markdown("**PC2 Top Loadings**")
            pc2 = loadings["PC2"].abs().sort_values(ascending=False)
            fig_pc2 = px.bar(
                x=pc2.values[:8], y=pc2.index[:8], orientation="h",
                color=loadings.loc[pc2.index[:8], "PC2"],
                color_continuous_scale="RdBu_r", range_color=[-1, 1],
                title="PC2 Feature Importance",
            )
            st.plotly_chart(fig_pc2, use_container_width=True)

    st.subheader("Loadings Table")
    st.dataframe(loadings.round(4).style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                 use_container_width=True)

# ── Tab 5 · Data ─────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Dataset Explorer")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw Dataset (first 200 rows)**")
        st.dataframe(df.head(200), use_container_width=True, height=350)
    with col2:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(4), use_container_width=True, height=350)

    st.markdown("**Feature Distributions**")
    feat = st.selectbox("Select Feature", df.columns.tolist())
    fig_hist = px.histogram(df, x=feat, nbins=60, color_discrete_sequence=["#6366f1"],
                            title=f"Distribution of {feat}", marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Correlation heatmap
    st.markdown("**Correlation Matrix**")
    corr = df.corr()
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto",
                         title="Feature Correlation Matrix", zmin=-1, zmax=1, height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
