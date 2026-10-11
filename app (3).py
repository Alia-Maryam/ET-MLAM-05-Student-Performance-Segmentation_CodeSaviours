import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Performance Segmentation", page_icon="🎓", layout="wide")

FEATURES = ["completion_rate", "avg_submission_time_days", "quiz_score"]
NICE = {
    "completion_rate": "Completion Rate (%)",
    "avg_submission_time_days": "Avg Submission Time (days)",
    "quiz_score": "Quiz Score",
}

LABELS_3 = {
    "top": "High performers — fast submitters, high completion, strong scores",
    "mid": "Average — steady but inconsistent performance",
    "low": "At-risk — low completion, slow submission, weak scores",
}
ACTIONS_3 = {
    "top": "Offer advanced/stretch material or peer-mentoring roles to keep them engaged.",
    "mid": "Send targeted reminders before deadlines and offer optional office hours for weak topics.",
    "low": "Schedule a 1-on-1 check-in and set smaller intermediate deadlines to rebuild momentum.",
}


# ---------- data ----------
@st.cache_data
def sample_data(seed: int = 42) -> pd.DataFrame:
    """Synthetic data shaped like the notebook's dataset (3 groups, 48 students)."""
    rng = np.random.default_rng(seed)
    groups = [(16, 91, 1.0, 85), (13, 35, 6.8, 46), (19, 66, 3.0, 62)]
    rows = []
    for n, c, t, q in groups:
        rows.append(pd.DataFrame({
            "completion_rate": np.clip(rng.normal(c, 7, n), 0, 100).round(1),
            "avg_submission_time_days": np.clip(rng.normal(t, 1.2, n), 0.1, None).round(2),
            "quiz_score": np.clip(rng.normal(q, 10, n), 0, 100).round(1),
        }))
    df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "student_id", [f"S{i + 1:03d}" for i in range(len(df))])
    return df


@st.cache_data
def run_elbow(X_scaled: np.ndarray, max_k: int):
    ks = list(range(1, max_k + 1))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in ks]
    return ks, inertias


def label_clusters(df: pd.DataFrame, k: int):
    """Rank clusters by an overall performance score so labels don't depend on arbitrary cluster ids."""
    means = df.groupby("cluster")[FEATURES].mean()
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    score = z["completion_rate"] + z["quiz_score"] - z["avg_submission_time_days"]
    order = score.sort_values(ascending=False).index.tolist()  # best -> worst

    labels, actions = {}, {}
    for rank, cid in enumerate(order):
        if k == 3:
            key = ["top", "mid", "low"][rank]
            labels[cid], actions[cid] = LABELS_3[key], ACTIONS_3[key]
        else:
            tag = "Top" if rank == 0 else ("Lowest" if rank == k - 1 else f"Tier {rank + 1}")
            labels[cid] = f"{tag} segment (rank {rank + 1} of {k})"
            if rank == 0:
                actions[cid] = ACTIONS_3["top"]
            elif rank == k - 1:
                actions[cid] = ACTIONS_3["low"]
            else:
                actions[cid] = ACTIONS_3["mid"]
    return labels, actions


# ---------- sidebar ----------
st.sidebar.title("⚙️ Settings")
upload = st.sidebar.file_uploader("Upload student_performance_dataset.csv", type="csv")
use_sample = st.sidebar.checkbox("Use sample data instead", value=upload is None)

if upload is not None and not use_sample:
    df = pd.read_csv(upload)
else:
    df = sample_data()
    st.sidebar.info("Using synthetic sample data. Upload your CSV to analyse real students.")

missing = [c for c in ["student_id"] + FEATURES if c not in df.columns]
if missing:
    st.error(f"CSV is missing required column(s): {', '.join(missing)}")
    st.stop()

df = df.dropna(subset=FEATURES).reset_index(drop=True)
if len(df) < 8:
    st.error("Need at least 8 students with complete data.")
    st.stop()

chosen_k = st.sidebar.slider("Number of clusters (k)", 2, 7, 3)
st.sidebar.caption("The notebook uses k = 3. Check the Elbow tab to justify other values.")

# ---------- model ----------
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
labels, actions = label_clusters(df, chosen_k)
df["segment_label"] = df["cluster"].map(labels)
dist = kmeans.transform(X_scaled)
df["dist_to_own_cluster"] = dist[np.arange(len(df)), df["cluster"]]

# ---------- header ----------
st.title("🎓 Student Performance Segmentation")
st.write("K-Means clustering of students by completion rate, submission speed and quiz score.")

m1, m2, m3 = st.columns(3)
m1.metric("Students", len(df))
m2.metric("Clusters (k)", chosen_k)
m3.metric("Inertia", f"{kmeans.inertia_:.1f}")

tab_data, tab_elbow, tab_clusters, tab_segments, tab_border = st.tabs(
    ["📄 Data", "📉 Elbow", "🔵 Clusters", "🧭 Segments & Actions", "🔍 Borderline Students"]
)

color_seq = px.colors.qualitative.Set2
df["cluster_str"] = df["cluster"].astype(str)

# ---------- data ----------
with tab_data:
    st.subheader("Dataset")
    st.dataframe(df[["student_id"] + FEATURES], use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)
    with st.expander("Scaling example (first student)"):
        st.write("Before scaling:", dict(zip(FEATURES, X[0].round(2))))
        st.write("After scaling:", dict(zip(FEATURES, X_scaled[0].round(3))))

# ---------- elbow ----------
with tab_elbow:
    st.subheader("Elbow method")
    ks, inertias = run_elbow(X_scaled, 7)
    fig = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers"))
    fig.add_vline(x=chosen_k, line_dash="dash", line_color="red",
                  annotation_text=f"chosen k = {chosen_k}")
    fig.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Inertia",
                      xaxis=dict(dtick=1), height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Look for the 'elbow' where adding clusters stops reducing inertia sharply.")

# ---------- clusters ----------
with tab_clusters:
    st.subheader("Cluster scatter plots")
    pairs = [
        ("completion_rate", "quiz_score"),
        ("avg_submission_time_days", "quiz_score"),
        ("completion_rate", "avg_submission_time_days"),
    ]
    cols = st.columns(3)
    for col, (x, y) in zip(cols, pairs):
        f = px.scatter(df, x=x, y=y, color="cluster_str", hover_data=["student_id"],
                       labels={x: NICE[x], y: NICE[y], "cluster_str": "Cluster"},
                       color_discrete_sequence=color_seq, title=f"{NICE[x]} vs {NICE[y]}")
        f.update_traces(marker=dict(size=10))
        f.update_layout(height=400, legend_title_text="Cluster")
        col.plotly_chart(f, use_container_width=True)

    st.subheader("3D view")
    f3 = px.scatter_3d(df, x=FEATURES[0], y=FEATURES[1], z=FEATURES[2], color="cluster_str",
                       hover_data=["student_id"], labels=NICE,
                       color_discrete_sequence=color_seq)
    f3.update_layout(height=550)
    st.plotly_chart(f3, use_container_width=True)

# ---------- segments ----------
with tab_segments:
    st.subheader("Cluster summary")
    summary = df.groupby("cluster")[FEATURES].mean().round(1)
    summary["num_students"] = df.groupby("cluster").size()
    summary["segment"] = summary.index.map(labels)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Recommended mentoring actions")
    for cid in summary.sort_values("quiz_score", ascending=False).index:
        with st.container(border=True):
            st.markdown(f"**Cluster {cid} — {labels[cid]}** ({int(summary.loc[cid, 'num_students'])} students)")
            st.write(f"➡️ {actions[cid]}")

    st.subheader("Look up a student")
    sid = st.selectbox("Student ID", df["student_id"])
    row = df[df["student_id"] == sid].iloc[0]
    st.write(f"**Segment:** {row['segment_label']}")
    st.write(f"**Suggested action:** {actions[row['cluster']]}")

    st.download_button(
        "⬇️ Download results as CSV",
        df[["student_id"] + FEATURES + ["cluster", "segment_label", "dist_to_own_cluster"]]
        .to_csv(index=False).encode("utf-8"),
        file_name="student_segments.csv",
        mime="text/csv",
    )

# ---------- borderline ----------
with tab_border:
    st.subheader("Students farthest from their cluster center")
    st.write("These may be borderline cases that don't fit their segment cleanly.")
    n = st.slider("How many to show", 3, min(20, len(df)), 6)
    out = (df.sort_values("dist_to_own_cluster", ascending=False)
             [["student_id"] + FEATURES + ["cluster", "dist_to_own_cluster"]].head(n))
    st.dataframe(out, use_container_width=True)
