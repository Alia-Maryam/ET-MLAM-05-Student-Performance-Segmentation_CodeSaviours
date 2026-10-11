import numpy as np
import pandas as pd
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
ACTIONS = {
    "top": "Offer advanced/stretch material or peer-mentoring roles to keep them engaged.",
    "mid": "Send targeted reminders before deadlines and offer optional office hours for weak topics.",
    "low": "Schedule a 1-on-1 check-in and set smaller intermediate deadlines to rebuild momentum.",
}


@st.cache_data
def sample_data(seed: int = 42) -> pd.DataFrame:
    """Synthetic data shaped like the notebook's dataset (3 groups, 48 students)."""
    rng = np.random.default_rng(seed)
    groups = [(16, 91, 1.0, 85), (13, 35, 6.8, 46), (19, 66, 3.0, 62)]
    parts = []
    for n, c, t, q in groups:
        parts.append(pd.DataFrame({
            "completion_rate": np.clip(rng.normal(c, 7, n), 0, 100).round(1),
            "avg_submission_time_days": np.clip(rng.normal(t, 1.2, n), 0.1, None).round(2),
            "quiz_score": np.clip(rng.normal(q, 10, n), 0, 100).round(1),
        }))
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "student_id", [f"S{i + 1:03d}" for i in range(len(df))])
    return df


@st.cache_data
def run_elbow(X_scaled: np.ndarray, max_k: int = 7):
    ks = list(range(1, max_k + 1))
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in ks]
    return ks, inertias


def label_clusters(df: pd.DataFrame, k: int):
    """Rank clusters by overall performance so labels never depend on arbitrary cluster ids."""
    means = df.groupby("cluster")[FEATURES].mean()
    z = (means - means.mean()) / means.std(ddof=0).replace(0, 1)
    score = z["completion_rate"] + z["quiz_score"] - z["avg_submission_time_days"]
    order = score.sort_values(ascending=False).index.tolist()  # best -> worst

    labels, actions = {}, {}
    for rank, cid in enumerate(order):
        key = "top" if rank == 0 else ("low" if rank == k - 1 else "mid")
        if k == 3:
            labels[cid] = LABELS_3[key]
        else:
            tag = {"top": "Top", "low": "Lowest", "mid": f"Tier {rank + 1}"}[key]
            labels[cid] = f"{tag} segment (rank {rank + 1} of {k})"
        actions[cid] = ACTIONS[key]
    return labels, actions


# ---------------- sidebar ----------------
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

# ---------------- model ----------------
X = df[FEATURES].values
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)
labels, actions = label_clusters(df, chosen_k)
df["segment_label"] = df["cluster"].map(labels)
dist = kmeans.transform(X_scaled)
df["dist_to_own_cluster"] = dist[np.arange(len(df)), df["cluster"]]
df["Cluster"] = "Cluster " + df["cluster"].astype(str)

# ---------------- header ----------------
st.title("🎓 Student Performance Segmentation")
st.write("K-Means clustering of students by completion rate, submission speed and quiz score.")

c1, c2, c3 = st.columns(3)
c1.metric("Students", len(df))
c2.metric("Clusters (k)", chosen_k)
c3.metric("Inertia", f"{kmeans.inertia_:.1f}")

tab_data, tab_elbow, tab_clusters, tab_segments, tab_border = st.tabs(
    ["📄 Data", "📉 Elbow", "🔵 Clusters", "🧭 Segments & Actions", "🔍 Borderline Students"]
)

with tab_data:
    st.subheader("Dataset")
    st.dataframe(df[["student_id"] + FEATURES], use_container_width=True)
    st.subheader("Summary statistics")
    st.dataframe(df[FEATURES].describe().round(2), use_container_width=True)
    with st.expander("Scaling example (first student)"):
        st.write("Before scaling:", dict(zip(FEATURES, X[0].round(2))))
        st.write("After scaling:", dict(zip(FEATURES, X_scaled[0].round(3))))

with tab_elbow:
    st.subheader("Elbow method")
    ks, inertias = run_elbow(X_scaled)
    st.line_chart(pd.DataFrame({"k": ks, "Inertia": inertias}).set_index("k"))
    st.caption(f"Look for the 'elbow' where adding clusters stops reducing inertia sharply. Chosen k = {chosen_k}.")

with tab_clusters:
    st.subheader("Cluster scatter plots")
    pairs = [
        ("completion_rate", "quiz_score"),
        ("avg_submission_time_days", "quiz_score"),
        ("completion_rate", "avg_submission_time_days"),
    ]
    cols = st.columns(3)
    for col, (x, y) in zip(cols, pairs):
        with col:
            st.markdown(f"**{NICE[x]} vs {NICE[y]}**")
            plot_df = df[["student_id", "Cluster", x, y]].rename(columns={x: NICE[x], y: NICE[y]})
            st.scatter_chart(plot_df, x=NICE[x], y=NICE[y], color="Cluster", size=60, height=380)

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

with tab_border:
    st.subheader("Students farthest from their cluster center")
    st.write("These may be borderline cases that don't fit their segment cleanly.")
    n = st.slider("How many to show", 3, min(20, len(df)), 6)
    st.dataframe(
        df.sort_values("dist_to_own_cluster", ascending=False)
          [["student_id"] + FEATURES + ["cluster", "dist_to_own_cluster"]].head(n),
        use_container_width=True,
    )
