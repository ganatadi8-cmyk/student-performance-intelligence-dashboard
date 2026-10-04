import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance Intelligence Dashboard",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def build_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(45, 100, n).round(1)
    study_hours = rng.uniform(0.5, 7.0, n).round(1)
    previous_score = rng.uniform(35, 95, n).round(1)
    sleep_hours = rng.uniform(4.5, 9.0, n).round(1)
    assignments = rng.integers(2, 11, n)

    noise = rng.normal(0, 5.5, n)
    final_score = (
        0.28 * attendance
        + 4.0 * study_hours
        + 0.40 * previous_score
        + 0.8 * sleep_hours
        + 1.2 * assignments
        - 25
        + noise
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    return pd.DataFrame({
        "attendance_percent": attendance,
        "study_hours_per_day": study_hours,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "assignments_completed": assignments,
        "final_score": final_score
    })

@st.cache_resource
def train_model(data):
    features = [
        "attendance_percent",
        "study_hours_per_day",
        "previous_score",
        "sleep_hours",
        "assignments_completed"
    ]
    X = data[features]
    y = data["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }
    return model, features, metrics

def risk_label(score):
    if score >= 75:
        return "Low Risk", "Strong performance range"
    if score >= 50:
        return "Moderate Risk", "Needs consistent improvement"
    return "High Risk", "Needs immediate academic support"

def improvement_tips(attendance, study_hours, previous_score, sleep_hours, assignments):
    tips = []
    if attendance < 75:
        tips.append("Improve attendance toward at least 75–85%.")
    if study_hours < 2.5:
        tips.append("Build a consistent 2.5–3 hour focused study routine.")
    if previous_score < 60:
        tips.append("Revise weak subjects using mistakes from previous tests.")
    if sleep_hours < 6:
        tips.append("Increase sleep to support focus and memory.")
    if assignments < 8:
        tips.append("Complete more assignments on time for extra practice.")
    if not tips:
        tips.append("Current habits look balanced. Maintain consistency.")
    return tips

data = build_dataset()
model, features, metrics = train_model(data)

st.title("🎓 Student Performance Intelligence Dashboard")
st.write(
    "Predict academic performance, estimate risk, understand important factors, "
    "and test improvement scenarios."
)

m1, m2, m3 = st.columns(3)
m1.metric("Model R²", f"{metrics['r2']:.3f}")
m2.metric("Model MAE", f"{metrics['mae']:.2f} marks")
m3.metric("Training Samples", len(data))

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Student Inputs")
    attendance = st.slider("Attendance (%)", 0, 100, 78)
    study_hours = st.slider("Study hours/day", 0.0, 10.0, 3.0, 0.5)
    previous_score = st.slider("Previous score", 0, 100, 68)
    sleep_hours = st.slider("Sleep hours/day", 3.0, 10.0, 7.0, 0.5)
    assignments = st.slider("Assignments completed (out of 10)", 0, 10, 8)
    analyze = st.button("Analyze Performance", type="primary", use_container_width=True)

with right:
    st.subheader("Factor Importance")
    importance = pd.DataFrame({
        "Factor": ["Attendance", "Study Hours", "Previous Score", "Sleep Hours", "Assignments"],
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.bar_chart(importance.set_index("Factor"))

if analyze:
    row = pd.DataFrame([{
        "attendance_percent": attendance,
        "study_hours_per_day": study_hours,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "assignments_completed": assignments
    }])

    prediction = float(model.predict(row[features])[0])
    prediction = max(0, min(100, prediction))
    risk, message = risk_label(prediction)

    st.divider()
    st.subheader("Performance Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Final Score", f"{prediction:.1f}/100")
    c2.metric("Risk Level", risk)
    c3.metric("Status", message)

    st.subheader("Personalized Suggestions")
    for item in improvement_tips(attendance, study_hours, previous_score, sleep_hours, assignments):
        st.write(f"• {item}")

    st.subheader("What-If Improvement Simulator")
    w1, w2, w3 = st.columns(3)
    new_attendance = w1.slider("Target attendance (%)", attendance, 100, min(100, attendance + 10))
    new_study = w2.slider("Target study hours/day", study_hours, 10.0, min(10.0, study_hours + 1.5), 0.5)
    new_assignments = w3.slider("Target assignments", assignments, 10, min(10, assignments + 1))

    scenario = pd.DataFrame([{
        "attendance_percent": new_attendance,
        "study_hours_per_day": new_study,
        "previous_score": previous_score,
        "sleep_hours": sleep_hours,
        "assignments_completed": new_assignments
    }])

    future_score = float(model.predict(scenario[features])[0])
    future_score = max(0, min(100, future_score))
    improvement = future_score - prediction

    s1, s2 = st.columns(2)
    s1.metric("What-If Predicted Score", f"{future_score:.1f}/100")
    s2.metric("Potential Improvement", f"{improvement:+.1f} marks")

st.divider()
st.caption(
    "Portfolio/demo project using synthetic data. Do not use for real academic decisions."
)
