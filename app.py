import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="BINUS Traffic Predictor",
    page_icon="🚦",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(90deg,#003366,#0055AA);
    color: white;
    margin-bottom: 20px;
}

.metric-card {
    background: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align:center;
}

.result-card {
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.footer {
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================

try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ======================
# HEADER
# ======================

st.markdown("""
<div class="hero">
<h1>🚦 FlowSense - Traffic Intelligence Dashboard</h1>
<h4>Machine Learning-Based Congestion Prediction for BINUS Alam Sutera</h4>

Predict traffic conditions using Random Forest Classification.

Features:
• Weather Analysis
• Time-Based Prediction
• Congestion Classification
• Traffic Intelligence Dashboard
</div>
""", unsafe_allow_html=True)

st.divider()

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Classes",
        "3"
    )

with col3:
    st.metric(
        "Features",
        "4"
    )

# ======================
# SIDEBAR
# ======================

st.sidebar.header("Prediction Inputs")

month = st.sidebar.selectbox(
    "Month",
    [4, 5]
)

day_name = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

time = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=15
)

weather_name = st.sidebar.selectbox(
    "Weather Condition",
    [
        "Clear",
        "Partially Cloudy",
        "Rain"
    ]
)

weather_map = {
    "Clear": 0,
    "Partially Cloudy": 1,
    "Rain": 2
}

# ======================
# PREDICTION
# ======================

if st.sidebar.button("Predict Traffic"):

    input_data = pd.DataFrame({
        "Month": [month],
        "DayOfWeek": [day_map[day_name]],
        "Time": [time],
        "Weather": [weather_map[weather_name]]
    })

    try:
        prediction = model.predict(input_data)[0]

        labels = {
            0: "Stable",
            1: "Unstable",
            2: "Breakdown"
        }

        result = labels.get(prediction, str(prediction))

        col1, col2 = st.columns([1, 1])

        with col1:

            st.subheader("Traffic Status")

            if result == "Stable":
                st.markdown("""
    <div class="result-card"
    style="background:#14532D;color:white;">
    🟢 STABLE
    <br>
    Road operating normally
    </div>
    """, unsafe_allow_html=True)

            elif result == "Unstable":
                st.markdown("""
    <div class="result-card"
    style="background:#854D0E;color:white;">
    🟡 UNSTABLE
    <br>
    Road approaching capacity
    </div>
    """, unsafe_allow_html=True)

            else:
                st.markdown("""
    <div class="result-card"
    style="background:#7F1D1D;color:white;">
    🔴 BREAKDOWN
    <br>
    Severe congestion detected
    </div>
    """, unsafe_allow_html=True)

        with col2:

            try:
                probs = model.predict_proba(input_data)[0]

                confidence = max(probs) * 100

                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence:.2f}%"
                )

                prob_df = pd.DataFrame({
                    "Condition": [
                        "Stable",
                        "Unstable",
                        "Breakdown"
                    ],
                    "Probability": probs
                })

                fig = px.pie(
    prob_df,
    names="Condition",
    values="Probability",
    hole=0.6,
    title="Prediction Probability"
)
                fig.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:
                st.warning(
                    f"Probability chart unavailable: {e}"
                )

        st.divider()

        st.subheader("Input Summary")

        summary_df = pd.DataFrame({
            "Feature": [
                "Month",
                "Day",
                "Hour",
                "Weather"
            ],
            "Value": [
                month,
                day_name,
                time,
                weather_name
            ]
        })

        st.dataframe(
            summary_df,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
    
        st.subheader("📊 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": [
                "Month",
                "DayOfWeek",
                "Time",
                "Weather"
            ],
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=True
        )

        importance_chart = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(
            importance_chart,
            use_container_width=True
        )

        st.subheader("📌 Traffic Insights")

        if result == "Stable":

            st.info("""
            • Traffic flow is smooth

            • Road operates below capacity

            • No significant delay expected
            """)

        elif result == "Unstable":

            st.warning("""
            • Road is nearing capacity

            • Moderate congestion expected

            • Consider leaving earlier
            """)

        else:

            st.error("""
            • Severe congestion detected

            • Demand exceeds road capacity

            • Alternative routes recommended
            """)

# ======================
# FOOTER
# ======================

st.divider()

st.markdown("""
<div class="footer">
FlowSense - Traffic Intelligence Dashboard<br>
Machine Learning Project - BINUS University
</div>
""", unsafe_allow_html=True)