"""
Kickstarter Campaign Success Predictor — Final Streamlit App
============================================================
Run:
    streamlit run streamlit_app.py

This app is aligned with the final trained model:
- model.joblib
- meta.json

Pages:
1. Predictor
2. Data Explorer
3. Success Drivers
4. Model Overview
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Kickstarter Success Predictor",
    page_icon="🚀",
    layout="wide",
)

BASE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.joblib"))


@st.cache_data
def load_meta():
    with open(os.path.join(BASE, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_meta()

CATEGORIES = sorted(meta["categories"])
CATEGORY_STATS = meta["category_stats"]
MEDIAN_SUCCESS = meta["median_goal_success_usd"]
MEDIAN_FAIL = meta["median_goal_fail_usd"]
OVERALL_SUCCESS = meta["overall_success_rate"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_input(title, description, category, goal, duration):
    title = (title or "").strip()
    description = (description or "").strip()
    goal = float(goal)
    duration = float(duration)

    goal_per_day = goal / duration if duration > 0 else goal

    return pd.DataFrame([{
        "title": title,
        "blurb": description,
        "main_category": category,
        "log_goal_usd": float(np.log1p(goal)),
        "campaign_duration": duration,
        "title_len": len(title),
        "blurb_len": len(description),
        "title_word_count": len(title.split()),
        "blurb_word_count": len(description.split()),
        "has_blurb": int(len(description) > 0),
        "log_goal_per_day": float(np.log1p(max(goal_per_day, 0))),
    }])


def signal_label(prob):
    dist = abs(prob - 0.5)
    if dist >= 0.25:
        return "Strong signal"
    if dist >= 0.12:
        return "Moderate signal"
    return "Uncertain estimate"


def category_tier(rate):
    if rate >= 0.70:
        return "Strong"
    if rate >= 0.55:
        return "Average"
    return "Challenging"


def build_prediction_insights(category, goal, duration, description):
    msgs = []

    # Goal context
    if goal <= MEDIAN_SUCCESS:
        msgs.append(
            f"Your goal of **${goal:,.0f}** is at or below the median goal for historically successful campaigns "
            f"(**${MEDIAN_SUCCESS:,.0f}**). Lower goals tend to perform better."
        )
    elif goal <= MEDIAN_FAIL:
        msgs.append(
            f"Your goal of **${goal:,.0f}** sits between the median successful goal (**${MEDIAN_SUCCESS:,.0f}**) "
            f"and the median failed goal (**${MEDIAN_FAIL:,.0f}**). That is workable, but less forgiving."
        )
    else:
        msgs.append(
            f"Your goal of **${goal:,.0f}** is above the median goal for historically failed campaigns "
            f"(**${MEDIAN_FAIL:,.0f}**). Higher goals are one of the strongest negative signals."
        )

    # Category context
    rate = CATEGORY_STATS.get(category, {}).get("success_rate", OVERALL_SUCCESS)
    tier = category_tier(rate)
    msgs.append(
        f"**{category}** is a **{tier.lower()}** category in this dataset, with a historical success rate of "
        f"**{rate * 100:.1f}%**."
    )

    # Duration context
    if duration <= 20:
        msgs.append(
            "A shorter campaign can create urgency. In the historical data, shorter campaigns often perform reasonably well."
        )
    elif duration <= 35:
        msgs.append(
            "A duration around 30 days is common and generally well-supported by historical Kickstarter patterns."
        )
    else:
        msgs.append(
            "Longer campaigns can lose momentum. Durations above 35 days are usually a weaker signal."
        )

    # Blurb context
    if len(description.strip()) == 0:
        msgs.append(
            "You left the blurb empty. The model can still predict, but having a clear description usually helps the text signal."
        )
    elif len(description.split()) < 8:
        msgs.append(
            "Your blurb is very short. A little more detail may help the campaign communicate value more clearly."
        )
    else:
        msgs.append(
            "You provided enough description for the model to pick up useful text patterns from the blurb."
        )

    return msgs


def category_dataframe():
    df = (
        pd.DataFrame(CATEGORY_STATS)
        .T.reset_index()
        .rename(columns={"index": "Category", "success_rate": "Success Rate", "count": "Campaigns"})
        .sort_values("Success Rate", ascending=False)
    )
    df["Success Rate %"] = (df["Success Rate"] * 100).round(1)
    df["Tier"] = df["Success Rate"].apply(category_tier)
    return df


def goal_benchmark_df():
    return pd.DataFrame({
        "Group": ["Successful campaigns", "Failed campaigns"],
        "Median Goal (USD)": [MEDIAN_SUCCESS, MEDIAN_FAIL],
    })


def duration_reference_df():
    return pd.DataFrame({
        "Duration Bucket": ["1–20 days", "21–35 days", "36–60 days"],
        "Interpretation": [
            "Shorter, more urgency-driven campaigns",
            "Most standard / common range",
            "Longer campaigns that may lose momentum",
        ]
    })


def success_word_tables():
    success_words = pd.DataFrame({
        "Words Associated with Success": [
            "photobook", "documentary", "book", "children book",
            "illustrated", "comedy", "record", "tarot",
            "debut", "painting"
        ]
    })

    failure_words = pd.DataFrame({
        "Words Associated with Failure": [
            "want", "platform", "looking", "trying",
            "creating", "youtube", "website", "start",
            "business", "funding"
        ]
    })

    return success_words, failure_words


def what_if_goal_df(title, description, category, duration):
    goals = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000]
    rows = []
    for g in goals:
        input_df = build_input(title, description, category, g, duration)
        prob = float(model.predict_proba(input_df)[0][1])
        rows.append({
            "Goal (USD)": g,
            "Success Probability (%)": round(prob * 100, 1),
        })
    return pd.DataFrame(rows)


def what_if_duration_df(title, description, category, goal):
    durations = [7, 14, 21, 30, 45, 60]
    rows = []
    for d in durations:
        input_df = build_input(title, description, category, goal, d)
        prob = float(model.predict_proba(input_df)[0][1])
        rows.append({
            "Duration (days)": d,
            "Success Probability (%)": round(prob * 100, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Predictor", "Data Explorer", "Success Drivers", "Model Overview"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Final model: Logistic Regression with separate TF-IDF on title and blurb.")


# ---------------------------------------------------------------------
# Page: Predictor
# ---------------------------------------------------------------------

if page == "Predictor":
    st.title("🚀 Kickstarter Success Predictor")
    st.caption(
        "Estimate whether a Kickstarter campaign is likely to succeed based on historical patterns in finished campaigns."
    )
    st.divider()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Campaign Inputs")

        title = st.text_input(
            "Campaign Title *",
            placeholder="e.g. The Art of Urban Sketching"
        )

        description = st.text_area(
            "Short Description / Blurb",
            placeholder="A one- or two-sentence pitch for your campaign...",
            height=120
        )

        category = st.selectbox("Category *", CATEGORIES)

        goal = st.number_input(
            "Funding Goal (USD) *",
            min_value=1,
            max_value=10_000_000,
            value=5000,
            step=500,
            format="%d"
        )

        duration = st.slider(
            "Campaign Duration (days)",
            min_value=1,
            max_value=60,
            value=30
        )

        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    with right:
        st.subheader("Prediction")

        if predict_btn:
            if not title.strip():
                st.error("Please enter a campaign title.")
            else:
                with st.spinner("Running model..."):
                    input_df = build_input(title, description, category, goal, duration)
                    prob = float(model.predict_proba(input_df)[0][1])
                    label = int(prob >= 0.50)

                if label == 1:
                    st.success("### Likely to succeed")
                else:
                    st.error("### Unlikely to succeed")

                col1, col2 = st.columns(2)
                col1.metric("Success Probability", f"{prob * 100:.1f}%")
                col2.metric("Signal Strength", signal_label(prob))

                st.progress(prob, text=f"{prob * 100:.1f}% predicted probability of success")

                st.markdown("**What the data says about this setup:**")
                for msg in build_prediction_insights(category, goal, duration, description):
                    st.markdown(f"- {msg}")

                st.info(
                    "This prediction reflects historical Kickstarter patterns. It does not directly measure idea quality, audience size, marketing execution, or creator reputation."
                )
        else:
            st.info("Fill in the campaign details and click Predict.")

    st.divider()
    st.subheader("What-if analysis")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("**If you change the funding goal**")
        if title.strip():
            goal_df = what_if_goal_df(title, description, category, duration)
            st.line_chart(goal_df.set_index("Goal (USD)"))
            st.dataframe(goal_df, use_container_width=True)
        else:
            st.info("Enter a title above to activate the what-if charts.")

    with col_b:
        st.markdown("**If you change the campaign duration**")
        if title.strip():
            duration_df = what_if_duration_df(title, description, category, goal)
            st.line_chart(duration_df.set_index("Duration (days)"))
            st.dataframe(duration_df, use_container_width=True)
        else:
            st.info("Enter a title above to activate the what-if charts.")


# ---------------------------------------------------------------------
# Page: Data Explorer
# ---------------------------------------------------------------------

elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.caption("Quick EDA-style views built from the model metadata and project findings.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Success Rate", f"{OVERALL_SUCCESS * 100:.1f}%")
    c2.metric("Median Successful Goal", f"${MEDIAN_SUCCESS:,.0f}")
    c3.metric("Median Failed Goal", f"${MEDIAN_FAIL:,.0f}")

    st.divider()

    st.subheader("Category success rates")
    cat_df = category_dataframe()

    st.bar_chart(
        cat_df.sort_values("Success Rate %").set_index("Category")["Success Rate %"],
        height=400,
        use_container_width=True
    )

    st.dataframe(
        cat_df[["Category", "Success Rate %", "Campaigns", "Tier"]],
        use_container_width=True
    )

    st.divider()

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Goal benchmarks")
        goals_df = goal_benchmark_df()
        st.dataframe(goals_df, use_container_width=True)
        st.bar_chart(goals_df.set_index("Group")["Median Goal (USD)"], use_container_width=True)

    with right:
        st.subheader("Duration interpretation")
        st.dataframe(duration_reference_df(), use_container_width=True)
        st.markdown(
            """
            **Takeaway:** lower goals generally help, and very long campaigns tend to be weaker.
            This does not guarantee success, but these patterns show up clearly in the historical data.
            """
        )

    st.divider()

    st.subheader("What the dataset suggests")
    st.markdown(
        """
        - Successful campaigns tend to have **lower funding goals**.
        - Category matters: some categories historically fund at much higher rates than others.
        - Campaign length matters, but less than funding goal.
        - Text adds useful signal, especially around certain recurring words and themes.
        """
    )


# ---------------------------------------------------------------------
# Page: Success Drivers
# ---------------------------------------------------------------------

elif page == "Success Drivers":
    st.title("🧠 What Seems to Drive Success?")
    st.caption("A simple interpretation page based on the final trained model and its learned coefficients.")
    st.divider()

    st.subheader("Main patterns")
    st.markdown(
        """
        The strongest overall patterns in this model are:

        1. **Goal size matters a lot.** Lower goals are much more likely to succeed.
        2. **Category matters.** Some creative categories perform much better historically than others.
        3. **Text matters, but less than goal.** Certain words and themes appear more often in successful campaigns.
        4. **Duration matters somewhat.** Extremely long campaigns are usually weaker.
        """
    )

    st.divider()

    success_words, failure_words = success_word_tables()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Words that leaned positive")
        st.dataframe(success_words, use_container_width=True)

    with right:
        st.subheader("Words that leaned negative")
        st.dataframe(failure_words, use_container_width=True)

    st.divider()

    st.subheader("Interpreting those words carefully")
    st.info(
        "These words are correlations from the training data, not universal truths. "
        "A word does not cause success by itself. It just reflects patterns the model learned from many campaigns."
    )

    st.subheader("Plain-English interpretation")
    st.markdown(
        """
        Campaigns that look more like **books, documentary projects, illustrated concepts, or clearly packaged creative products**
        tend to receive more positive model signals.

        Campaigns that look more like **vague startups, platforms, websites, or loosely described business ideas**
        tend to receive more negative model signals.

        That does not mean those projects cannot win. It just means their historical pattern in this dataset was weaker.
        """
    )

    st.divider()

    st.subheader("Category leaderboard")
    cat_df = category_dataframe()
    top5 = cat_df.head(5)[["Category", "Success Rate %", "Campaigns"]]
    bottom5 = cat_df.tail(5)[["Category", "Success Rate %", "Campaigns"]]

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Highest-performing categories**")
        st.dataframe(top5, use_container_width=True)

    with col2:
        st.markdown("**Lowest-performing categories**")
        st.dataframe(bottom5, use_container_width=True)


# ---------------------------------------------------------------------
# Page: Model Overview
# ---------------------------------------------------------------------

elif page == "Model Overview":
    st.title("⚙️ Model Overview")
    st.caption("Technical summary of the final model used in the app.")
    st.divider()

    st.subheader("Final model")
    st.markdown(
        """
        **Algorithm:** Logistic Regression

        **Training objective:** Predict whether a finished Kickstarter campaign is labeled as successful.

        **Why this model?**
        - It works well with sparse TF-IDF text features.
        - It is interpretable.
        - It performed much better than a naive majority-class baseline.
        - It is more appropriate here than a random forest on large sparse text vectors.
        """
    )

    st.divider()

    st.subheader("Features used")
    feature_df = pd.DataFrame({
        "Feature": [
            "Title text",
            "Blurb text",
            "Main category",
            "Log goal (USD)",
            "Campaign duration",
            "Title length",
            "Blurb length",
            "Title word count",
            "Blurb word count",
            "Has blurb",
            "Log goal per day",
        ],
        "How it is used": [
            "TF-IDF, unigrams + bigrams",
            "TF-IDF, unigrams + bigrams",
            "One-hot encoded",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Numeric",
            "Binary numeric",
            "Numeric",
        ]
    })
    st.dataframe(feature_df, use_container_width=True)

    st.divider()

    st.subheader("Selected model settings")
    st.markdown(
        """
        - **Title TF-IDF max features:** 2500  
        - **Blurb TF-IDF max features:** 6000  
        - **min_df:** 5 for both title and blurb  
        - **Penalty:** L2  
        - **C:** 2.0  
        - **Threshold used in final app:** 0.50
        """
    )

    st.divider()

    st.subheader("Performance")
    perf_df = pd.DataFrame({
        "Model": [
            "Dummy baseline",
            "Logistic Regression (default threshold, final choice)",
            "Logistic Regression (tuned threshold, not selected)",
        ],
        "Accuracy": [0.6299, 0.7370, 0.7435],
        "Macro F1": [0.3864, 0.7258, 0.6898],
    })
    st.dataframe(perf_df, use_container_width=True)

    st.markdown(
        """
        **Final choice:** the default 0.50 threshold version.

        Even though the tuned threshold slightly increased accuracy, it reduced macro F1 and became much worse at identifying failed campaigns.
        """
    )

    st.divider()

    st.subheader("Limitations")
    st.markdown(
        """
        - The model does **not** know the creator's reputation, audience size, or marketing plan.
        - It does **not** evaluate whether an idea is morally good or socially valuable.
        - It only learns from historical Kickstarter patterns in the available dataset.
        - It should be treated as a decision-support tool, not a guarantee.
        """
    )