
import streamlit as st
from datetime import datetime

from model import predict_review


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="centered"
)


# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stButton button {
    border-radius: 10px;
    height: 3rem;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.title("🔍 Fake Review Detector")

st.markdown(
    """
    ### AI-Powered Review Analysis

    Analyze product reviews using **Natural Language Processing**
    and a **Random Forest Classifier**.

    Enter the review details below to check whether it is genuine or fake.
    """
)

st.divider()


# ---------------- REVIEW DETAILS ----------------

st.subheader("📝 Review Information")

title = st.text_input(
    "Review Title",
    placeholder="Enter the review headline..."
)

review_text = st.text_area(
    "Review Content",
    placeholder="Paste the complete product review here...",
    height=180
)


# ---------------- PRODUCT DETAILS ----------------

st.subheader("📦 Product Information")

col1, col2 = st.columns(2)

with col1:

    asin = st.text_input(
        "Product ASIN",
        placeholder="Enter ASIN"
    )

    rating = st.slider(
        "Product Rating",
        min_value=1,
        max_value=5,
        value=5
    )

with col2:

    parent_asin = st.text_input(
        "Parent ASIN",
        placeholder="Enter Parent ASIN"
    )

    verified_purchase = st.selectbox(
        "Verified Purchase",
        options=["True", "False"]
    )


# ---------------- USER DETAILS ----------------

st.subheader("👤 Reviewer Information")

user_id = st.text_input(
    "Reviewer User ID",
    placeholder="Enter Reviewer ID"
)

helpful_vote = st.number_input(
    "Helpful Votes",
    min_value=0,
    value=0,
    step=1
)


# ---------------- TIMESTAMP ----------------

st.subheader("📅 Review Timestamp")

col3, col4 = st.columns(2)

with col3:

    review_date = st.date_input(
        "Review Date",
        value=datetime.now().date()
    )

with col4:

    review_time = st.time_input(
        "Review Time",
        value=datetime.now().time().replace(
            second=0,
            microsecond=0
        )
    )

timestamp = datetime.combine(
    review_date,
    review_time
)


st.divider()


# ---------------- PREDICTION ----------------

if st.button(
    "🔍 Analyze Review",
    use_container_width=True
):

    if not title.strip() or not review_text.strip():

        st.warning(
            "Please enter both the review title and review content."
        )

    elif not asin.strip() or not parent_asin.strip() or not user_id.strip():

        st.warning(
            "Please enter the Product ASIN, Parent ASIN and Reviewer ID."
        )

    else:

        try:

            with st.spinner("Analyzing review using Machine Learning..."):

                prediction, confidence, probabilities = predict_review(

                    title=title,

                    text=review_text,

                    asin=asin,

                    parent_asin=parent_asin,

                    user_id=user_id,

                    verified_purchase=verified_purchase,

                    rating=rating,

                    helpful_vote=helpful_vote,

                    timestamp=timestamp

                )


            # ---------------- DISPLAY RESULT ----------------

            st.divider()

            st.subheader("📊 Analysis Result")

            # CHANGE THIS MAPPING IF YOUR DATASET USES REVERSE LABELS
            LABEL_MAP = {
                "0": "Fake Review",
                "1": "Genuine Review"
            }

            predicted_label = str(prediction)

            result = LABEL_MAP.get(
                predicted_label,
                f"Class {predicted_label}"
            )

            if result == "Fake Review":

                st.error(
                    "⚠️ This review is predicted to be FAKE."
                )

            elif result == "Genuine Review":

                st.success(
                    "✅ This review is predicted to be GENUINE."
                )

            else:

                st.info(
                    f"Predicted category: {result}"
                )


            # Confidence

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(int(confidence), 0), 100)
            )


            # Class probabilities

            st.subheader("Class Probability")

            for label, probability in probabilities.items():

                class_name = LABEL_MAP.get(
                    label,
                    f"Class {label}"
                )

                st.write(
                    f"**{class_name}: {probability:.2f}%**"
                )

                st.progress(
                    min(max(int(probability), 0), 100)
                )


            st.caption(
                "Predictions are based on the trained Random Forest model. "
                "Confidence represents the model's predicted class probability, "
                "not a guarantee of correctness."
            )


        except FileNotFoundError as e:

            st.error(
                "Model artifacts are missing. "
                "Please train the model first using: python model.py"
            )

            st.code(str(e))


        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Fake Review Detector | NLP + Random Forest Classifier"
)
