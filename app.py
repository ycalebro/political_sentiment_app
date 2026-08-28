# app.py
import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Political Sentiment Tracker",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Political Sentiment Tracker")
st.markdown(
    "Comparing cross-subreddit sentiment towards key political entities on"
    " Reddit."
)

data_path = "data/sentiment_results.csv"

# Verify CSV exists
if not os.path.exists(data_path):
    st.error(
        f"❌ Data file missing at `{data_path}`! Run `analyzer.py` first to"
        " generate data."
    )
else:
    sentiment_df = pd.read_csv(data_path)

    if sentiment_df.empty:
        st.warning("The dataset is empty. Run `analyzer.py` again.")
    else:
        # Sidebar Filter
        entities = sentiment_df["entity"].unique()
        selected_entity = st.sidebar.selectbox("Select Target Entity", entities)

        filtered_df = sentiment_df[sentiment_df["entity"] == selected_entity]

        # Metric Displays
        col1, col2, col3 = st.columns(3)

        with col1:
            avg_score = filtered_df["compound_score"].mean()
            st.metric(
                label=f"Average Sentiment ({selected_entity})",
                value=f"{avg_score:+.2f}",
                delta="Positive" if avg_score > 0 else "Negative",
            )

        with col2:
            st.metric(
                label="Total Mentions Analyzed", value=len(filtered_df)
            )

        with col3:
            top_sub = (
                filtered_df["subreddit"].value_counts().idxmax()
                if not filtered_df.empty
                else "N/A"
            )
            st.metric(label="Most Active Subreddit", value=f"r/{top_sub}")

        st.divider()

        # Row 1: Distribution Histogram & Mean Bar Chart
        col_left, col_right = st.columns(2)

        with col_left:
            fig_hist = px.histogram(
                filtered_df,
                x="compound_score",
                color="subreddit",
                barmode="overlay",
                title=f"Sentiment Distribution for {selected_entity}",
                labels={
                    "compound_score": (
                        "Sentiment Score (-1.0 Negative to +1.0 Positive)"
                    )
                },
                opacity=0.7,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            avg_by_sub = (
                filtered_df.groupby("subreddit")["compound_score"]
                .mean()
                .reset_index()
            )
            fig_bar = px.bar(
                avg_by_sub,
                x="subreddit",
                y="compound_score",
                color="subreddit",
                title="Average Sentiment by Subreddit",
                labels={"compound_score": "Mean Sentiment Score"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Row 2: Box Plot for Variance Analysis
        st.subheader("📊 Sentiment Variance & Volatility Analysis")
        st.markdown(
            "Box plots highlight median sentiment, overall variance (box width),"
            " and outlier posts."
        )

        fig_box = px.box(
            filtered_df,
            x="subreddit",
            y="compound_score",
            color="subreddit",
            points="all",  # Overlays individual data points on top of the box plot
            hover_data=["title"],  # Displays the post title when hovering over points
            title=f"Sentiment Variance for '{selected_entity}' across Subreddits",
            labels={
                "subreddit": "Subreddit Community",
                "compound_score": "Compound Sentiment Score",
            },
        )
        # Add baseline indicator at neutral 0.0
        fig_box.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            annotation_text="Neutral Line (0.0)",
        )

        st.plotly_chart(fig_box, use_container_width=True)

        # Raw Data Table
        st.subheader("Sample Posts Analyzed")
        st.dataframe(
            filtered_df[
                ["created_at", "subreddit", "title", "compound_score"]
            ].head(10),
            use_container_width=True,
        )