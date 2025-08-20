import streamlit as st
import plotly.express as px
from s3_utils import load_topics_by_year_df, load_topics_count_df

st.set_page_config(page_title="Patent Topics", layout="wide")
st.title("Patent Topics Dashboard")
st.subheader("Filing years - 2013-2017, patent number exists")
st.subheader("[Click here to visit the GitHub repo](https://github.com/atalyaalon/patents-topic-modeling)")

st.header("Selected trending topics Found in exploration")

def modify_fig_layout(fig) -> None:
    fig.update_layout(
        legend=dict(
            font=dict(size=20),
            title=dict(font=dict(size=24))
        ),
        xaxis=dict(
            title=dict(font=dict(size=18)),
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title=dict(font=dict(size=18)),
            tickfont=dict(size=18)
        )
    )

    fig.update_traces(
        textfont_size=20
    )

df_by_year = load_topics_by_year_df()
df_topics = load_topics_count_df()

# First Selected Topic
increasing_topic_ids = [25]
df_by_year_increasing = df_by_year.loc[df_by_year.topic_id.isin(increasing_topic_ids)]
curr_topic_name = df_by_year_increasing['topic_words'].iloc[0]
st.subheader(f"Selected topic: {curr_topic_name} - showing an increase of 549% between 2013 and 2016")
year_order = sorted(df_by_year_increasing['year'].unique())

fig = px.line(
    df_by_year_increasing,
    x='year',
    y='count',
    color='topic_words',
    markers=True,
    category_orders={"year": year_order}
)
fig.update_layout(
    xaxis=dict(
        title="Year",
        dtick=1
    ),
    yaxis_title="Number of Patents"
)
modify_fig_layout(fig)
st.plotly_chart(fig, use_container_width=True)

# Additional Selected Topics
st.subheader(f"Additional selected topics - showing an increase between 2013 and 2016")

increasing_topic_ids = [252, 101, 124, 187]
df_by_year_increasing = df_by_year.loc[df_by_year.topic_id.isin(increasing_topic_ids)]
year_order = sorted(df_by_year_increasing['year'].unique())

fig = px.line(
    df_by_year_increasing,
    x='year',
    y='count',
    color='topic_words',
    markers=True,
    category_orders={"year": year_order}
)
fig.update_layout(
    xaxis=dict(
        title="Year",
        dtick=1
    ),
    yaxis_title="Number of Patents"
)
modify_fig_layout(fig)
st.plotly_chart(fig, use_container_width=True)

st.markdown("Patents in the UAV/drones, Augmented Reality, 3D printing, eye tracking, and IoT experienced significant growth between 2013 and 2016.\n\nIn 2017, filings declined compared to 2016, which mirrors the overall drop in total patents filed that year. This trend warrants further investigation.\n")


st.header("High-level statistics")
# Top 10 topics
st.subheader("Top 10 Topics")

df_topics_count = df_topics.copy()
df_topics_count = df_topics_count.loc[df_topics_count.topic_id != -1]
df_topics_count = df_topics_count.nlargest(10, 'count')

fig = px.bar(
    df_topics_count,
    x="topic_words",
    y="count",
    color="topic_words",
    text="count",
)
fig.update_layout(
    showlegend=False,
    xaxis_title="Topic",
    yaxis_title="Number of Patents"
)
modify_fig_layout(fig)
st.plotly_chart(fig, use_container_width=True)

# Patents by year
st.subheader("Total Patents by Filing Year")
df_count_per_year = df_by_year.groupby("year", as_index=False)["count"].sum()

fig = px.bar(
    df_count_per_year,
    x="year",
    y="count",
    labels={"count": "Number of Patents", "year": "Year"}
)
modify_fig_layout(fig)
st.plotly_chart(fig, use_container_width=True)

# Pie chart topic status
df_topics["topic_status"] = df_topics["topic_id"].apply(lambda x: "No Topic" if x == -1 else "Topic Exists")
df_status = df_topics.groupby("topic_status", as_index=False)["count"].sum()
st.subheader("Patents by Topic Status")
fig = px.pie(
    df_status,
    names="topic_status",
    values="count",
    color="topic_status"
)
fig.update_layout(
    legend=dict(
        x=0,
        y=1,
        xanchor='left',
        yanchor='top',
        orientation='v',
        font=dict(size=14)
    )
)
modify_fig_layout(fig)
st.plotly_chart(fig, use_container_width=True)

st.markdown("Note: This dashboard only presents patents that received a patent number")