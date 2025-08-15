import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from s3_utils import load_embeddings, load_df, load_model, load_patent_to_idx, load_faiss_index

st.set_page_config(page_title="Patent Explorer", layout="wide")

st.title("Patent Topic Explorer")
st.subheader("Filing dates: January 2016 - sample dataset")

# Use sample dataset, load all data into memory
# TODO: adjust outputs and code to load full dataset
df, embeddings_normalized, patent_to_idx, model = load_df(prefix=f"outputs_sample"), load_embeddings(prefix=f"outputs_sample"), load_patent_to_idx(prefix=f"outputs_sample"), load_model(prefix=f"outputs_sample")

patent_input = st.text_input("Enter UPSTO Patent Number (e.g. 9713127)", value="9713127")
st.write("Note: This is a sample dataset for patents filed in Janurary 2016")

if patent_input:
    if patent_input not in patent_to_idx:
        st.error("Patent number not found in system.")
    else:
        idx = patent_to_idx[patent_input]
        link = f"https://patents.google.com/patent/US{patent_input}"
        st.write(f"### [Patent US{patent_input} link]({link})")
        st.subheader(f"Patent title: {df.iloc[idx]['title']}")

        topic_id = df.iloc[idx]["topic"]
        if topic_id == -1:
            st.subheader("Topic: No dominant topic found.")
        else:
            topic_words = model.get_topic(topic_id)
            if topic_words:
                words = ", ".join(
                    [word for word, _ in sorted(topic_words, key=lambda x: x[1], reverse=True)[:5]]
                )
                st.subheader(f"Top words in topic: {words}")
            else:
                st.subheader("No topic words found.")

        # Similar patents
        # TODO: use FAISS index instead of cosine_similarity for full dataset
        query_emb = embeddings_normalized[idx].reshape(1, -1)
        similarities = cosine_similarity(query_emb, embeddings_normalized)[0]
        similarities[idx] = -1
        top_indices = similarities.argsort()[-5:][::-1]

        st.write("### Similar patents using cosine similarity:")
        for i in top_indices:
            pn_sim = df["patent_number"].iloc[i]
            sim_score = similarities[i]
            doc_title = df["title"].iloc[i]
            doc_link = f"https://patents.google.com/patent/US{pn_sim}"
            st.write(f"- **{pn_sim}** (score: {sim_score:.3f}): {doc_title} - [Link]({doc_link})")
