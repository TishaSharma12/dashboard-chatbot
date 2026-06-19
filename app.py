import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import re

st.set_page_config(page_title="Smart Dashboard Assistant", layout="centered")

st.title("🤖 Smart Dashboard Assistant")

# =========================
# ✅ NORMALIZATION
# =========================

def normalize_text(text):
    text = text.lower()
    text = text.replace("dotn", ".").replace("dot", ".")
    text = text.replace("m&r", "mnr").replace("c&s", "cns").replace("e&i", "eni")
    text = re.sub(r"[^a-z0-9.]", "", text)
    return text

# =========================
# ✅ LOAD DATA
# =========================

df = pd.read_csv("dashboards.csv")
df.fillna("", inplace=True)

df["LOB_clean"] = df["LOB"].apply(normalize_text)

df["combined"] = df["Dashboard Name"] + " " + df["LOB"] + " " + df["LOB_clean"]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"])

LOB_LIST = df["LOB_clean"].unique().tolist()

# =========================
# ✅ SIDEBAR FILTER
# =========================

selected_lob = st.sidebar.selectbox(
    "Select LOB",
    ["All"] + sorted(df["LOB"].unique())
)

selected_lob_clean = normalize_text(selected_lob)

# =========================
# ✅ CHAT MEMORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask like: MNR executive dashboard")

# =========================
# ✅ FILTER ONLY VIEW (FIXED)
# =========================

if selected_lob != "All" and not user_input:
    filtered_df = df[df["LOB_clean"] == selected_lob_clean]

    st.subheader(f"📊 Dashboards for {selected_lob}")

    for _, row in filtered_df.iterrows():
        st.markdown(f"{row['Dashboard Name']} → {row['Link']}")

# =========================
# ✅ CHATBOT LOGIC
# =========================

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    user_lower = user_input.lower()
    user_clean = normalize_text(user_input)

    # ✅ Greeting
    if any(word in user_lower for word in ["hi", "hello", "hey"]):
        bot_response = """
👋 **Hello!**

Try:
• CNS dashboards  
• MNR executive dashboards  
"""

    else:

        # ✅ Detect LOB from chat
        detected_lob = None
        for lob in LOB_LIST:
            if lob in user_clean:
                detected_lob = lob
                break

        # ✅ ✅ FINAL FILTER LOGIC (FIXED CORE)

        # Start with ALL data
        filtered_df = df.copy()

        # Apply sidebar filter FIRST
        if selected_lob != "All":
            filtered_df = filtered_df[filtered_df["LOB_clean"] == selected_lob_clean]

        # Apply chat LOB filter ALSO (intersection)
        if detected_lob:
            filtered_df = filtered_df[filtered_df["LOB_clean"] == detected_lob]

        # ✅ PURE LOB CASE
        if detected_lob and len(user_clean) <= len(detected_lob) + 2:

            bot_response = f"📊 **All {detected_lob.upper()} Dashboards**<br><br>"

            for _, row in filtered_df.iterrows():
                bot_response += f"{row['Dashboard Name']} → {row['Link']}<br>"

        else:
            # ✅ SEARCH
            user_vec = vectorizer.transform([user_input])
            similarity = cosine_similarity(user_vec, X).flatten()

            filtered_df = filtered_df.copy()
            filtered_df["similarity"] = similarity[filtered_df.index]

            fuzzy_scores = filtered_df["combined"].apply(
                lambda x: fuzz.token_set_ratio(user_input, x) / 100
            )

            filtered_df["score"] = filtered_df["similarity"] * 0.6 + fuzzy_scores * 0.4

            results = filtered_df.sort_values("score", ascending=False).head(3)

            bot_response = "✅ Results:<br><br>"

            for _, row in results.iterrows():
                bot_response += f"{row['Dashboard Name']} → {row['Link']}<br>"

    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response, unsafe_allow_html=True)
