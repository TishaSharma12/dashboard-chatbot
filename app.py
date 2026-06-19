import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ✅ PAGE CONFIG
st.set_page_config(page_title="Smart Dashboard Assistant", layout="centered")

st.title("🤖 Smart Dashboard Assistant")

# =========================
# ✅ NORMALIZATION
# =========================
def normalize_text(text):
    text = text.lower()
    text = text.replace("dotn", ".").replace("dot", ".")
    text = text.replace("m&r", "mnr")
    text = text.replace("c&s", "cns")
    text = text.replace("e&i", "eni")
    text = re.sub(r"[^a-z0-9.]", "", text)
    return text

# =========================
# ✅ LOAD DATA
# =========================
df = pd.read_csv("dashboards.csv")
df.fillna("", inplace=True)

# ✅ CLEANED FIELDS
df["LOB_clean"] = df["LOB"].apply(lambda x: normalize_text(str(x)))
df["search_text"] = df["Dashboard Name"] + " " + df["LOB"]

# ✅ VECTOR SEARCH
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["search_text"])

# =========================
# ✅ SIDEBAR FILTER
# =========================
selected_lob = st.sidebar.selectbox(
    "🔎 Select LOB",
    ["All"] + sorted(df["LOB"].unique())
)

selected_lob_clean = normalize_text(selected_lob)

# =========================
# ✅ UI CARD
# =========================
def build_card(row):
    return f"""
<div style="
background:#ffffff;
padding:14px;
border-radius:12px;
margin-bottom:10px;
box-shadow:0 2px 8px rgba(0,0,0,0.08);
border-left:5px solid #4F8BF9;
">
<b>📊 {row['Dashboard Name']}</b><br>
<small style="color:#6b7280;">LOB: {row['LOB']}</small><br><br>

<a href="{row['Link']}" target="_blank"
style="text-decoration:none;background:#4F8BF9;color:white;padding:6px 10px;border-radius:6px;">
🔗 Open Dashboard
</a>

</div>
"""

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
# ✅ FILTER ONLY VIEW
# =========================
if selected_lob != "All" and not user_input:
    filtered_df = df[df["LOB_clean"] == selected_lob_clean]

    st.subheader(f"📊 Dashboards for {selected_lob}")

    for _, row in filtered_df.iterrows():
        st.markdown(build_card(row), unsafe_allow_html=True)

# =========================
# ✅ KEYWORD EXTRACTION
# =========================
def extract_keywords(text):
    STOP_WORDS = {"dashboard", "dashboards", "report", "reports", "link"}
    words = re.findall(r"[a-z0-9&.]+", text.lower())
    return [normalize_text(w) for w in words if w not in STOP_WORDS]

# =========================
# ✅ CHATBOT LOGIC
# =========================
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    user_lower = user_input.lower()
    user_clean = normalize_text(user_input)

    # ✅ HI PROMPT
    if any(word in user_lower for word in ["hi", "hello", "hey"]):
        bot_response = """
👋 Hello!

I can help you find dashboards quickly.

✅ Try:
• "MNR dashboard"
• "CNS dashboards"
• "optum.com"

Or use the filter on the left 👉
"""

    else:

        # =========================
        # ✅ 1. LOB DETECTION (TOP PRIORITY ✅)
        # =========================
        detected_lob = None

        for lob in df["LOB"].unique():
            if normalize_text(lob) in user_clean:
                detected_lob = lob
                break

        if detected_lob:
            matches = df[df["LOB_clean"] == normalize_text(detected_lob)]

            bot_response = f"📊 **All dashboards for {detected_lob}:**<br><br>"

            for _, row in matches.iterrows():
                bot_response += build_card(row)

        else:
            # =========================
            # ✅ 2. KEYWORD MATCH (B2B / NAME)
            # =========================
            keywords = extract_keywords(user_input)

            if keywords:
                # STRICT MATCH
                matches = df[df["search_text"].apply(
                    lambda x: all(kw in normalize_text(x) for kw in keywords)
                )]

                # FALLBACK
                if matches.empty:
                    matches = df[df["search_text"].apply(
                        lambda x: any(kw in normalize_text(x) for kw in keywords)
                    )]

                if not matches.empty:
                    bot_response = "🎯 **Matching dashboards:**<br><br>"
                    for _, row in matches.iterrows():
                        bot_response += build_card(row)
                else:
                    bot_response = "❌ No matching dashboard found."

            else:
                # =========================
                # ✅ 3. SEMANTIC SEARCH
                # =========================
                filtered_df = df.copy()

                if selected_lob != "All":
                    filtered_df = filtered_df[
                        filtered_df["LOB_clean"] == selected_lob_clean
                    ]

                user_vec = vectorizer.transform([user_input])
                similarity = cosine_similarity(user_vec, X).flatten()

                filtered_df["score"] = similarity[filtered_df.index]

                results = filtered_df.sort_values(
                    by="score", ascending=False
                ).head(3)

                bot_response = "✅ **Top matches:**<br><br>"

                for _, row in results.iterrows():
                    bot_response += build_card(row)

    st.session_state.messages.append(
        {"role": "assistant", "content": bot_response}
    )

    with st.chat_message("assistant"):
        st.markdown(bot_response, unsafe_allow_html=True)
