import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import re

# ✅ Page config
st.set_page_config(page_title="Smart Dashboard Assistant", layout="centered")

# ✅ UI styling
st.markdown("""
<style>
body { background-color: #f5f7fb; }
[data-testid="stChatMessage"] { border-radius: 12px; padding: 10px; }
section[data-testid="stSidebar"] { background-color: #f9fafb; }
textarea { border-radius: 10px !important; }
h1 { color: #111827; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Smart Dashboard Assistant")

# =========================
# ✅ IMPROVED NORMALIZATION
# =========================

def normalize_text(text):
    text = text.lower()

    # ✅ Handle dot cases
    text = text.replace("dot", ".")
    text = text.replace("dotn", ".")  # handles optumdotncom

    # ✅ Convert LOB patterns BEFORE removing symbols
    
    # M&R → MNR
    text = text.replace("m&r", "mnr")
    text = text.replace("m & r", "mnr")

    # C&S → CNS
    text = text.replace("c&s", "cns")
    text = text.replace("c & s", "cns")

    # E&I → ENI
    text = text.replace("e&i", "eni")
    text = text.replace("e & i", "eni")

    # ✅ Now remove special characters
    text = re.sub(r"[^a-z0-9.]", "", text)

    return text

# =========================
# ✅ LOAD DATA
# =========================

df = pd.read_csv("dashboards.csv")
df.fillna("", inplace=True)

df["LOB_clean"] = df["LOB"].apply(normalize_text)

df["combined"] = (
    df["Dashboard Name"] + " " + df["LOB"] + " " + df["LOB_clean"] + " dashboard report"
)

# ✅ Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"])

# ✅ LOB list
LOB_LIST = df["LOB_clean"].unique().tolist()

# =========================
# ✅ SIDEBAR
# =========================

st.sidebar.header("🔎 Filter Dashboards")
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

user_input = st.chat_input("Ask for dashboards or use filters...")

# =========================
# ✅ FILTER ONLY VIEW
# =========================

if selected_lob != "All" and not user_input:
    filtered_df = df[df["LOB_clean"] == selected_lob_clean]

    st.subheader(f"📊 Dashboards for {selected_lob}")

    for _, row in filtered_df.iterrows():
        st.markdown(f"""
<div style="background:#fff;padding:12px;border-radius:12px;margin-bottom:10px;
box-shadow:0 2px 8px rgba(0,0,0,0.08); border-left:5px solid #4F8BF9;">
<b>📊 {row['Dashboard Name']}</b><br>
<small>LOB: {row['LOB']}</small><br>
{row['Link']}
</div>
""", unsafe_allow_html=True)

# =========================
# ✅ CHATBOT
# =========================

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    user_lower = user_input.lower()
    user_clean = normalize_text(user_input)

    # ✅ GREETING
    if any(word in user_lower for word in ["hi", "hello", "hey"]):
        bot_response = """👋 Hello!

Try:
• CNS
• C&S
• optumdotcom
"""

    # ✅ HELP
    elif "help" in user_lower:
        bot_response = """
📖 Works with variations:

• C&S = CNS  
• M&R = MNR  
• E&I = ENI  
• optumdotcom = optum.com  
"""

    # ✅ PURE LOB MATCH
    elif user_clean in LOB_LIST:

        filtered_df = df[df["LOB_clean"] == user_clean]

        bot_response = f"📊 **All dashboards for {user_clean.upper()}**<br><br>"

        for _, row in filtered_df.iterrows():
            bot_response += f"""
<div style="background:#fff;padding:12px;border-radius:12px;margin-bottom:10px;
box-shadow:0 2px 8px rgba(0,0,0,0.08); border-left:5px solid #4F8BF9;">
<b>📊 {row['Dashboard Name']}</b><br>
<small>LOB: {row['LOB']}</small><br>
{row['Link']}
</div>
"""

    # ✅ SMART SEARCH
    else:

        if selected_lob != "All":
            filtered_df = df[df["LOB_clean"] == selected_lob_clean].copy()
        else:
            filtered_df = df.copy()

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X).flatten()

        filtered_df["similarity"] = similarity[filtered_df.index]

        fuzzy_scores = filtered_df["combined"].apply(
            lambda x: fuzz.token_set_ratio(user_input, x) / 100
        )

        filtered_df["score"] = filtered_df["similarity"] * 0.6 + fuzzy_scores * 0.4

        results = filtered_df.sort_values(by="score", ascending=False).head(3)

        bot_response = "✅ **Top matches:**<br><br>"

        for _, row in results.iterrows():
            bot_response += f"""
<div style="background:#fff;padding:12px;border-radius:12px;margin-bottom:10px;
box-shadow:0 2px 8px rgba(0,0,0,0.08); border-left:5px solid #4F8BF9;">
<b>📊 {row['Dashboard Name']}</b><br>
<small>LOB: {row['LOB']}</small><br>
{row['Link']}
</div>
"""

    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response, unsafe_allow_html=True)
