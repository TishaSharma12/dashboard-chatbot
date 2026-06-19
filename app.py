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
# ✅ NORMALIZATION
# =========================

def normalize_text(text):
    text = text.lower()

    # dot handling
    text = text.replace("dotn", ".")
    text = text.replace("dot", ".")

    # LOB conversions
    text = text.replace("m&r", "mnr")
    text = text.replace("c&s", "cns")
    text = text.replace("e&i", "eni")

    # remove special chars
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

# ✅ Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["combined"])

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

user_input = st.chat_input("Ask like: 'MNR executive dashboard'")

# =========================
# ✅ CHATBOT
# =========================

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    user_lower = user_input.lower()
    user_clean = normalize_text(user_input)

    # ✅ GREETING (IMPROVED)
    if any(word in user_lower for word in ["hi", "hello", "hey"]):
        bot_response = """
👋 **Hello! Welcome to your Dashboard Assistant**

I can help you quickly find the right dashboards.

### 🚀 Try asking:
• 📊 *"CNS dashboards"*  
• 📈 *"MNR executive dashboards"*  
• 📑 *"Show me ENI reports"*  

### 💡 Tip:
Just type naturally — like:  
*"show me MNR executive dashboard"*

Or use filters on the left 👉
"""

    # ✅ HELP (IMPROVED)
    elif "help" in user_lower:
        bot_response = """
📖 **Here’s how you can use me:**

🔍 **Search by dashboard name:**
• MNR Dashboard  
• Revenue Report  

📊 **Search by LOB (supports variations):**
• CNS → also works for C&S  
• MNR → also works for M&R  
• ENI → also works for E&I  

🔗 **Ask for dashboards naturally:**
• "Give me MNR dashboard link"  
• "CNS executive dashboards"  

🎯 **Smart behavior:**
• Only LOB → shows ALL dashboards  
• LOB + keywords → filtered + ranked results  

💡 **Examples:**
• "CNS executive dashboard"  
• "optumdotcom report"
"""

    else:

        # ✅ Detect LOB inside query
        detected_lob = None
        for lob in LOB_LIST:
            if lob in user_clean:
                detected_lob = lob
                break

        # ✅ Apply filtering logic
        if detected_lob:
            filtered_df = df[df["LOB_clean"] == detected_lob].copy()
        elif selected_lob != "All":
            filtered_df = df[df["LOB_clean"] == selected_lob_clean].copy()
        else:
            filtered_df = df.copy()

        # ✅ PURE LOB CASE
        if detected_lob and len(user_clean) <= len(detected_lob) + 2:

            bot_response = f"📊 **All dashboards for {detected_lob.upper()}**<br><br>"

            for _, row in filtered_df.iterrows():
                bot_response += f"""
<div style="background:#fff;padding:12px;border-radius:12px;margin-bottom:10px;
box-shadow:0 2px 8px rgba(0,0,0,0.08); border-left:5px solid #4F8BF9;">
<b>📊 {row['Dashboard Name']}</b><br>
<small>LOB: {row['LOB']}</small><br>
{row['Link']}
</div>
"""

        else:
            # ✅ SMART SEARCH
            user_vec = vectorizer.transform([user_input])
            similarity = cosine_similarity(user_vec, X).flatten()

            filtered_df["similarity"] = similarity[filtered_df.index]

            fuzzy_scores = filtered_df["combined"].apply(
                lambda x: fuzz.token_set_ratio(user_input, x) / 100
            )

            filtered_df["score"] = filtered_df["similarity"] * 0.6 + fuzzy_scores * 0.4

            results = filtered_df.sort_values(by="score", ascending=False).head(3)

            if detected_lob:
                bot_response = f"📊 **Top {detected_lob.upper()} dashboards:**<br><br>"
            else:
                bot_response = "✅ **Top Dashboard Matches:**<br><br>"

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
