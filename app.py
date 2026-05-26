import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import webbrowser

# ✅ Page config
st.set_page_config(page_title="Dashboard Chatbot", layout="centered")

st.title("🤖 Dashboard Chatbot")
st.info("💡 Type 'help' to see what you can ask.")

# ✅ Load CSV
df = pd.read_csv("dashboards.csv")

# Combine columns for better search
df["combined"] = df["Dashboard Name"] + " " + df["LOB"] + " dashboard report data"

# ✅ Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["combined"])

# ✅ Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ✅ Chat input (PROMPT BOX)
user_input = st.chat_input("Ask for dashboard link...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # ✅ PROCESS INPUT
    user_lower = user_input.lower()

    # ✅ Greeting
    if any(word in user_lower for word in ["hi", "hello", "hey"]):
        bot_response = """Hello! 👋  
I can help you find dashboard links.  
Type **help** to see what you can ask.
"""

    # ✅ Help command
    elif "help" in user_lower:
        bot_response = """
📖 **Here’s how you can use me:**

🔍 Ask by dashboard name:
• MNR Dashboard  

📊 Ask by LOB:
• UHC Mobile  
 
🔗 Ask for links:
• Give me MNR dashboard link  

💡 Tips:
• Be specific for better results  
• Example: "UHC Mobile Executive dashboard link"
"""

    # ✅ AI Matching Logic
    else:
        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)

        if similarity.max() < 0.3:
            bot_response = "❌ No matching dashboard found. Try typing **help**."
        else:
            index = similarity.argmax()
            result = df.iloc[index]

            bot_response = f"""
✅ **Dashboard:** {result['Dashboard Name']}  
📊 **LOB:** {result['LOB']}  
🔗 **Link:** {result['Link']}
"""

            # ✅ Optional: auto open link
            webbrowser.open(result["Link"])

    # ✅ Show bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response)