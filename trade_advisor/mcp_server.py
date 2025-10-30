import streamlit as st
import requests, yfinance as yf, json
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
from groq import Groq
from serpapi.google_search import GoogleSearch
from sklearn.pipeline import Pipeline
import joblib
class_model = joblib.load("trade_advisor/trade_text_classifier.pkl")

# ----------------------------- API KEYS -----------------------------
TWELVE_DATA_API_KEY = st.secrets(["apis"]["data"])
FINNHUB_API_KEY = st.secrets(["apis"]["finhub"])  
SERPAPI_KEY = st.secrets(["apis"]["serp_api"])         
# ----------------------------- Groq Client -----------------------------
client = Groq(api_key=st.secrets(["apis"]["groq"]))

chat_history = [
    {"role": "system", "content": "You are Valkon AI a financial stock Advisor."}
]

def query_llm(prompt: str) -> str:
    """Query Groq's LLaMA-3.1-8B model with chat history."""
    global chat_history
    try:
        # Add new user message to history
        chat_history.append({"role": "user", "content": prompt})
        if len(chat_history) > 5:
            chat_history = chat_history[-5:]
        # Send the full conversation (context) to LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_history,
            stream=False
        )

        # Get assistant reply
        reply = response.choices[0].message.content.strip()

        # Add assistant reply to history
        chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        return f"⚠️ LLM Error: {e}"

# ----------------------------- Fetch Stock Context -----------------------------
def fetch_context(symbol: str):
    """Fetch live stock data, history, and news."""
    try:
        ticker = yf.Ticker(symbol)
        price_data = ticker.history(period="1d")
        if not price_data.empty:
            stock_price = round(price_data["Close"].iloc[-1],3)
        else:
            quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            quote_data = requests.get(quote_url, timeout=10).json()
            stock_price = quote_data.get("c", "N/A")
    except Exception:
        stock_price = "N/A"

    try:
        df = yf.download(symbol, period="1y").tail(30).reset_index()
        df["Date"] = df["Date"].astype(str)
        df = df.replace({np.nan: None})
        historical_data30 = df.to_dict(orient="records")
        historical_data = df.tail(10).to_dict(orient="records")
        company_info = yf.Ticker(symbol).info
        company_name = company_info.get("longName", symbol)

        news_url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"
        news_list = requests.get(news_url, timeout=10).json()[:5]

        simplified_news = []
        for n in news_list:
            headline = n.get("headline", "")
            summary = n.get("summary", "")
            sentiment = TextBlob(headline + " " + summary).sentiment.polarity
            simplified_news.append({
                "headline": headline,
                "summary": summary,
                "sentiment": float(sentiment)
            })

        return {
            "stock": stock_price,
            "historical30":historical_data30,
            "historical": historical_data,
            "company_name": company_name,
            "news": simplified_news,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

# ----------------------------- Analyze Stock -----------------------------
def analyze_stock(symbol: str, context):
    price = context["stock"]
    prompt = f"""
    Stock Symbol: {symbol}
    Current Price: {price}
    Historical Last 10 Days: {context['historical']}
    Latest News Headlines:
    """
    for news in context['news']:
        prompt += f"- {news['headline']} | {news['summary']} | Sentiment: {news['sentiment']}\n"
    prompt += "\nProvide a BUY/SELL/HOLD recommendation with confidence percentage and reasoning."
    return query_llm(prompt)

# ----------------------------- Google Search Function -----------------------------
def google_search(query):
    """Search the web using SerpAPI."""
    try:
        search = GoogleSearch({
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY
        })
        results = search.get_dict()
        summaries = []
        for r in results.get("organic_results", [])[:5]:
            summaries.append(f"{r.get('title')}: {r.get('snippet')}")
        return summaries
    except Exception as e:
        return [f"⚠️ Google Search Error: {e}"]

# ----------------------------- Chat with LLM -----------------------------
def chat_with_llm(user_question):
    """Fetch Google news + stock data + feed to LLM."""
    try:
        # extract possible stock symbol (simple guess)
        words = user_question.split()
        symbol_guess = None
        for w in words:
            if w.isupper() and len(w) <= 6:
                symbol_guess = w + ".NS"
                break

        # fetch context if stock symbol detected
        stock_context = fetch_context(symbol_guess) if symbol_guess else None
        google_results = google_search(f"news related to stock {user_question}")

        prompt = f"""
        You are Valkon AI a financial stock Advisor.
        User Question: {user_question}

        Relevant Google Results:
        {json.dumps(google_results, indent=2)}

        Stock Data (if available):
        {json.dumps(stock_context, indent=2) if stock_context else "No symbol found"}

        Based on this info, give a clear, concise answer.
        """
        return query_llm(prompt)

    except Exception as e:
        return f"⚠️ Chat Error: {e}"

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Valkon AI - Trade Advisor", page_icon="📈", layout="wide")
st.title("Valkontek Embedded & IOT services PVT.LTD")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a section:", ["📊 Stock Advisor", "💬 Chat with LLM"])

# ----------------------------- PAGE 1: Stock Advisor -----------------------------
if page == "📊 Stock Advisor":
    st.title("Valkon AI — Stock Advisor")
    st.write("Your personal AI-powered assistant for live stock insights and smart trading decisions.")

    symbol = st.text_input("Enter Stock Symbol (e.g. INFY.NS, TCS.NS, RELIANCE.NS):", "RELIANCE.NS")
    stock = st.selectbox("Select Stock:", ["NSE"])
    if stock == "NSE" and not symbol.endswith(".NS"):
        symbol = symbol + ".NS"
    

    if st.button("🔍 Fetch & Analyze"):
        with st.spinner("Fetching live data..."):
            context = fetch_context(symbol)
            if "error" in context:
                st.error(f"❌ Error: {context['error']}")
            else:
                st.success(f"✅ Data fetched successfully (Last updated: {context['last_updated']})")
                company_name = context["company_name"]
                st.subheader(company_name)

                price = context["stock"]
                st.metric(label=f"💰 {symbol} — Current Price", value=f"₹{price}")

                st.subheader("📉 Last 10 Days Price Data")
                hist_df = pd.DataFrame(context["historical"])
                st.dataframe(hist_df)


                st.subheader("📰 Latest News & Sentiments")
                for news in context["news"]:
                    sentiment = news['sentiment']
                    sentiment_emoji = "🟢" if sentiment > 0 else ("🔴" if sentiment < 0 else "⚪")
                    st.markdown(f"**{sentiment_emoji} {news['headline']}** — {news['summary']}")
                    st.caption(f"Sentiment Score: {sentiment:.2f}")

                st.subheader("🤖 AI Stock Recommendation")
                with st.spinner("Analyzing with AI..."):
                    llm_output = analyze_stock(symbol, context)
                    st.write(llm_output)

                    prompt2 = f"""
                    stock:{company_name}
                    info:
                    {llm_output}\n Draft this info as an alert message like given format below.no need extra matter.
                    format:
                    Buy Alert: Reliance Industries
                    At Price: 72.46
                    Confidence: 87% | Expected Gain: +5.2%
                    Reason: Positive Q2 results + strong RSI trend.
                    Suggested Stop Loss: ₹2560
                    """
                    st.subheader("🔔 Alert Message")
                    st.write(query_llm(prompt2))

# ----------------------------- PAGE 2: Chat with LLM -----------------------------
elif page == "💬 Chat with LLM":
    st.title("💬 Chat with Advisor")
    st.write("Ask anything about market trends, stock insights, or global financial news. I’ll fetch the latest data before answering!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask your question:")

    if user_input:
        if user_input.strip():
            with st.spinner("Fetching live info and analyzing..."):
                pred=class_model.predict([user_input])
                if pred=="required research":
                    answer = chat_with_llm(user_input)
                else:
                    answer=query_llm(user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Valkon AI", answer))

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"🧑‍💻 **You:** {msg}")
        else:
            st.markdown(f"🤖 **Valkon AI:** {msg}")

st.markdown("---")
st.caption("Powered by Valkon AI 🚀")


