import streamlit as st
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import re
import json
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict


class TherapyBot:
    def __init__(self):
        # Get API key from Streamlit secrets
        if 'groq_api_key' not in st.secrets:
            st.error("Groq API key not found in secrets. Please add it to your secrets.toml file.")
            st.stop()
            
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            groq_api_key=st.secrets["groq_api_key"]
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.system_prompt = """You are Aria, an empathetic AI therapeutic assistant. Your role is to:
        1. Listen actively and show understanding of users' emotions
        2. Provide supportive, non-judgmental responses
        3. Ask clarifying questions when needed
        4. Offer coping strategies and gentle guidance
        5. Recognize crisis situations and direct to professional help
        6. Maintain boundaries and clarify you're an AI assistant, not a replacement for professional therapy
        7. Keep conversations confidential and respect privacy
        8. Focus on emotional support and practical coping strategies

        Important guidelines:
        - Never provide medical advice or diagnoses
        - Always take mentions of self-harm or suicide seriously
        - Maintain a warm, professional tone
        - Ask open-ended questions
        - Reflect back what you hear to show understanding
        - Always refer to yourself as "Aria" in your responses

        Emergency resources to provide when needed:
        - National Mental Health Helpline (India): 14416 or 1-800-599-0019
        - AASRA (India): +91-9820466726 or +91-22-27546669
        - Vandrevala Foundation Helpline (India): 1860 266 2345 or +91 9999 666 555"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
        )

        self.mood_history = []

    def detect_crisis(self, message: str) -> bool:
        crisis_keywords = [
            "suicide", "kill myself", "end it all", "self harm",
            "hurt myself", "don't want to live", "better off dead",
            "take my life", "end my life", "give up", "no way out",
            "life is pointless", "worthless", "I can't go on",
            "can't take this anymore", "hopeless", "I feel trapped",
            "I want to disappear", "I hate myself", "I don't matter",
            "no one cares", "I want to die", "I'm done", "this is too much",
            "can't handle this", "overwhelmed", "can't breathe",
            "crying all the time", "breaking down", "shattered",
            "why am I alive", "wish I wasn't here", "empty inside",
            "numb", "in pain", "drowning", "sinking", "lost",
            "alone", "unloved", "no one listens", "isolated"
        ]
        return any(keyword in message.lower() for keyword in crisis_keywords)

    def get_crisis_response(self) -> str:
        return """I'm very concerned about what you're telling me and I want to make sure you're safe.
        Please know that you're not alone and there are people who want to help:

        National Mental Health Helpline (India): 14416 or 1-800-599-0019
        AASRA (India): +91-9820466726 or +91-22-27546669
        Vandrevala Foundation Helpline (India): 1860 266 2345 or +91 9999 666 555

        Would you be willing to reach out to one of these services? They have trained professionals who can provide immediate support."""

    def track_mood(self, message: str, response: str):
        timestamp = datetime.now().isoformat()
        mood_indicators = {
            'positive': ['happy', 'better', 'good', 'great', 'hopeful', 'excited', 'joyful', 'content', 'optimistic'],
            'negative': ['sad', 'depressed', 'anxious', 'worried', 'stressed', 'hopeless', 'down', 'angry',
                         'frustrated'],
            'neutral': ['okay', 'fine', 'normal', 'meh', 'so-so', 'neutral', 'indifferent']
        }

        mood = 'neutral'
        for sentiment, words in mood_indicators.items():
            if any(word in message.lower() for word in words):
                mood = sentiment
                break

        self.mood_history.append({
            'timestamp': timestamp,
            'mood': mood,
            'message': message
        })

    def get_response(self, message: str) -> str:
        if self.detect_crisis(message):
            return self.get_crisis_response()

        response = self.conversation({"input": message})
        self.track_mood(message, response['text'])

        return response['text']

    def get_mood_summary(self) -> Dict:
        if not self.mood_history:
            return {"message": "No mood data available yet"}

        df = pd.DataFrame(self.mood_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Get mood counts
        mood_counts = df['mood'].value_counts().to_dict()

        # Get mood trends (last 7 entries)
        recent_moods = df.tail(7)['mood'].tolist()

        # Calculate mood change
        mood_values = {'positive': 1, 'neutral': 0, 'negative': -1}
        if len(df) >= 2:
            last_mood = df.iloc[-1]['mood']
            previous_mood = df.iloc[-2]['mood']
            mood_change = mood_values[last_mood] - mood_values[previous_mood]
        else:
            mood_change = 0

        return {
            "total_entries": len(df),
            "mood_distribution": mood_counts,
            "latest_mood": df.iloc[-1]['mood'] if not df.empty else "No data",
            "recent_moods": recent_moods,
            "mood_change": mood_change,
            "timestamp_data": df[['timestamp', 'mood']].to_dict('records')
        }


def create_mood_timeline(mood_data):
    if not mood_data:
        return None

    df = pd.DataFrame(mood_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    color_map = {
        'positive': '#4CAF50',
        'neutral': '#FFC107',
        'negative': '#f44336'
    }

    fig = px.line(df, x='timestamp', y='mood',
                  title='Mood Timeline',
                  color_discrete_map=color_map)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        title_font_color='#ffffff',
        height=300,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        )
    )

    return fig


def create_mood_distribution_pie(mood_distribution):
    if not mood_distribution:
        return None

    colors = {
        'positive': '#4CAF50',
        'neutral': '#FFC107',
        'negative': '#f44336'
    }

    fig = go.Figure(data=[go.Pie(
        labels=list(mood_distribution.keys()),
        values=list(mood_distribution.values()),
        marker_colors=[colors[mood] for mood in mood_distribution.keys()],
        textinfo='percent+label',
        hole=0.5,
    )])

    fig.update_layout(
        title='Mood Distribution',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        title_font_color='#ffffff'
    )

    return fig


def main():
    st.set_page_config(
        page_title="Aria - AI Therapy Assistant",
        page_icon="🤗",
        layout="wide"
    )
    st.markdown("""
        <style>
        /* Main app styling */
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }

        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Chat container styling */
        .chat-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }

        /* Message styling */
        .message {
            padding: 15px;
            margin: 10px 0;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            animation: glow 2s ease-in-out infinite alternate;
        }

        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin-left: 20px;
            color: white;
        }

        .assistant-message {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            margin-right: 20px;
            color: white;
        }

        /* Analysis card styling */
        .analysis-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: white;
        }

        /* Mood indicator styling */
        .mood-indicator {
            text-align: center;
            padding: 15px;
            border-radius: 15px;
            margin: 10px 0;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }

        /* Glowing animations */
        @keyframes glow {
            from {
                box-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #0073e6;
            }
            to {
                box-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #0073e6;
            }
        }

        /* Input styling */
        .stTextInput > div > div {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }

        .stTextInput > div > div:focus {
            box-shadow: 0 0 10px rgba(255,255,255,0.3);
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }

        /* Chart container styling */
        .chart-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        /* Override Streamlit's default text colors */
        .stMarkdown, .stText {
            color: white !important;
        }

        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns for main layout
    col1, col2 = st.columns([7, 3])

    with col1:
        st.markdown('<h1 class="main-header">Aria - AI Therapy Assistant</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color: white; text-align: center; font-size: 1.2rem;">Your safe space for conversation and support</p>',
            unsafe_allow_html=True)

        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            # Check if the API key is available in secrets before initializing the bot
            if 'groq_api_key' in st.secrets:
                st.session_state.bot = TherapyBot()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I'm Aria, and I'm here to listen and support you. How are you feeling today?"
                })
            else:
                st.error("Groq API key not found in secrets. Please add it to your secrets.toml file.")
                st.stop()

        # Chat container
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(
                        f'<div class="message {message["role"]}-message">{message["content"]}</div>',
                        unsafe_allow_html=True
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Share your thoughts..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="message user-message">{prompt}</div>',
                    unsafe_allow_html=True
                )

            response = st.session_state.bot.get_response(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="message assistant-message">{response}</div>',
                    unsafe_allow_html=True
                )

    # Analysis Column
    with col2:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: white; text-align: center;">Mood Analysis</h2>', unsafe_allow_html=True)

        if 'bot' in st.session_state:
            mood_summary = st.session_state.bot.get_mood_summary()
            if "message" in mood_summary:
                st.info(mood_summary["message"])
            else:
                # Current Mood Indicator
                latest_mood = mood_summary['latest_mood']
                mood_colors = {
                    'positive': '#4CAF50',
                    'neutral': '#FFC107',
                    'negative': '#f44336'
                }
                mood_emojis = {
                    'positive': '😊',
                    'neutral': '😐',
                    'negative': '😔'
                }

                st.markdown(
                    f'<div class="mood-indicator" style="background: linear-gradient(135deg, {mood_colors[latest_mood]}33, {mood_colors[latest_mood]}11)">'
                    f'<h3>Current Mood {mood_emojis[latest_mood]}</h3>'
                    f'<p style="font-size: 1.2rem;">{latest_mood.capitalize()}</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Mood change indicator
                mood_change = mood_summary['mood_change']
                if mood_change > 0:
                    st.markdown(
                        '<div class="mood-indicator" style="background: rgba(76, 175, 80, 0.1)">Mood trending upward 📈</div>',
                        unsafe_allow_html=True)
                elif mood_change < 0:
                    st.markdown(
                        '<div class="mood-indicator" style="background: rgba(244, 67, 54, 0.1)">Mood trending downward 📉</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="mood-indicator" style="background: rgba(255, 193, 7, 0.1)">Mood stable ➡</div>',
                        unsafe_allow_html=True)

                # Visualizations
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### Mood Distribution")
                pie_chart = create_mood_distribution_pie(mood_summary['mood_distribution'])
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### Mood Timeline")
                timeline = create_mood_timeline(mood_summary['timestamp_data'])
                if timeline:
                    st.plotly_chart(timeline, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Session Stats
                st.markdown(
                    f'<div class="mood-indicator">'
                    f'<h3>Session Stats</h3>'
                    f'<p>Total messages: {mood_summary["total_entries"]}</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()