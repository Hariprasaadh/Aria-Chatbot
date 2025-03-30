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
            
        # Initialize main conversation LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            groq_api_key=st.secrets["groq_api_key"]
        )
        
        # Initialize sentiment analysis LLM with lower temperature for consistency
        self.sentiment_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            groq_api_key=st.secrets["groq_api_key"]
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Main conversation prompt
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

        # Sentiment analysis prompt
        self.sentiment_prompt = """You are an AI specialized in analyzing the emotional content and sentiment of therapy messages.

        Analyze the following message and output a JSON object with these fields:
        - mood: Must be exactly one of ["positive", "negative", "neutral"]
        - primary_emotion: The most dominant specific emotion (e.g., joy, sadness, anger, anxiety, hope, gratitude, etc.)
        - intensity: A number from 1-10 indicating how strong the emotion is (1=barely detectable, 10=extremely intense)
        - sentiment_reasons: Brief explanation of why you classified it this way (max 2 sentences)

        Focus on the overall emotional tone. Consider both explicit statements of feelings and subtler indications of the person's emotional state.
        
        Your output must be valid JSON only without any other text or explanation.
        """

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
        self.crisis_threshold = 9  # Intensity threshold to trigger crisis response

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

    def analyze_sentiment(self, message: str) -> Dict:
        """Uses LLM to analyze the sentiment of a message"""
        try:
            # Create the full prompt for sentiment analysis
            sentiment_input = f"Message to analyze: {message}\n\nProvide sentiment analysis in JSON format."
            
            # Get sentiment analysis from LLM
            sentiment_response = self.sentiment_llm.invoke([
                {"role": "system", "content": self.sentiment_prompt},
                {"role": "user", "content": sentiment_input}
            ])
            
            # Extract the JSON from the response
            json_str = sentiment_response.content
            
            # Clean the response if it contains markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
                
            # Parse JSON
            sentiment_data = json.loads(json_str)
            
            # Ensure we have all required fields
            required_fields = ["mood", "primary_emotion", "intensity", "sentiment_reasons"]
            for field in required_fields:
                if field not in sentiment_data:
                    sentiment_data[field] = "unknown" if field != "intensity" else 5
                    
            return sentiment_data
            
        except Exception as e:
            # Fallback if any error occurs
            st.error(f"Error in sentiment analysis: {str(e)}")
            return {
                "mood": "neutral",
                "primary_emotion": "unknown",
                "intensity": 5,
                "sentiment_reasons": "Error in sentiment analysis"
            }

    def track_mood(self, message: str, response: str):
        timestamp = datetime.now().isoformat()
        
        # Use LLM to analyze sentiment
        sentiment_data = self.analyze_sentiment(message)
        
        self.mood_history.append({
            'timestamp': timestamp,
            'mood': sentiment_data["mood"],
            'primary_emotion': sentiment_data["primary_emotion"],
            'intensity': sentiment_data["intensity"],
            'sentiment_reasons': sentiment_data["sentiment_reasons"],
            'message': message
        })

    def get_response(self, message: str) -> str:
        # First, analyze sentiment to detect potential crisis
        sentiment_data = self.analyze_sentiment(message)
        
        # Check for crisis either by keywords or high intensity negative emotions
        is_crisis = (self.detect_crisis(message) or 
                    (sentiment_data["mood"] == "negative" and 
                     sentiment_data["intensity"] >= self.crisis_threshold))
        
        if is_crisis:
            return self.get_crisis_response()

        response = self.conversation({"input": message})
        
        # Track mood after getting response
        self.track_mood(message, response['text'])

        return response['text']

    def get_mood_summary(self) -> Dict:
        if not self.mood_history:
            return {"message": "No mood data available yet"}

        df = pd.DataFrame(self.mood_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get mood counts
        mood_counts = df['mood'].value_counts().to_dict()
        
        # Get emotion distribution
        emotion_counts = df['primary_emotion'].value_counts().to_dict()
        
        # Calculate average intensity
        avg_intensity = df['intensity'].mean() if 'intensity' in df else 5
        
        # Get recent moods and emotions
        recent_entries = df.tail(7)
        recent_moods = recent_entries['mood'].tolist()
        recent_emotions = recent_entries['primary_emotion'].tolist()
        
        # Calculate mood change trend
        mood_values = {'positive': 1, 'neutral': 0, 'negative': -1}
        
        # For overall trend calculation
        if len(df) >= 3:
            # Get last three entries for trend
            last_three = df.tail(3)['mood'].map(mood_values).tolist()
            if last_three[0] < last_three[1] < last_three[2]:
                mood_trend = "improving"
            elif last_three[0] > last_three[1] > last_three[2]:
                mood_trend = "declining"
            else:
                mood_trend = "fluctuating"
        else:
            mood_trend = "insufficient data"
            
        # For simple last change
        if len(df) >= 2:
            last_mood = df.iloc[-1]['mood']
            previous_mood = df.iloc[-2]['mood']
            mood_change = mood_values.get(last_mood, 0) - mood_values.get(previous_mood, 0)
        else:
            mood_change = 0
            
        # Generate insights
        insights = []
        if len(df) >= 3:
            # Check if all recent moods are the same
            if len(set(recent_moods[-3:])) == 1:
                if recent_moods[-1] == 'positive':
                    insights.append("Consistently positive mood in recent messages")
                elif recent_moods[-1] == 'negative':
                    insights.append("Consistently negative mood in recent messages")
            
            # Check for mood swings
            if 'positive' in recent_moods and 'negative' in recent_moods:
                insights.append("Mood fluctuations detected")
                
            # Check intensity trends
            if 'intensity' in df.columns:
                recent_intensities = recent_entries['intensity'].tolist()
                if sum(recent_intensities[-3:]) / 3 > 7:
                    insights.append("High emotional intensity in recent messages")

        return {
            "total_entries": len(df),
            "mood_distribution": mood_counts,
            "emotion_distribution": emotion_counts,
            "latest_mood": df.iloc[-1]['mood'] if not df.empty else "No data",
            "latest_emotion": df.iloc[-1]['primary_emotion'] if not df.empty else "No data",
            "latest_intensity": float(df.iloc[-1]['intensity']) if not df.empty and 'intensity' in df else 5,
            "average_intensity": float(avg_intensity),
            "recent_moods": recent_moods,
            "recent_emotions": recent_emotions,
            "mood_change": mood_change,
            "mood_trend": mood_trend,
            "insights": insights,
            "timestamp_data": df[['timestamp', 'mood', 'primary_emotion', 'intensity']].to_dict('records')
        }


    def create_mood_timeline(mood_data):
    if not mood_data:
        return None

    df = pd.DataFrame(mood_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Map categorical moods to numeric values for plotting
    mood_value_map = {
        'positive': 3,
        'neutral': 2,
        'negative': 1
    }
    
    # Create a numerical column for mood
    df['mood_value'] = df['mood'].map(mood_value_map)

    fig = px.line(df, x='timestamp', y='mood_value',
                  title='Mood Timeline')
    
    # Customize y-axis to show original mood labels
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['negative', 'neutral', 'positive']
        )
    )
    
    # Add intensity as a secondary axis if available
    if 'intensity' in df.columns:
        fig.add_scatter(
            x=df['timestamp'],
            y=df['intensity'],
            mode='lines+markers',
            name='Emotional Intensity',
            yaxis="y2",
            line=dict(color='#E91E63', width=1, dash='dot'),
            marker=dict(size=8)
        )
        
        # Add secondary y-axis with corrected properties
        fig.update_layout(
            yaxis2=dict(
                title="Intensity",
                title_font=dict(color='#E91E63'),  # Corrected from titlefont to title_font
                tickfont=dict(color='#E91E63'),
                anchor="x",
                overlaying="y",
                side="right",
                range=[0, 10],
                showgrid=False
            )
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        title_font_color='#ffffff',  # Corrected from title_font_color
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
                  
    # Add intensity as a secondary axis if available
    if 'intensity' in df.columns:
        fig.add_scatter(
            x=df['timestamp'],
            y=df['intensity'],
            mode='lines+markers',
            name='Emotional Intensity',
            yaxis="y2",
            line=dict(color='#E91E63', width=1, dash='dot'),
            marker=dict(size=8)
        )
        
        # Add secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title="Intensity",
                titlefont=dict(color='#E91E63'),
                tickfont=dict(color='#E91E63'),
                anchor="x",
                overlaying="y",
                side="right",
                range=[0, 10],
                showgrid=False
            )
        )

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
            tickfont=dict(color='#ffffff'),
            categoryorder='array',
            categoryarray=['negative', 'neutral', 'positive']
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
        marker_colors=[colors.get(mood, '#9C27B0') for mood in mood_distribution.keys()],
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


def create_emotion_bar_chart(emotion_distribution):
    if not emotion_distribution:
        return None
        
    # Sort by frequency
    sorted_emotions = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)
    emotions = [item[0] for item in sorted_emotions]
    counts = [item[1] for item in sorted_emotions]
    
    # Use a colorful palette
    colors = px.colors.qualitative.Set3
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions, 
            y=counts,
            marker_color=colors[:len(emotions)]
        )
    ])
    
    fig.update_layout(
        title='Primary Emotions',
        xaxis_title='Emotion',
        yaxis_title='Frequency',
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        title_font_color='#ffffff',
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='#ffffff')
        )
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
        
        /* Insight styling */
        .insight-item {
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #9C27B0;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 0 8px 8px 0;
        }
        
        /* Intensity meter styling */
        .intensity-meter {
            height: 8px;
            border-radius: 4px;
            margin: 8px 0;
            background: linear-gradient(to right, #4CAF50, #FFC107, #f44336);
        }
        
        .intensity-marker {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: white;
            border: 2px solid #333;
            position: relative;
            top: -10px;
            transform: translateX(-50%);
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
        st.markdown('<h2 style="color: white; text-align: center;">Emotional Analysis</h2>', unsafe_allow_html=True)

        if 'bot' in st.session_state:
            mood_summary = st.session_state.bot.get_mood_summary()
            if "message" in mood_summary:
                st.info(mood_summary["message"])
            else:
                # Current Mood and Emotion Indicator
                latest_mood = mood_summary['latest_mood']
                latest_emotion = mood_summary['latest_emotion']
                latest_intensity = mood_summary['latest_intensity']
                
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
                    f'<p style="font-size: 1.2rem;">{latest_emotion.capitalize()} ({latest_mood.capitalize()})</p>'
                    f'<div class="intensity-meter"></div>'
                    f'<div class="intensity-marker" style="margin-left: {latest_intensity * 10}%;"></div>'
                    f'<p style="font-size: 0.9rem; margin-top: 5px;">Intensity: {latest_intensity}/10</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Mood trend indicator
                mood_trend = mood_summary['mood_trend']
                trend_icons = {
                    "improving": "📈 Improving",
                    "declining": "📉 Declining",
                    "fluctuating": "📊 Fluctuating",
                    "insufficient data": "📋 Insufficient data"
                }
                
                st.markdown(
                    f'<div class="mood-indicator" style="background: rgba(255, 255, 255, 0.1)">Mood trend: {trend_icons[mood_trend]}</div>',
                    unsafe_allow_html=True)

                # Insights
                if mood_summary['insights']:
                    st.markdown("### Insights")
                    for insight in mood_summary['insights']:
                        st.markdown(
                            f'<div class="insight-item">{insight}</div>',
                            unsafe_allow_html=True
                        )

                # Visualizations
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### Mood Distribution")
                pie_chart = create_mood_distribution_pie(mood_summary['mood_distribution'])
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### Emotion Analysis")
                emotion_chart = create_emotion_bar_chart(mood_summary['emotion_distribution'])
                if emotion_chart:
                    st.plotly_chart(emotion_chart, use_container_width=True)
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
                    f'<p>Average emotional intensity: {mood_summary["average_intensity"]:.1f}/10</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()