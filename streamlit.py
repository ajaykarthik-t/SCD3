import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import plotly.graph_objects as go

class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x

def load_model():
    try:
        # Load preprocessing objects
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        scaler = joblib.load('feature_scaler.joblib')

        # Load model
        input_dim = 56  # 6 numerical features + 50 text features
        model = SpamClassifier(input_dim)
        model.load_state_dict(torch.load('spam_classifier_model.pth'))
        model.eval()

        return model, vectorizer, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "lightyellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        },
        title={'text': title}
    ))

    fig.update_layout(height=250)
    return fig

def main():
    st.set_page_config(page_title="Spam Call Detector", layout="wide")

    # Header
    st.title("🎭 Spam Call Detector")
    st.markdown("""
        Enter either a phone number or conversation text to analyze the likelihood of a spam call.
    """)

    # Load model
    model, vectorizer, scaler = load_model()

    if not all([model, vectorizer, scaler]):
        st.error("Failed to load model components. Please ensure all model files are present.")
        return

    # Input Section
    st.subheader("📝 Input Details")
    phone_number = st.text_input("Phone Number", placeholder="Enter phone number here...")
    conversation_text = st.text_area("Conversation Content", placeholder="Enter the conversation or message content here...", height=100)

    analyze_button = st.button("🔍 Analyze")

    if analyze_button:
        with st.spinner("Analyzing input..."):
            # Dummy phone number validation logic
            known_phone_numbers = {"1234567890": "spam", "9876543210": "not_spam"}  # Example lookup

            if phone_number.strip() in known_phone_numbers:
                result = known_phone_numbers[phone_number.strip()]
                spam_prob = 0.9 if result == "spam" else 0.1
            else:
                if conversation_text.strip():
                    # Process conversation text
                    text_features = vectorizer.transform([conversation_text]).toarray()
                    dummy_numerical_features = np.zeros((1, 6))  # Placeholder numerical features

                    features = np.hstack([dummy_numerical_features, text_features])

                    # Convert to tensor and predict
                    with torch.no_grad():
                        features_tensor = torch.FloatTensor(features)
                        prediction = model(features_tensor)
                        spam_prob = float(prediction[0])
                else:
                    st.error("Please provide either a valid phone number or conversation text.")
                    return

            # Display Results
            st.plotly_chart(create_gauge_chart(spam_prob, "Spam Probability"), use_container_width=True)

            if spam_prob > 0.7:
                st.error("🚨 High probability of being a spam call! Exercise caution.")
            elif spam_prob > 0.4:
                st.warning("⚠️ Moderate risk of being a spam call. Proceed with caution.")
            else:
                st.success("✅ Likely a legitimate call.")

if __name__ == "__main__":
    main()
