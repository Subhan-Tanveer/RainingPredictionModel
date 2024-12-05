import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

# Add page title
st.title("🌦️ Rainfall Prediction App")
st.markdown("**Predict whether it will rain based on weather conditions!** ☔")

# Background audio for .wav file with autoplay and looping
audio_file_path = "mixkit-rain-and-thunder-storm-2390.wav"  # Update this path as needed
with open(audio_file_path, "rb") as audio_file:
    st.audio(audio_file, format="audio/wav", start_time=0, loop=True)

# CSS to hide the audio controls and autoplay in the background
page_bg_audio = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.gifer.com/N8i8.gif");
    background-size: cover;
    background-position: center;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0);
}

/* Hide the audio controls */
audio {
    opacity: 0;
    pointer-events: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
}
</style>
"""
st.markdown(page_bg_audio, unsafe_allow_html=True)

# Input section header
st.header("🌤️ Enter the Weather Details Below")

# Input fields for user with emojis
pressure = st.number_input("🌀 Pressure (hPa)", min_value=900.0, max_value=1050.0, value=1015.9, step=0.1)
dewpoint = st.number_input("💧 Dewpoint (°C)", min_value=-20.0, max_value=40.0, value=19.9, step=0.1)
humidity = st.number_input("🌫️ Humidity (%)", min_value=0.0, max_value=100.0, value=95.0, step=1.0)
cloud = st.number_input("☁️ Cloud Coverage (%)", min_value=0.0, max_value=100.0, value=81.0, step=1.0)
sunshine = st.number_input("☀️ Sunshine Duration (hours)", min_value=0.0, max_value=24.0, value=0.0, step=0.1)
winddirection = st.number_input("🧭 Wind Direction (°)", min_value=0.0, max_value=360.0, value=40.0, step=1.0)
windspeed = st.number_input("🌬️ Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=13.7, step=0.1)

# Create a DataFrame from the input data
input_data = [pressure, dewpoint, humidity, cloud, sunshine, winddirection, windspeed]
input_df = pd.DataFrame([input_data], columns=feature_names)

# Predict the outcome when the button is clicked
if st.button("🚀 Predict"):
    try:
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.success("🌧️ Prediction: **Rainfall** is likely! 🌈")
        else:
            st.success("☀️ Prediction: **No Rainfall** expected! Enjoy the clear skies! 😎")
    except Exception as e:
        st.error(f"Error: {e}")

# Display additional model details
st.subheader("📖 About the Model")
st.write("""
This app uses a pre-trained machine learning model to predict rainfall based on weather parameters like pressure, humidity, cloud coverage, and more.
Enter the values above to get an accurate prediction. 🌐
""")
