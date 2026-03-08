import os
from dotenv import load_dotenv
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from src.predict import ImageClassifier
from PIL import Image
from langchain.chat_models import init_chat_model

st.set_page_config(page_title="Tomato leaf disease classifier", layout="centered")
st.title("🌱 Tomato leaf disease classifier")
st.write("Upload an image below and click **Submit** to see the classification results.")
st.write("AI-powered treatment recommendations are provided automatically.")

LANGUAGES = {
    "English": "English",
    "German": "Deutsch",
    "Hungarian": "Magyar"
}

with st.sidebar:
    st.title("Settings")
    selected_lang = st.selectbox("Response Language", options=list(LANGUAGES.keys()), index=0)
    
    # Save language to session state directly upon selection
    st.session_state.language = LANGUAGES[selected_lang]
    st.success(f"Language set to {selected_lang}.")
        
# Retrieve API key securely from Streamlit Secrets (Cloud) or local .env
try:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
except FileNotFoundError:
    # This happens when running locally without a .streamlit/secrets.toml file.
    api_key = os.environ.get("GROQ_API_KEY")
        

@st.cache_resource
def load_classifier():
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "models", "efficientnet_b0_ff.pth")
    return ImageClassifier(model_path=model_path)

classifier = load_classifier()

def classify_image(image):
    image_path = "uploaded_image.png"
    image.save(image_path)
    label, output_path = classifier.predict(image_path)
    return label, Image.open(output_path)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

submit_button = st.button(
    "Analyze Image", 
    type="primary", 
    use_container_width=True,
    disabled=(uploaded_file is None) 
)

if submit_button:
    with st.spinner('Analyzing...'):
        label, result_plot_path = classifier.predict(uploaded_file)
        
        # Display the Classification Result
        st.success(f"### Diagnosis: {label.replace('_', ' ')}")
        st.image(Image.open(result_plot_path), use_container_width=True)

        # --- CHATBOT RECOMMENDATION LOGIC ---
        # Only proceed if the plant is NOT healthy and API key exists
        if label.lower() != "healthy":
            if api_key:
                with st.chat_message("assistant"):
                    st.write("✨ **AI Recommendation:**")
                    target_lang = st.session_state.get("language", "English")
                    with st.spinner("Consulting Qwen-32B..."):
                        try:
                            # Initialize model using init_chat_model
                            llm = init_chat_model(
                                "qwen/qwen3-32b", 
                                model_provider="groq", 
                                api_key=api_key,
                                temperature=0.6,
                                reasoning_effort= "none"
                            )
                            
                            prompt = f"""
                            The plant image has been diagnosed with: {label}.
                            Provide a brief, professional recommendation for a farmer on how to treat this specific problem.
                            Keep the advice actionable and concise.
                            
                            IMPORTANT: You must provide the entire response in the {target_lang} language.
                            """
                            
                            response = llm.invoke(prompt)
                            st.markdown(response.content)
                            
                        except Exception as e:
                            st.error(f"Chatbot error: {e}")
            else:
                st.warning("No Groq API Key found. Unable to fetch AI recommendations. Please configure it in Streamlit Secrets.")
        else:
            healthy_msgs = {"English": "The plant is healthy!", "Deutsch": "Die Pflanze ist gesund!", "Magyar": "A növény egészséges!"}
            current_lang = st.session_state.get("language", "English")
            st.info(healthy_msgs.get(current_lang, healthy_msgs["English"]))