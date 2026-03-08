# 🌱 Tomato Leaf Disease Classifier

Welcome to the Tomato Leaf Disease Classifier project! This application leverages deep learning to identify various diseases from tomato leaf images. It also integrates an LLM (Qwen-32B via Groq) to provide actionable, translated treatment recommendations depending on the diagnosed disease.

## 🚀 Features
* **Disease Classification**: Utilizes a fine-tuned EfficientNet-B0 model to classify images into 11 distinct classes (10 diseases + 1 Healthy).
* **AI Recommendations**: Employs Qwen-32B via the Groq API to provide treatment recommendations for detected diseases.
* **Multilingual Support**: LLM-generated recommendations can be output in English, German, or Hungarian.
* **Modern Interface**: Built with an interactive Streamlit frontend.
* **Containerized**: Fully containerized using Docker for seamless deployment.

## 📊 Project Overview & Methodology
This project includes an extensive machine learning pipeline modeled within `disease_detection_notebook.ipynb`. 
1. **Dataset**: Uses the `cookiefinder/tomato-disease-multiple-sources` dataset from Kaggle.
2. **Data Preprocessing**: Images are resized to 256x256, normalized, and heavily augmented.
3. **Imbalance Handling**: Class weights are calculated to adjust the CrossEntropyLoss, tackling dataset imbalances.
4. **Model Architecture & Training**: 
   * A Baseline `CustomCnnModel` was designed.
   * Transfer learning was applied using a pretrained `EfficientNet-B0` model.
   * Fine-tuning was tracked in **MLflow** and logged remotely via **DagsHub**.
   * Hyperparameter optimization was conducted with **Optuna**.
5. **Evaluation**: Models were evaluated using accuracy, F1-macro, and F1-weighted scores, utilizing Early Stopping to prevent overfitting.

---

## 🛠️ Environment Setup

This project uses Python 3.11 and the [`uv`](https://github.com/astral-sh/uv) package manager for extremely fast dependency resolutions, as defined in `pyproject.toml`.

### Local Setup (Using uv)
1. Install `uv` on your system if you haven't already:
   ```bash
   pip install uv
   ```
2. Clone this repository and navigate to the project root.
3. Sync the dependencies and create the virtual environment:
   ```bash
   uv sync --frozen
   ```
4. Activate the virtual environment:
   * **Windows**: `.venv\Scripts\activate`
   * **Linux/Mac**: `source .venv/bin/activate`

*Note: You can also use standard `pip` by running `pip install -r requirements.txt`, though `uv` is highly recommended.*

---

## 🏃 Running the Application

### Option 1: Running Locally with Streamlit
Once your environment is set up and activated:
1. Ensure you have your `efficientnet_b0_ff.pth` model saved in the `models/` directory.
2. Launch the Streamlit application:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open the provided Local URL (usually `http://localhost:8501`) in your browser.
4. *(Optional)* To get AI-powered treatment advice, ensure your `.env` file contains `GROQ_API_KEY=your_key`.

### Option 2: Running with Docker
The repository includes a ready-to-use Dockerfile. It uses `python:3.11-slim` and natively integrates `uv` for lightning-fast container builds.

1. **Build the Docker Image:**
   Make sure you are in the project root directory where the `Dockerfile` is located.
   ```bash
   docker build -t tomato_leaf_disease_classifier .
   ```

2. **Run the Docker Container:**
   Map the default container port (8501) to your local machine port (8501):
   ```bash
   docker run -p 8501:8501 --name tomato_app tomato_leaf_disease_classifier
   ```

3. **Access the App:**
   Open your browser and navigate to `http://localhost:8501`.

*Note*: If you need to rebuild or restart the container, simply use standard Docker CLI commands like `docker stop tomato_app` and `docker rm tomato_app`.

---

## ☁️ Deploying to Streamlit Cloud
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New App**.
3. Select this repository and `streamlit_app.py` as your main file.
4. **Important**: Before hitting deploy, click on **Advanced Settings** and add your Groq API key into the **Secrets** section like so:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy!**

---

## 📁 Repository Structure
* `disease_detection_notebook.ipynb`: Core notebook covering ETL, PyTorch dataset creation, Custom CNNs, EfficientNet fine-tuning, MLflow tracking, and Optuna tuning.
* `streamlit_app.py`: The main Streamlit web application script.
* `src/predict.py`: Core logic for loading the model weights and running inference.
* `Dockerfile`: Containerization instructions.
* `pyproject.toml` / `uv.lock`: Project metadata and exact dependency lockdown. 
