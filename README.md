# Image-Captioning-Application

A Streamlit-based web application that automatically generates descriptive captions for uploaded images using the ViT-GPT2 image captioning model from Hugging Face.
This project bridges Computer Vision and Natural Language Processing (NLP) to produce human-like captions for images.



### Features

1. Upload images in JPG, JPEG, PNG, or WEBP format
2. Generates multiple caption variations per image
3. GPU acceleration (if available)
4. Secure Hugging Face authentication using environment variables (.env)
5. Built with Streamlit, Transformers, and PyTorch



### Tech Stack

1. Frontend/UI: Streamlit
2. Model: ViT-GPT2 (Vision Transformer + GPT-2)
3. Libraries: Transformers, PyTorch, Pillow, python-dotenv



### Setup

1. Clone the repository

```
git clone https://github.com/your-username/Image-Captioning-Application.git
cd Image-Captioning-Application
```

2. Create a virtual environment and activate it

```
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate   # On macOS/Linux
```

3. Install dependencies

```
pip install -r requirements.txt
```



### Setup Using `uv` package manager

1. `uv` installed (there are other ways to do it too check - [**uv installation**](https://docs.astral.sh/uv/getting-started/installation/))

```
pip install uv
```

2. Clone the repository

```
git clone https://github.com/DataAnalyticClub-NIST/<repo-name>.git
cd Image-Captioning-Application
```

3. Create a virtual environment and activate it

```
uv venv
```

4. Install dependencies

```
uv pip install -r requirements.txt
```

5. Activate virtual environment 

```
.venv\Scripts\activate      # On Windows
source .venv/bin/activate   # On macOS/Linux
```



### Run the App

1. Set up Hugging Face Token

- Create an account on Hugging Face
- Generate a Read access token from Settings → Access Tokens
- Create a .env file in the project root and add:

```
HF_ACCESS_TOKEN=your_huggingface_token_here
```

2. Run app.py
```
streamlit run app.py
```

After running, Streamlit will provide a local URL (like http://localhost:8501
).
Open it in your browser to use the application.



### Contribution Guidelines

If you want to add or modify something:
- Create an issue first describing what you plan to do.
- Fork the repo or create a new branch.
- Submit a pull request (PR) once your work is ready.
