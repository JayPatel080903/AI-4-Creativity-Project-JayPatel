# AI-4-Creativity-Project-Template (25/26)

**Student name:** Jay Patel  
**Student number:** 2312665
**Project title:** Fake News Detection System  
**Link to project video recording:** https://drive.google.com/file/d/1W9rxz54U8uunG0G609TP9Ne0eL0B_rgO/view?usp=drivesdk

## Setup Instructions

Instructions for setting up the conda environment, any files that need downloading, and the specific technical instructions for how to run your code project:

```bash
# Step 1: Clone or download the project
git clone <repository-url>
cd AI-4-Creativity-Project-JayPatel

# Step 2: Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or source venv/bin/activate  # On macOS/Linux

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the application
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

# Fake News Detection System

A machine learning-powered web application that classifies news articles as **FAKE** or **REAL** using a fine-tuned DistilBERT transformer model.

# Dataset Source

https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

## Features

- 📰 Real-time fake news detection
- 🤖 Fine-tuned DistilBERT model trained on the ISOT Fake News Dataset
- 📊 Confidence score and probability breakdown
- 🎯 Fast inference with optimized tokenization
- 💻 User-friendly web interface built with Streamlit

## Model Details

- **Model Architecture**: DistilBERT (Sequence Classification)
- **Training Dataset**: ISOT Fake News Dataset
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Input**: News article text (up to 256 tokens)
- **Output**: Classification (FAKE/REAL) with confidence percentage

## Ethical Considerations

This system may misclassify legitimate journalism as fake. 
Bias in the training dataset may influence predictions.
The system should not be used as a sole decision-making authority.

## Project Structure

```
AI-4-Creativity-Project-JayPatel/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── fake_news_model/            # Pre-trained model directory
    ├── config.json             # Model configuration
    ├── model.safetensors       # Model weights
    ├── tokenizer_config.json   # Tokenizer configuration
    ├── vocab.txt               # Vocabulary file
    └── special_tokens_map.json # Special tokens mapping
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project

```bash
# If cloning from a repository
git clone <repository-url>
cd AI-4-Creativity-Project-JayPatel

# Or navigate to the project directory if already downloaded
cd AI-4-Creativity-Project-JayPatel
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **streamlit**: Web framework for the UI
- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers library
- **safetensors**: For safe model serialization

## Running the Application

### Using Streamlit

```bash
streamlit run app.py
```

The application will start and open in your default web browser at `http://localhost:8501`

### What to Expect

1. A web interface with the title "📰 Fake News Detection System"
2. A text area where you can paste news article content
3. A "Predict" button to classify the article
4. Results showing:
   - Classification (FAKE or REAL)
   - Confidence level (as a percentage)
   - Probability breakdown for both classes

## Usage

1. **Launch the application** using the command above
2. **Enter a news article** in the text area
3. **Click "Predict"** to get the classification
4. **View results**:
   - Green checkmark (✅) = Real article
   - Red alert (🚨) = Fake article
   - Confidence percentage and probability breakdown

## Technical Details

### Model Loading

The model is cached in memory using Streamlit's `@st.cache_resource` decorator for optimal performance on subsequent predictions.

### Input Processing

- **Max Length**: 256 tokens
- **Padding**: Applied to shorter sequences
- **Truncation**: Applied to longer sequences
- **Device**: CPU (supports GPU with modifications)

### Predictions

The model outputs probabilities for each class. The prediction is made based on which probability is higher:
- Class 0: FAKE
- Class 1: REAL

## Troubleshooting

### Error: "Model folder not found"

**Solution**: Ensure the `fake_news_model` directory exists in the project root with all required model files:
- `config.json`
- `model.safetensors`
- `tokenizer_config.json`
- `vocab.txt`
- `special_tokens_map.json`

### Error: Module import errors

**Solution**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port already in use

**Solution**: Run Streamlit on a different port:
```bash
streamlit run app.py --server.port 8502
```

### Slow predictions

**Solution**: Ensure you're using a system with adequate resources. For GPU acceleration, modify the code to use CUDA instead of CPU.

## Performance Notes

- First prediction may take a few seconds as the model is loaded and cached
- Subsequent predictions are faster due to caching
- Works best with articles of moderate length (recommended: 100-2000 words)

## Future Enhancements

- GPU support for faster inference
- Batch processing for multiple articles
- Confidence threshold customization
- Model fine-tuning UI
- Article source analysis
- Multi-language support

---

**Version**: 1.0  
**Last Updated**: January 2026
