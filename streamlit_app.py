
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
import sys

# Import Vocabulary at module level for pickle compatibility
from model import ImageCaptioningModel, Vocabulary

# Make Vocabulary available in __main__ namespace for pickle
sys.modules['__main__'].Vocabulary = Vocabulary

# Set page config
st.set_page_config(
    page_title="Neural Storyteller - Image Captioning",
    page_icon="📸",
    layout="wide"
)

# Title
st.title("📸 Neural Storyteller - Image Captioning")
st.markdown("Upload an image and let AI describe it for you!")

# Load model and vocabulary (cached)
@st.cache_resource
def load_model_and_vocab():
    # Load vocabulary
    with open('vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Initialize model with notebook architecture params
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_size=384,
        hidden_size=768,
        num_layers=1
    )
    
    # Load weights
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab

# Feature extractor
@st.cache_resource
def load_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

# Extract features
def extract_features(image, feature_extractor):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        features = features.view(1, -1)
    return features.squeeze(0)

# Load models
try:
    caption_model, vocab = load_model_and_vocab()
    feature_extractor = load_feature_extractor()
    st.success("✓ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Generated Caption")
        
        # Settings
        max_length = st.slider("Max Caption Length", 10, 30, 20)
        
        if st.button("Generate Caption", type="primary"):
            with st.spinner("Generating caption..."):
                # Extract features
                features = extract_features(image, feature_extractor)
                
                # Generate caption (greedy decoding)
                caption = caption_model.generate_caption(
                    features, vocab, max_length=max_length
                )
                
                # Display caption
                st.success("Caption Generated!")
                st.markdown(f"### 📝 {caption.capitalize()}")

# Sidebar info
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app uses a Seq2Seq model with LSTM decoder to generate "
    "natural language descriptions for images. The model was trained "
    "on the Flickr30k dataset."
)

st.sidebar.title("🎯 Model Details")
st.sidebar.markdown("""
- **Encoder**: ResNet50 + Linear projection (768-dim)
- **Decoder**: LSTM with 384-dim word embeddings
- **Hidden Size**: 768
- **Vocabulary**: ~7689 words
- **Dataset**: Flickr30k (31783 images)
""")
