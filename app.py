import os
import time
from io import BytesIO

import requests
import streamlit as st
import torch
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from utils.custom import css_code

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

TTS_API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"


@st.cache_resource
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model


@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    model.eval()
    return tokenizer, model


def progress_bar(amount_of_time: int) -> None:
    progress_text = "Please wait, Generative models hard at work"
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(amount_of_time):
        time.sleep(0.04)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()


def generate_caption(image: Image.Image) -> str:
    processor, model = load_captioning_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"GENERATED CAPTION: {caption}")
    return caption


def generate_story(scenario: str, genre: str) -> str:
    tokenizer, model = load_story_model()
    prompt = (
        f"You are a creative storyteller.\n\n"
        f"Scene: {scenario}\n\n"
        f"Write a vivid {genre} story in 40-60 words describing the scene. "
        f"Add emotions, imagination, and action."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
        )
    story = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"SCENE INPUT: {scenario}")
    print(f"GENERATED STORY: {story}")
    return story


def generate_speech(message: str) -> bytes | None:
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "X-Wait-For-Model": "true",
    }
    payload = {"inputs": message}

    for attempt in range(3):
        response = requests.post(TTS_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            return response.content

        # Model still loading — HF returns 503 with an estimated_time field
        if response.status_code == 503:
            try:
                wait = response.json().get("estimated_time", 20)
            except Exception:
                wait = 20
            st.info(f"TTS model is loading, retrying in {wait:.0f}s… (attempt {attempt + 1}/3)")
            time.sleep(min(float(wait), 30))
            continue

        # Any other error — extract a human-readable message
        try:
            detail = response.json().get("error", response.text[:300])
        except Exception:
            detail = response.text[:300]
        st.error(f"TTS API error ({response.status_code}): {detail}")
        return None

    st.error("TTS model did not become ready after 3 attempts. Try again in a minute.")
    return None


def main() -> None:
    st.set_page_config(page_title="Image-to-Story Speech Generator", page_icon="🖼️")
    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.write("**Image-to-Story Speech Generator**")
        st.write("Upload an image to generate a short story and hear it narrated.")

    st.header("Image-to-Story Speech Generator")

    load_captioning_model()
    load_story_model()

    genre: str = st.selectbox("Select a story genre", ["Adventure", "Horror", "Funny", "Sci-Fi"])

    uploaded_file = st.file_uploader("Please choose a file to upload", type="jpg")

    if uploaded_file is not None:
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        progress_bar(100)

        scenario = generate_caption(image)
        story = generate_story(scenario, genre)
        audio_bytes = generate_speech(story)

        with st.expander("Generated image caption", expanded=True):
            st.write(scenario)
        with st.expander("Generated short story", expanded=True):
            st.write(story)

        if audio_bytes:
            st.audio(audio_bytes, format="audio/flac")


if __name__ == "__main__":
    main()
