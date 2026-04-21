# Image-to-Story Speech Generator

A Streamlit app that takes an uploaded image and produces a voice-narrated short story using a three-stage GenAI pipeline.

## System Design

![system-design](img/system-design.drawio.png)

## Pipeline

1. **Image → Caption**: `Salesforce/blip-image-captioning-base` (HuggingFace `transformers`, runs locally on CPU) produces a scene description from the uploaded JPG.
2. **Caption → Story**: `google/flan-t5-large` (HuggingFace `transformers`, runs locally on CPU) generates a ~50-word short story conditioned on the caption and a user-selected genre.
3. **Story → Speech**: `espnet/kan-bayashi_ljspeech_vits` (HuggingFace Inference API) converts the story to FLAC audio, played back in the browser.

## Setup

**1. Clone and install dependencies**

```bash
pip install -r requirements.txt
```

**2. Create a `.env` file in the project root**

```
HUGGINGFACE_API_TOKEN=<your-huggingface-token>
```

**3. Run the app**

```bash
streamlit run app.py
```

## Demo

![Demo: Couple Test Image Output](img-audio/CoupleOutput.jpg)

Audio samples for test images are in the `img-audio/` folder.

## Requirements

```
streamlit
transformers
torch
sentencepiece
accelerate
Pillow
python-dotenv
requests
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
