# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Environment Setup

Create a `.env` file in the root with:
```
OPENAI_API_KEY=<your-openai-key>
HUGGINGFACE_API_TOKEN=<your-huggingface-token>
```

## Architecture

This is a single-file Streamlit app (`app.py`) implementing a three-stage GenAI pipeline:

1. **Image → Text**: Uploaded JPG is passed to `Salesforce/blip-image-captioning-base` (loaded locally via HuggingFace `transformers`) to produce a scene description.
2. **Text → Story**: The description is fed into a LangChain `LLMChain` using `gpt-3.5-turbo` (OpenAI) with a prompt template that produces a ~50-word short story.
3. **Story → Speech**: The story is sent to the `espnet/kan-bayashi_ljspeech_vits` model via the HuggingFace Inference API (REST), returning FLAC audio that is played in the browser.

Key implementation details:
- BLIP model runs locally via `transformers.pipeline("image-to-text", ...)` — requires torch and sufficient RAM.
- HuggingFace TTS is called via `requests.post` to `https://api-inference.huggingface.co/models/...`, not locally.
- `utils/custom.py` only contains CSS injected via `st.markdown` for sidebar/main padding adjustments.
- Audio output is written to `audio.flac` in the working directory on each run.
