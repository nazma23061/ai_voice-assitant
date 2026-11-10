
from faster_whisper import WhisperModel
import tempfile
import os

# Load the speech recognition model
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

def convert_speech(audio_bytes):

    # Save recorded audio temporarily
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as audio_file:

        audio_file.write(audio_bytes)
        audio_path = audio_file.name

    try:
        # Convert speech into text
        segments, info = model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(
            segment.text for segment in segments
        )

        result = text.strip()

        print("WHISPER TRANSCRIPTION:", repr(result))

        return result

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
