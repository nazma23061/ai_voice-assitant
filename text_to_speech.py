
import subprocess
import tempfile
import os
import re

def convert_text_to_speech(text):

    # Remove Markdown formatting
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Remove headings, bullets and Markdown symbols
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-•]\s*", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s*", "", text)

    # Remove remaining unwanted symbols
    text = re.sub(r"[#*_~>|]", "", text)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    with tempfile.TemporaryDirectory() as temp_dir:

        output_file = os.path.join(temp_dir, "response.wav")

        subprocess.run(
            [
                "piper",
                "--model", "en_US-lessac-medium",
                "--output_file", output_file
            ],
            input=text,
            text=True,
            check=True,
            capture_output=True
        )

        with open(output_file, "rb") as audio_file:
            audio_bytes = audio_file.read()

    return audio_bytes
