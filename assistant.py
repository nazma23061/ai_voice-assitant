
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)


def get_ai_response(question, history=None, document_context=None):

    contents = []

    # Add previous conversation
    if history:
        for message in history:

            role = message["role"]

            if role == "assistant":
                role = "model"

            if role not in ["user", "model"]:
                continue

            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=message["content"]
                        )
                    ]
                )
            )

    # Add document context when available
    if document_context:

        question = f"""
Answer the user's question using the following
retrieved document information.

DOCUMENT INFORMATION:
{document_context}

USER QUESTION:
{question}

Instructions:
- Give answers based on the supplied document information.
- Do not invent facts that are not supported by the documents.
- Mention the source PDF and page number.
- If the information is insufficient, clearly say so.
"""

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question)
            ]
        )
    )

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are Nova AI, an intelligent personal assistant. "
                "Be accurate, helpful and conversational. "
                "When document context is supplied, prioritize it "
                "and never invent document citations."
            )
        )
    )

    if not response.text:
        return "Sorry, I could not generate an answer."

    return response.text
