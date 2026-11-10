
import streamlit as st

from speech_to_text import convert_speech
from assistant import get_ai_response
from text_to_speech import convert_text_to_speech

from database import (
    create_database,
    save_message,
    get_messages,
    clear_messages
)

from document_assistant import (
    extract_pdf_text,
    split_documents,
    create_vector_database,
    search_documents,
    save_vector_database,
    load_vector_database
)


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Nova AI 2.1",
    page_icon="🎙️",
    layout="wide"
)

create_database()


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #17243a);
    color: white;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #8ab4ff;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #c0cce0;
    font-size: 16px;
    margin-bottom: 25px;
}

[data-testid="stSidebar"] {
    background-color: #111c2e;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = get_messages()

if "audio_response" not in st.session_state:
    st.session_state.audio_response = None

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True

if "document_index" not in st.session_state:
    index, chunks = load_vector_database()

    st.session_state.document_index = index
    st.session_state.document_chunks = chunks

    st.session_state.document_names = list(
        dict.fromkeys(
            chunk["source"] for chunk in chunks
        )
    )


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("🎙️ Nova AI 2.1")
    st.caption("Intelligent Voice & Document Assistant")

    st.divider()

    st.session_state.voice_enabled = st.toggle(
        "Enable voice responses",
        value=st.session_state.voice_enabled
    )

    st.divider()

    st.subheader("💬 Conversation")

    st.write(
        f"Saved messages: {len(st.session_state.history)}"
    )

    if st.button("Clear conversation"):

        clear_messages()

        st.session_state.history = []
        st.session_state.audio_response = None
        st.session_state.transcribed_text = ""

        st.rerun()

    st.divider()

    # ---------------- DOCUMENT UPLOAD ----------------

    st.subheader("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if st.button("Process new document"):

        if uploaded_file is None:

            st.warning("Please upload a PDF first.")

        else:

            try:

                with st.spinner("Processing new PDF..."):

                    documents = extract_pdf_text([uploaded_file])

                    chunks = split_documents(documents)

                    index = create_vector_database(chunks)

                    # Replace the previous saved database
                    save_vector_database(index, chunks)

                    # Replace previous session data
                    st.session_state.document_index = index
                    st.session_state.document_chunks = chunks
                    st.session_state.document_names = [
                        uploaded_file.name
                    ]

                st.success(
                    f"Successfully processed {uploaded_file.name}"
                )

            except Exception as e:

                st.error(f"Document processing failed: {e}")

    if st.session_state.document_names:

        st.success("Documents ready for questions")

        for name in st.session_state.document_names:
            st.caption("📄 " + name)

    else:

        st.info("No processed documents yet.")


# ---------------- MAIN HEADER ----------------

st.markdown(
    '<div class="main-title">✨ Nova AI 2.1</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Voice Intelligence • Document Understanding • AI Memory'
    '</div>',
    unsafe_allow_html=True
)


# ---------------- CHAT HISTORY ----------------

for message in st.session_state.history:

    role = (
        "user" if message["role"] == "user"
        else "assistant"
    )

    with st.chat_message(role):
        st.write(message["content"])


# ---------------- QUESTION INPUT ----------------

st.divider()

st.subheader("💬 Ask Nova AI")

document_mode = st.toggle(
    "Answer using uploaded documents only",
    value=False
)

question = st.text_input(
    "Type your question",
    key="typed_question",
    placeholder="Ask Nova anything..."
)


# ---------------- VOICE INPUT ----------------

st.subheader("🎤 Or speak to Nova")

audio = st.audio_input("Record your question")

if audio is not None:

    if st.button("Convert speech to text"):

        try:

            with st.spinner("Recognizing speech..."):

                text = convert_speech(audio.getvalue())

            st.session_state.transcribed_text = text

            if text:
                st.success("Speech recognized!")
            else:
                st.warning("No speech detected.")

        except Exception as e:

            st.error(f"Speech recognition error: {e}")


if st.session_state.transcribed_text:

    question = st.text_area(
        "Review your spoken question",
        value=st.session_state.transcribed_text,
        key="voice_question"
    )


# ---------------- ASK NOVA ----------------

if st.button("🚀 Ask Nova AI", use_container_width=True):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            document_context = None

            if document_mode:

                if st.session_state.document_index is None:

                    st.warning("Upload and process a PDF first.")
                    st.stop()

                with st.spinner("Searching the current document..."):

                    results = search_documents(
                        question,
                        st.session_state.document_index,
                        st.session_state.document_chunks
                    )

                if not results:

                    st.warning(
                        "No relevant information found in the document."
                    )
                    st.stop()

                document_context = "\n\n".join(
                    [
                        (
                            f"Source: {item['source']}, "
                            f"Page: {item['page']}\n"
                            f"{item['text']}"
                        )
                        for item in results
                    ]
                )

            with st.spinner("Nova is thinking..."):

                # Do not pass old conversation into document-only mode
                answer = get_ai_response(
                    question,
                    history=[] if document_mode else st.session_state.history,
                    document_context=document_context
                )

            save_message("user", question)
            save_message("model", answer)

            st.session_state.history = get_messages()

            st.session_state.transcribed_text = ""
            st.session_state.audio_response = None

            if st.session_state.voice_enabled:

                try:

                    with st.spinner("Generating voice..."):

                        st.session_state.audio_response = (
                            convert_text_to_speech(answer[:1500])
                        )

                except Exception as e:

                    st.warning(f"Voice generation failed: {e}")

            st.rerun()

        except Exception as e:

            st.error(f"Nova AI error: {e}")


# ---------------- AUDIO OUTPUT ----------------

if st.session_state.audio_response:

    st.subheader("🔊 Listen to Nova")

    st.audio(
        st.session_state.audio_response,
        format="audio/wav"
    )


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center><small>"
    "Nova AI 2.1 • Intelligent Voice & Document Assistant"
    "</small></center>",
    unsafe_allow_html=True
)
