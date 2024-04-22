import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

@st.cache_data
def load_model(model_path):
    """
    Load the fine-tuned model and tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer

def translate_sentence(sentence, model, tokenizer):
    """
    Translate a given sentence from English to German.
    """
    if not sentence.strip():  # If the input is empty or just whitespace
        return ""

    # Create a translation pipeline
    translator = pipeline("translation_en_to_de", model=model, tokenizer=tokenizer, device=torch.cuda.current_device() if torch.cuda.is_available() else -1)
    # Translate the sentence
    translation = translator(sentence)
    return translation[0]['translation_text']

def main():
    st.title("English to German Translation. (ML for NLP 2 demo)")
    model_path = "./en_de_model"

    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)

    sentence = st.text_area("Enter a sentence in English to translate to German:", height=100)
    translate_button = st.button("Translate")

    if translate_button:
        translated_sentence = translate_sentence(sentence, model, tokenizer)
        if translated_sentence:
            st.write(f"Translated Sentence: {translated_sentence}")

if __name__ == "__main__":
    main()
