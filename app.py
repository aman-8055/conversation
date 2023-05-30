import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    st.title("DialoGPT Chatbot")

    # Let's chat for 5 lines
    for step in range(5):
        user_input = st.text_input("User:")
        if not user_input:
            break

        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        st.text("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

if __name__ == "__main__":
    main()
