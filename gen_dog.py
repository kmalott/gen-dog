import streamlit as st
import numpy as np
import torch
import os

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = torch.load("BSQTokenizer.th", weights_only=False, map_location=torch.device(device)).to(device)
    ar_model = torch.load("AutoRegressive.th", weights_only=False, map_location=torch.device(device)).to(device)
    with st.spinner("Generating image...", show_time=False):
        generations = ar_model.generate(1, 32, 32, device=device)
    images = tk_model.decode(tk_model.decode_int(generations)).cpu().transpose(1,3)
    np_img = (255 * (images).clip(0, 1)).to(torch.uint8).numpy()
    return np_img

st.set_page_config(page_title="Dog Generator")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("This Dog Does Not Exist")
    st.text("Click the Generate button for a new dog!")
    # img = generate()
    img = None
    if img is None:
        st.image("dog.jpg", width=256)
    else:
        st.image(img, width=256)
    gen = st.button("Generate")

if gen:
    st.rerun()