import streamlit as st
import numpy as np
from PIL import Image

from instafilter import Instafilter

st.set_option("deprecation.showfileUploaderEncoding", False)

st.beta_set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

url = "https://github.com/thoppe/instafilter"
st.markdown("# [Instafilter]({url}) demo")

model_name = st.sidebar.selectbox(
    "Choose a filter",
    sorted(Instafilter.get_models()),
    index=20,
)
model = Instafilter(model_name)

raw_image_bytes = st.file_uploader("Choose an image...")

if raw_image_bytes is not None:

    img0 = np.array(Image.open(raw_image_bytes))

    with st.spinner(text="Applying filter..."):
        # Apply the model, convert to BGR first and after
        img1 = model(img0[:, :, ::-1], is_RGB=False)[:, :, ::-1]

    st.image(
        [img1, img0], width=550, caption=[f"{model_name} filter", "Original"]
    )
