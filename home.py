import streamlit as st
from PIL import Image

st.markdown('<h3 style="text-align: center;">工 程 机 械 动 力 匹 配 系 统</h3>', unsafe_allow_html=True)
image = Image.open('image/loader.jpg')
st.image(image, caption='')

st.markdown('<p style="text-align: right;">Copyright © 2023, ZQ.</p>', unsafe_allow_html=True)
