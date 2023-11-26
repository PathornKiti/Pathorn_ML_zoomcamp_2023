import streamlit as st
import requests


def main():

    st.title("Intent Classification and Slots Filling")
    message = st.text_input('Enter Text')

    if st.button('Predict'):
        payload = {
            "text": message
        }
        res = requests.post(f"http://service:8000/predict/",json=payload )
        with st.spinner('Please wait....'):
            st.write(res.json())




if __name__ == '__main__':
    main()