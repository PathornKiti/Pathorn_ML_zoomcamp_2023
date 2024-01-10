import streamlit as st
import requests
from PIL import Image
from io import BytesIO

def get_prediction(image_url):
    #local
    #api_url = "http://backend:8000/predict"
    #Cloud
    api_url = "https://backend-np5jrb4y6q-uc.a.run.app/predict"
    payload = {"img_url": image_url}
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    st.title("Alzheimer MRI Image Prediction Web App")
    st.write("Enter the URL of the brain MRI image you want to predict:")
    
    image_url = st.text_input("Image URL")
    if st.button("Predict"):
        if image_url:
            prediction_result = get_prediction(image_url)
            if prediction_result:
                st.write("Prediction:", prediction_result)
                try:
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        st.image(image, caption='MRI Image', use_column_width=True)
                    else:
                        st.write("Failed to fetch the image from the provided URL.")
                except Exception as e:
                    st.write(f"An error occurred: {e}")
            else:
                st.write("Failed to get a prediction. Please check the URL and try again.")
        else:
            st.write("Please enter an image URL.")

if __name__ == "__main__":
    main()
