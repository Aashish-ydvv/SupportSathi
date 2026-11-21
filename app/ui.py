import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="SupportSathi", page_icon="💬")

st.title("SupportSathi")
st.write("Classify online shopping support messages in English, Hindi, or Hinglish.")
st.info(
    "Prototype trained on 30 AI-generated examples. "
    "Accuracy has not been evaluated. Predictions may be incorrect."
)

with st.form("prediction_form"):
    message = st.text_area(
        "Customer message",
        placeholder="Mera parcel kab aayega?",
        max_chars=1000,
        height=130,
    )
    submitted = st.form_submit_button("Classify")

if submitted:
    if not message.strip():
        st.warning("Enter a support message.")
    else:
        try:
            with st.spinner("Classifying..."):
                response = requests.post(
                    f"{API_URL}/v1/predict",
                    json={"text": message},
                    timeout=15,
                )
                response.raise_for_status()
                result = response.json()

            st.subheader("Predicted category")
            st.write(result["intent"].replace("_", " ").title())
            st.caption(
                f"Model: {result['model_version']} · "
                f"Inference time: {result['latency_ms']} ms"
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Start the FastAPI server first.")
        except requests.exceptions.Timeout:
            st.error("The API took too long to respond. Please try again.")
        except requests.exceptions.HTTPError as error:
            st.error(f"The API returned an error: {error.response.status_code}")
        except (ValueError, KeyError):
            st.error("The API returned an unexpected response.")
        except requests.exceptions.RequestException:
            st.error("The request failed. Please try again.")