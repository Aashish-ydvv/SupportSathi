import json
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://127.0.0.1:8000"

MODEL_NAMES = {
    "baseline": "TF-IDF baseline",
    "lora": "MiniLM + LoRA",
}

st.set_page_config(
    page_title="SupportSathi",
    page_icon="💬",
    layout="centered",
)

st.title("SupportSathi")
st.write(
    "Classify online shopping support messages "
    "in English, Hindi, or Hinglish."
)
st.warning(
    "Development prototype trained on 100 AI-generated examples. "
    "Predictions may be incorrect. Avoid entering personal information."
)

classify_tab, results_tab, labels_tab = st.tabs([
    "Classify a message",
    "Validation results",
    "Category guide",
])

with classify_tab:
    with st.form("prediction_form"):
        selected_model = st.selectbox(
            "Model",
            options=list(MODEL_NAMES),
            format_func=lambda value: MODEL_NAMES[value],
        )
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
                        json={
                            "text": message,
                            "model": selected_model,
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()

                st.subheader(
                    result["intent"].replace("_", " ").title()
                )
                st.caption(f"Model: {result['model_version']}")
                st.caption(
                    f"API processing time: {result['latency_ms']} ms"
                )
                st.caption(
                    "This time excludes network transfer and UI rendering."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the API. Start the FastAPI server first."
                )
            except requests.exceptions.Timeout:
                st.error("The API timed out. Please try again.")
            except requests.exceptions.HTTPError as error:
                try:
                    detail = error.response.json().get(
                        "detail", "Request failed."
                    )
                except ValueError:
                    detail = "Request failed."
                st.error(
                    f"API error {error.response.status_code}: {detail}"
                )
            except (ValueError, KeyError):
                st.error("The API returned an unexpected response.")
            except requests.exceptions.RequestException:
                st.error("The request failed. Please try again.")

with results_tab:
    st.subheader("Development validation comparison")
    report_path = ROOT / "reports/validation_v2.json"

    if report_path.exists():
        try:
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )

            rows = []
            for key, name in [
                ("tfidf_baseline_v2", "TF-IDF baseline"),
                ("lora_v2", "MiniLM + LoRA"),
            ]:
                metrics = report[key]
                rows.append({
                    "Model": name,
                    "Accuracy": f"{metrics['accuracy']:.2%}",
                    "Macro-F1": f"{metrics['macro_f1']:.4f}",
                })

            st.table(rows)
            st.caption(
                f"Training examples: {report['training_examples']} · "
                f"Validation examples: {report['validation_examples']}"
            )
            st.write(
                "Macro-F1 gives each category equal weight. "
                "Higher values indicate better performance on this dataset."
            )
            st.info(
                "These are results on a small, AI-generated development "
                "validation set used during model development. "
                "They are not final-test or real-world accuracy estimates."
            )
        except (ValueError, KeyError):
            st.error(
                "The validation report is invalid. "
                "Run scripts/evaluate_validation.py again."
            )
    else:
        st.info(
            "Run scripts/evaluate_validation.py to generate the report."
        )

with labels_tab:
    config_path = ROOT / "configs/labels.json"
    try:
        config = json.loads(
            config_path.read_text(encoding="utf-8-sig")
        )
        for label, description in config["labels"].items():
            st.markdown(
                f"**{label.replace('_', ' ').title()}** — {description}"
            )
        st.caption(
            "An explicit requested action takes priority "
            "over the problem description."
        )
    except (OSError, ValueError, KeyError):
        st.error("Could not read the category definitions.")