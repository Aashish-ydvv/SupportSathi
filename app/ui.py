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
st.write("A retail support intent classifier for Roman-script Hinglish.")
st.caption("Compare a traditional text classifier with a LoRA-adapted model.")

st.warning(
    "Research prototype trained on public synthetic data. "
    "Predictions may be incorrect. Do not enter personal information."
)

try:
    config = json.loads(
        (ROOT / "configs/labels_public_v1.json").read_text(
            encoding="utf-8-sig"
        )
    )
except (OSError, ValueError):
    st.error("Could not load configs/labels_public_v1.json.")
    st.stop()

classify_tab, results_tab, guide_tab = st.tabs([
    "Classify",
    "Model comparison",
    "Category guide",
])

with classify_tab:
    with st.form("prediction_form"):
        selected_model = st.selectbox(
            "Choose a model",
            options=list(MODEL_NAMES),
            format_func=lambda value: MODEL_NAMES[value],
        )
        message = st.text_area(
            "Customer message",
            placeholder="Mera order kahan hai? Tracking link bhej do.",
            max_chars=1000,
            height=130,
        )
        submitted = st.form_submit_button("Classify message")

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

                intent = result["intent"]
                if intent not in config["labels"]:
                    raise ValueError("Unknown category returned by API.")

                st.subheader(intent.replace("_", " ").title())
                st.write(config["labels"][intent])
                st.caption(f"Model: {result['model_version']}")
                st.caption(
                    f"API processing time: {result['latency_ms']} ms. "
                    "Excludes request queueing, network transfer, "
                    "and interface rendering."
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
    selected_split = st.radio(
        "Evaluation split",
        options=["Final test", "Validation"],
        horizontal=True,
    )

    report_name = (
        "public_test.json"
        if selected_split == "Final test"
        else "public_validation.json"
    )
    report_path = ROOT / "reports" / report_name

    if not report_path.exists():
        st.info(f"Evaluation report not found: {report_name}")
    else:
        try:
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )

            rows = []
            for model_key, name in MODEL_NAMES.items():
                metrics = report["models"][model_key]
                rows.append({
                    "Model": name,
                    "Accuracy": f"{metrics['accuracy']:.2%}",
                    "Macro-F1": f"{metrics['macro_f1']:.4f}",
                    "Mistakes": len(metrics["mistakes"]),
                })

            st.table(rows)
            st.caption(
                f"Training examples: {report['training_examples']} · "
                f"{selected_split} examples: "
                f"{report['evaluation_examples']}"
            )
            st.write(
                "Accuracy measures the fraction of correct predictions. "
                "Macro-F1 gives each category equal weight."
            )

            if selected_split == "Validation":
                st.info(
                    "Validation macro-F1 was used to select the LoRA "
                    "checkpoint. This is a development result."
                )
            else:
                st.info(
                    "The final test split was excluded from fitting and "
                    "checkpoint selection. Results apply to this cleaned "
                    "synthetic dataset, not real customer traffic."
                )

            st.caption(
                "Exact normalized text duplicates were removed across "
                "splits. Paraphrase leakage has not been audited."
            )

            mistake_model = st.selectbox(
                "Inspect mistakes",
                options=list(MODEL_NAMES),
                format_func=lambda value: MODEL_NAMES[value],
            )
            mistakes = report["models"][mistake_model]["mistakes"]

            if mistakes:
                st.dataframe(
                    mistakes,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.write("No mistakes recorded for this split.")

        except (OSError, ValueError, KeyError):
            st.error("Could not read the evaluation report.")

with guide_tab:
    for label, description in config["labels"].items():
        st.markdown(
            f"**{label.replace('_', ' ').title()}** — {description}"
        )

    st.markdown(
        "[Public dataset and documentation]"
        "(https://huggingface.co/datasets/"
        "Hari5115/hinglish-retail-intent-dataset)"
    )
    st.caption(
        "The source dataset lists an MIT license. "
        "Its messages are primarily synthetic Roman-script Hinglish. "
        "Devanagari Hindi performance has not been separately evaluated."
    )