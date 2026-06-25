"""
app.py
======
MediScan AI — a Streamlit web application for chest X-ray pneumonia
detection. Provides an end-to-end interface: upload an X-ray, get a
prediction with a confidence score, see a Grad-CAM explanation of the
model's reasoning, and review the model's overall test-set performance.

This file only orchestrates the UI. All model logic (prediction, Grad-CAM)
lives in predict.py and gradcam.py respectively, so the app simply imports
and calls into them — there is no duplicated inference code here.

Run with:
    streamlit run app.py
"""

import json
import os
from typing import Optional

import streamlit as st
from PIL import Image

from evaluate import CONFUSION_MATRIX_PATH, METRICS_JSON_PATH
from gradcam import generate_gradcam, quantify_heatmap_region
from llm_explainer import explain_gradcam_result, is_explainer_configured
from predict import load_model_for_inference, predict_image
from train import MODEL_SAVE_PATH, get_device
from utils.dataset import CLASS_NAMES

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit command executed.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediScan AI · Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]


# ---------------------------------------------------------------------------
# Visual design system
# ---------------------------------------------------------------------------
# A clinical reading-room palette: a deep navy backdrop (similar to a PACS /
# radiology viewer), a single desaturated teal accent for interactive
# elements, and reserved semantic colors (amber for a positive pneumonia
# finding, green for a normal finding) used ONLY in the result readout —
# never decoratively — so color always carries a real signal in this app.
#
# Typography pairs IBM Plex Sans (precise, clinical, technical-grade
# grotesk used widely in data/instrument software) for interface text with
# IBM Plex Mono for numeric readouts (confidence, metrics), echoing the
# look of a measurement instrument rather than a generic web app.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg-deep: #0B1A2B;
    --bg-panel: #122036;
    --bg-card: #16273F;
    --border-soft: #25405E;
    --text-primary: #E7EDF4;
    --text-muted: #8FA3BD;
    --accent: #4FB3AC;
    --accent-soft: rgba(79, 179, 172, 0.15);
    --finding-normal: #4CAF82;
    --finding-pneumonia: #E0A23D;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: var(--bg-deep);
    color: var(--text-primary);
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--border-soft);
}

/* Eyebrow label style used above section headings */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.3rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.15;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.02rem;
    max-width: 640px;
    line-height: 1.55;
}

/* Generic card surface used throughout the app */
.scan-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.scan-card h4 {
    margin-top: 0;
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Diagnostic readout block — the signature element: styled like an
   instrument measurement panel rather than a standard progress bar */
.readout {
    border-radius: 10px;
    padding: 1.6rem;
    border: 1px solid var(--border-soft);
}

.readout-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.readout-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1.1;
    margin: 0.2rem 0 0.6rem 0;
}

.readout-value.normal { color: var(--finding-normal); }
.readout-value.pneumonia { color: var(--finding-pneumonia); }

.confidence-track {
    width: 100%;
    height: 10px;
    border-radius: 6px;
    background-color: #0B1A2B;
    overflow: hidden;
    border: 1px solid var(--border-soft);
}

.confidence-fill {
    height: 100%;
    border-radius: 6px;
}

.confidence-fill.normal { background-color: var(--finding-normal); }
.confidence-fill.pneumonia { background-color: var(--finding-pneumonia); }

.confidence-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}

/* Metric tiles in the Model Metrics section */
.metric-tile {
    background-color: var(--bg-card);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    text-align: left;
}

.metric-tile .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.metric-tile .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: var(--accent);
    margin-top: 0.25rem;
}

hr {
    border-color: var(--border-soft);
}

.footer-note {
    color: var(--text-muted);
    font-size: 0.82rem;
    line-height: 1.6;
    border-top: 1px solid var(--border-soft);
    padding-top: 1rem;
    margin-top: 2rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resource loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """
    Load the trained model once per Streamlit server process and reuse it
    across all user interactions, rather than reloading the checkpoint from
    disk on every single upload (which would be slow, especially on CPU).

    Returns:
        Tuple of (model, device) if the checkpoint exists, or (None, device)
        if no trained model has been saved yet.
    """
    device = get_device()
    if not os.path.isfile(MODEL_SAVE_PATH):
        return None, device
    model = load_model_for_inference(MODEL_SAVE_PATH, device)
    return model, device


def load_test_metrics() -> Optional[dict]:
    """
    Load the saved test-set metrics produced by evaluate.py, if available.

    Returns:
        A dict with keys "accuracy", "precision", "recall", "f1_score",
        "num_test_samples", or None if evaluate.py has not been run yet.
    """
    if not os.path.isfile(METRICS_JSON_PATH):
        return None
    with open(METRICS_JSON_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar — page navigation, model information & runtime context
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='eyebrow'>MediScan AI</div>"
        "<div style='font-size:1.3rem; font-weight:700; margin-bottom:0.8rem;'>"
        "Pneumonia Detection</div>",
        unsafe_allow_html=True,
    )

    # Page navigation. Streamlit reruns the whole script on every interaction,
    # so the selected page is just a plain variable used later to decide
    # which render_*_page() function to call — no extra state management
    # needed beyond this radio widget's own built-in state.
    selected_page = st.radio(
        label="Navigate",
        options=["🔬 Detection", "📚 Learn About Pneumonia"],
        label_visibility="collapsed",
    )

    st.markdown("<br/>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='scan-card'>
            <h4>Model</h4>
            <p style='margin:0; color:var(--text-primary); font-size:0.92rem;'>
            EfficientNet-B0 (ImageNet pretrained)<br/>
            Frozen backbone &middot; fine-tuned classifier head
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, device = get_cached_model()
    device_label = str(device).upper()
    model_status = "Loaded" if model is not None else "Not found"
    status_color = "var(--finding-normal)" if model is not None else "var(--finding-pneumonia)"

    st.markdown(
        f"""
        <div class='scan-card'>
            <h4>Runtime</h4>
            <p style='margin:0 0 0.4rem 0; font-size:0.92rem;'>
                Device: <span style='font-family:"IBM Plex Mono", monospace;'>{device_label}</span>
            </p>
            <p style='margin:0; font-size:0.92rem;'>
                Checkpoint: <span style='color:{status_color}; font-weight:600;'>{model_status}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='scan-card'>
            <h4>Classes</h4>
            <p style='margin:0; font-size:0.92rem;'>NORMAL &nbsp;&middot;&nbsp; PNEUMONIA</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='footer-note'>For research and portfolio demonstration "
        "only. Not a certified diagnostic device and must not be used for "
        "real clinical decisions.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page: Detection — the original upload / predict / explain / metrics flow
# ---------------------------------------------------------------------------
def render_detection_page() -> None:
    """
    Render the main detection workflow: hero section, X-ray upload,
    prediction with confidence score, Grad-CAM explainability, and the
    model's saved test-set performance metrics.

    This is the original single-page app, now wrapped in a function so it
    can be selected via the sidebar page navigation alongside the
    educational "Learn About Pneumonia" page.
    """
    st.markdown("<div class='eyebrow'>Chest X-Ray Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>MediScan AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>An assistive deep learning system that "
        "screens chest X-rays for radiographic signs of pneumonia. Upload an "
        "image to get a prediction, a confidence readout, and a Grad-CAM "
        "explanation of the regions the model focused on.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    if model is None:
        st.warning(
            f"No trained model checkpoint was found at `{MODEL_SAVE_PATH}`. "
            f"Run `python train.py` to train and save a model before using "
            f"this app.",
            icon="⚠️",
        )

    st.markdown("---")

    # --- Upload section ---
    st.markdown("<div class='eyebrow'>Step 1</div>", unsafe_allow_html=True)
    st.markdown("#### Upload a chest X-ray")

    uploaded_file = st.file_uploader(
        label="Accepted formats: PNG, JPG, JPEG",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=False,
    )

    uploaded_image: Optional[Image.Image] = None
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)

    st.markdown("---")

    # --- Prediction + Explainability sections ---
    if uploaded_image is not None and model is not None:
        with st.spinner("Analyzing X-ray..."):
            prediction = predict_image(uploaded_image, model, device)
            gradcam_result = generate_gradcam(uploaded_image, model, device)

        st.markdown("<div class='eyebrow'>Step 2</div>", unsafe_allow_html=True)
        st.markdown("#### Prediction")

        col_image, col_readout = st.columns([1, 1], gap="large")

        with col_image:
            st.markdown("<div class='scan-card'><h4>Uploaded X-Ray</h4></div>", unsafe_allow_html=True)
            st.image(uploaded_image, use_container_width=True)

        with col_readout:
            is_pneumonia = prediction.predicted_class == "PNEUMONIA"
            finding_class_css = "pneumonia" if is_pneumonia else "normal"
            confidence_pct = prediction.confidence * 100
            readout_border = "var(--finding-pneumonia)" if is_pneumonia else "var(--finding-normal)"

            st.markdown(
                f"""
                <div class='readout' style='border-color:{readout_border};'>
                    <div class='readout-label'>Predicted Finding</div>
                    <div class='readout-value {finding_class_css}'>{prediction.predicted_class}</div>
                    <div class='confidence-track'>
                        <div class='confidence-fill {finding_class_css}' style='width:{confidence_pct:.1f}%;'></div>
                    </div>
                    <div class='confidence-caption'>CONFIDENCE&nbsp;&nbsp;{confidence_pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='scan-card'>
                    <h4>Class Probabilities</h4>
                    <p style='margin:0.3rem 0; font-family:"IBM Plex Mono", monospace; font-size:0.92rem;'>
                        NORMAL &nbsp;&nbsp;&nbsp; {prediction.normal_probability * 100:.1f}%
                    </p>
                    <p style='margin:0.3rem 0; font-family:"IBM Plex Mono", monospace; font-size:0.92rem;'>
                        PNEUMONIA &nbsp; {prediction.pneumonia_probability * 100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # --- Explainability section ---
        st.markdown("<div class='eyebrow'>Step 3</div>", unsafe_allow_html=True)
        st.markdown("#### Explainability — Grad-CAM")
        st.markdown(
            "<p style='color:var(--text-muted); font-size:0.92rem; max-width:680px;'>"
            "The heatmap highlights the regions of the X-ray that most "
            "influenced the model's prediction. Warmer colors (red/yellow) "
            "indicate areas the model weighted most heavily.</p>",
            unsafe_allow_html=True,
        )

        col_heatmap, col_overlay = st.columns([1, 1], gap="large")

        with col_heatmap:
            st.markdown("<div class='scan-card'><h4>Activation Heatmap</h4></div>", unsafe_allow_html=True)
            st.image(gradcam_result.heatmap, use_container_width=True, clamp=True)

        with col_overlay:
            st.markdown("<div class='scan-card'><h4>Heatmap Overlay</h4></div>", unsafe_allow_html=True)
            st.image(gradcam_result.overlay, use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # --- Plain-language explanation, generated from objective heatmap
        # measurements via an LLM (Groq). This is an optional enhancement:
        # the app works fully without it, and the button only appears if a
        # GROQ_API_KEY has been configured.
        region_facts = quantify_heatmap_region(gradcam_result.heatmap)

        if is_explainer_configured():
            st.markdown(
                "<div class='scan-card'><h4>Plain-Language Explanation (AI-Generated)</h4></div>",
                unsafe_allow_html=True,
            )
            if st.button("Explain this heatmap in plain language", key="explain_gradcam_button"):
                with st.spinner("Generating explanation..."):
                    explanation_text = explain_gradcam_result(
                        predicted_class=prediction.predicted_class,
                        confidence=prediction.confidence,
                        dominant_region=region_facts["dominant_region"],
                        dominant_region_strength=region_facts["dominant_region_strength"],
                        midline_ratio=region_facts["midline_ratio"],
                        is_midline_dominant=region_facts["is_midline_dominant"],
                    )
                st.markdown(
                    f"""
                    <div class='scan-card' style='border-color:var(--accent);'>
                        <p style='margin:0; line-height:1.7; color:var(--text-primary);'>{explanation_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Generated by an LLM (via Groq) from measured heatmap "
                    "statistics — not a clinical interpretation. The "
                    "underlying numbers are computed directly from the "
                    "Grad-CAM output, not guessed by the language model."
                )
        else:
            st.caption(
                "💡 Set a `GROQ_API_KEY` environment variable to enable an "
                "AI-generated plain-language explanation of this heatmap."
            )

        st.markdown("---")

    elif uploaded_image is not None and model is None:
        st.error(
            "Cannot run a prediction because no trained model checkpoint is "
            "available. Run `python train.py` first.",
            icon="🚫",
        )

    # --- Model metrics section ---
    st.markdown("<div class='eyebrow'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("#### Test Set Metrics")

    test_metrics = load_test_metrics()

    if test_metrics is None:
        st.info(
            "No saved metrics found yet. Run `python evaluate.py` after "
            "training to generate test-set metrics and a confusion matrix.",
            icon="ℹ️",
        )
    else:
        metric_cols = st.columns(4, gap="medium")
        metric_definitions = [
            ("Accuracy", test_metrics["accuracy"]),
            ("Precision", test_metrics["precision"]),
            ("Recall", test_metrics["recall"]),
            ("F1 Score", test_metrics["f1_score"]),
        ]

        for column, (label, value) in zip(metric_cols, metric_definitions):
            with column:
                st.markdown(
                    f"""
                    <div class='metric-tile'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value'>{value * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"<p style='color:var(--text-muted); font-size:0.85rem; margin-top:0.8rem;'>"
            f"Evaluated on {test_metrics['num_test_samples']} held-out test images "
            f"across classes: {', '.join(CLASS_NAMES)}.</p>",
            unsafe_allow_html=True,
        )

        if os.path.isfile(CONFUSION_MATRIX_PATH):
            with st.expander("View confusion matrix"):
                st.image(CONFUSION_MATRIX_PATH, use_container_width=False)

# ---------------------------------------------------------------------------
# Page: Learn About Pneumonia — static educational content
# ---------------------------------------------------------------------------
# This page contains general public-health information only. It is not
# personalized to any uploaded image or prediction, and it deliberately
# avoids giving medical advice (dosages, treatment instructions, or any
# response that reacts to "your" result) — its job is context, not diagnosis.
# Content is paraphrased from CDC, WHO, and Mayo Clinic public guidance.
def render_learn_page() -> None:
    """
    Render a static educational page covering what pneumonia is, general
    prevention measures, when to seek medical care, and external research
    resources. Entirely independent of the model — no predictions or
    uploaded images are referenced here.
    """
    st.markdown("<div class='eyebrow'>Patient & Public Education</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>Learn About Pneumonia</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>General health information for context. "
        "This page is educational only — it is not personalized advice and "
        "does not respond to any X-ray prediction.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    st.info(
        "This information is general and for educational purposes only. "
        "It does not replace professional medical advice, diagnosis, or "
        "treatment. Always consult a qualified healthcare provider with "
        "questions about a specific health condition.",
        icon="ℹ️",
    )
    st.markdown("---")

    # --- About Pneumonia ---
    st.markdown("<div class='eyebrow'>Overview</div>", unsafe_allow_html=True)
    st.markdown("#### What Is Pneumonia?")
    st.markdown(
        """
        <div class='scan-card'>
            <p style='margin:0 0 0.8rem 0; line-height:1.7;'>
            Pneumonia is an infection that inflames the air sacs in one or
            both lungs. It can be caused by bacteria, viruses, or fungi, and
            less commonly by parasites. The air sacs may fill with fluid or
            pus, which is what produces the cloudy, patchy appearance often
            seen on a chest X-ray.
            </p>
            <p style='margin:0 0 0.8rem 0; line-height:1.7;'>
            <strong style='color:var(--text-primary);'>Common causes</strong><br/>
            Bacterial pneumonia (often caused by <em>Streptococcus
            pneumoniae</em>) and viral pneumonia (caused by viruses such as
            influenza, RSV, or COVID-19) are the most frequent types
            encountered in the community. Healthcare-acquired pneumonia,
            picked up during a hospital stay, often involves different,
            sometimes more resistant organisms.
            </p>
            <p style='margin:0; line-height:1.7;'>
            <strong style='color:var(--text-primary);'>Common symptoms</strong><br/>
            Cough (often producing mucus), fever or chills, shortness of
            breath, chest pain that worsens with breathing or coughing, and
            fatigue. Older adults sometimes show subtler signs, such as
            confusion or a drop in alertness, rather than a high fever.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Prevention ---
    st.markdown("<div class='eyebrow'>Prevention</div>", unsafe_allow_html=True)
    st.markdown("#### General Prevention Measures")
    st.markdown(
        """
        <div class='scan-card'>
            <p style='margin:0 0 0.8rem 0; line-height:1.7;'>
            <strong style='color:var(--text-primary);'>Vaccination</strong><br/>
            Pneumococcal and influenza vaccines are the most effective tools
            for lowering pneumonia risk, and are generally recommended for
            young children, adults 50 and older, and anyone with certain
            chronic health conditions. A healthcare provider can advise on
            which vaccines are appropriate for a given age and health
            profile.
            </p>
            <p style='margin:0 0 0.8rem 0; line-height:1.7;'>
            <strong style='color:var(--text-primary);'>Everyday hygiene</strong><br/>
            Regular handwashing, covering coughs and sneezes, and avoiding
            close contact with people who are sick all reduce the spread of
            the bacteria and viruses that commonly cause pneumonia.
            </p>
            <p style='margin:0; line-height:1.7;'>
            <strong style='color:var(--text-primary);'>Risk factors worth knowing</strong><br/>
            Risk is higher in young children, adults 65 and older, smokers,
            and people with underlying conditions such as asthma, COPD,
            diabetes, or a weakened immune system. Avoiding smoking and
            managing chronic conditions well are both protective.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- When to see a doctor ---
    st.markdown("<div class='eyebrow'>Seeking Care</div>", unsafe_allow_html=True)
    st.markdown("#### When to See a Doctor")
    st.markdown(
        """
        <div class='scan-card'>
            <p style='margin:0 0 0.8rem 0; line-height:1.7;'>
            It's a good idea to talk to a healthcare provider if cough,
            fever, or breathing difficulty doesn't improve in a few days, or
            worsens at any point.
            </p>
            <p style='margin:0; color:var(--finding-pneumonia); font-weight:600; line-height:1.7;'>
            Seek urgent medical attention for any of the following:
            </p>
            <ul style='margin:0.5rem 0 0 0; line-height:1.8; color:var(--text-primary);'>
                <li>Difficulty breathing or shortness of breath at rest</li>
                <li>Chest pain that is severe or persistent</li>
                <li>A high fever (especially 102°F / 39°C or above) that doesn't respond to medication</li>
                <li>Confusion, disorientation, or a sudden drop in alertness</li>
                <li>Bluish lips or face, which can indicate low oxygen levels</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Research & study resources ---
    st.markdown("<div class='eyebrow'>Further Reading</div>", unsafe_allow_html=True)
    st.markdown("#### Research & Study Resources")
    st.markdown(
        """
        <div class='scan-card'>
            <p style='margin:0 0 0.8rem 0; line-height:1.7; color:var(--text-muted);'>
            For deeper reading on pneumonia itself, or on the deep learning
            techniques used in this project:
            </p>
            <ul style='margin:0; line-height:2; color:var(--text-primary);'>
                <li><strong>CDC — Pneumonia</strong><br/>
                    <a href="https://www.cdc.gov/pneumonia/index.html" target="_blank" style="color:var(--accent);">cdc.gov/pneumonia</a></li>
                <li><strong>World Health Organization — Pneumonia Fact Sheet</strong><br/>
                    <a href="https://www.who.int/news-room/fact-sheets/detail/pneumonia" target="_blank" style="color:var(--accent);">who.int — Pneumonia fact sheet</a></li>
                <li><strong>Kaggle — Chest X-Ray Images (Pneumonia) Dataset</strong><br/>
                    <a href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia" target="_blank" style="color:var(--accent);">kaggle.com — dataset used to train this model</a></li>
                <li><strong>Selvaraju et al., 2017 — Grad-CAM paper</strong><br/>
                    <a href="https://arxiv.org/abs/1610.02391" target="_blank" style="color:var(--accent);">arxiv.org/abs/1610.02391</a>, the explainability technique used in this app</li>
                <li><strong>Tan & Le, 2019 — EfficientNet paper</strong><br/>
                    <a href="https://arxiv.org/abs/1905.11946" target="_blank" style="color:var(--accent);">arxiv.org/abs/1905.11946</a>, the backbone architecture used in this project</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='footer-note'>Health information on this page is "
        "general public guidance, not a substitute for professional "
        "medical advice. MediScan AI is a portfolio deep learning project "
        "and is not a certified diagnostic device.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if selected_page == "🔬 Detection":
    render_detection_page()
else:
    render_learn_page()
