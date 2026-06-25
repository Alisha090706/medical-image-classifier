"""
llm_explainer.py
=================
Generates a plain-language explanation of a Grad-CAM result using an LLM
hosted on Groq's inference API.

Design principle
-----------------
The LLM is never asked to "look at" or interpret the heatmap image directly.
Instead, gradcam.quantify_heatmap_region() first computes precise, objective
facts about WHERE the activation is concentrated (which 3x3 grid region is
dominant, how much energy sits in the central/spine column versus the lung
fields). Those facts — numbers, not pixels — are what get sent to the LLM as
text. The LLM's only job is to phrase already-correct facts in plain
language; it is not asked to perceive or judge the image itself, which would
be unreliable and unverifiable.

The explanation is always written as a description of model behavior
("the model focused on..."), never as a clinical claim about the patient's
actual health, and the output always includes a reminder that this is not a
diagnosis.

If no Groq API key is configured, or if the API call fails for any reason
(network issue, rate limit, invalid key), this module returns a clear
fallback message rather than crashing the app — the rest of MediScan AI
(prediction, raw Grad-CAM images, metrics) must keep working regardless of
whether this optional feature is available.
"""

import os
from typing import Optional

# The groq package is an optional dependency: only required if the user
# actually wants to use this feature. Importing it lazily inside functions
# (rather than at module level) means the rest of the app still runs fine
# even if `groq` is not installed and this feature is simply unavailable.
GROQ_MODEL: str = "llama-3.3-70b-versatile"

FALLBACK_MESSAGE: str = (
    "AI-generated explanation is unavailable. Add a GROQ_API_KEY environment "
    "variable to enable plain-language Grad-CAM explanations, or refer to "
    "the heatmap and the description above."
)


def is_explainer_configured() -> bool:
    """
    Check whether the Groq explainer can be used, i.e. whether a
    GROQ_API_KEY environment variable has been set.

    Returns:
        True if a key is present (non-empty), False otherwise. This lets
        app.py decide whether to show the "Explain in plain language"
        button at all, rather than showing it and then failing.
    """
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


def _build_prompt(
    predicted_class: str,
    confidence: float,
    dominant_region: str,
    dominant_region_strength: float,
    midline_ratio: float,
    is_midline_dominant: bool,
) -> str:
    """
    Build the user-facing prompt sent to the LLM, embedding only the
    objectively-computed heatmap facts — no image data.

    Args:
        predicted_class: "NORMAL" or "PNEUMONIA", the model's prediction.
        confidence: Softmax probability of the predicted class, in [0, 1].
        dominant_region: Grid region with highest average activation, e.g.
            "upper-left".
        dominant_region_strength: Average activation in that dominant
            region, in [0, 1].
        midline_ratio: Fraction of total activation energy in the central
            column, in [0, 1].
        is_midline_dominant: Whether midline_ratio exceeds the threshold
            used to flag spine/mediastinum-centered activation.

    Returns:
        A prompt string instructing the LLM to explain these facts in plain
        language, with explicit constraints on tone and scope.
    """
    midline_note = (
        "Note: a large share of the model's attention fell on the central "
        "column of the image (where the spine and heart typically sit), "
        "rather than the lung fields on either side. This can indicate the "
        "model is partly relying on positional or midline image features "
        "rather than genuine lung tissue patterns, which is a known "
        "limitation worth mentioning."
        if is_midline_dominant
        else "The model's attention was concentrated away from the central "
        "midline, which is generally more consistent with focusing on lung "
        "tissue rather than the spine or heart shadow."
    )

    return (
        "You are writing a short, plain-language explanation of a Grad-CAM "
        "visualization for a chest X-ray deep learning classifier, aimed at "
        "a general audience (e.g. someone preparing to explain this project "
        "in a job interview). Use the following objectively measured facts "
        "only — do not invent additional visual details, and do not make "
        "any claim about the person's actual health or diagnosis.\n\n"
        f"- Model's prediction: {predicted_class} (confidence: {confidence * 100:.1f}%)\n"
        f"- The single region with the strongest model attention is: {dominant_region} "
        f"(on a 3x3 grid: rows are upper/middle/lower, columns are left/center/right)\n"
        f"- Average activation strength in that region: {dominant_region_strength:.2f} (scale 0 to 1)\n"
        f"- Fraction of total attention in the central (midline) column: {midline_ratio * 100:.0f}%\n"
        f"- {midline_note}\n\n"
        "Write 3-4 sentences in plain, conversational language explaining "
        "what this means about where the model focused and why that matters "
        "for trusting (or questioning) the prediction. End with one sentence "
        "reminding the reader this describes model behavior only, not a "
        "medical diagnosis."
    )


def explain_gradcam_result(
    predicted_class: str,
    confidence: float,
    dominant_region: str,
    dominant_region_strength: float,
    midline_ratio: float,
    is_midline_dominant: bool,
) -> str:
    """
    Call the Groq API to generate a plain-language explanation of a
    Grad-CAM result, based on precomputed, objective heatmap facts.

    Args:
        predicted_class: "NORMAL" or "PNEUMONIA".
        confidence: Softmax probability of the predicted class, in [0, 1].
        dominant_region: Grid region with highest average activation.
        dominant_region_strength: Average activation in that region.
        midline_ratio: Fraction of total activation energy in the central
            column.
        is_midline_dominant: Whether midline_ratio exceeds the flagging
            threshold.

    Returns:
        A plain-language explanation string from the LLM, or
        FALLBACK_MESSAGE if the API key is missing or the call fails for
        any reason. This function never raises — failures are caught and
        converted into a user-readable fallback so the rest of the app
        keeps working.
    """
    if not is_explainer_configured():
        return FALLBACK_MESSAGE

    try:
        # Imported here (not at module level) so the rest of the app does
        # not require the `groq` package to be installed unless this
        # specific optional feature is actually used.
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])

        prompt = _build_prompt(
            predicted_class=predicted_class,
            confidence=confidence,
            dominant_region=dominant_region,
            dominant_region_strength=dominant_region_strength,
            midline_ratio=midline_ratio,
            is_midline_dominant=is_midline_dominant,
        )

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain machine learning model behavior in "
                        "clear, honest, plain language. You never make "
                        "medical diagnostic claims, and you always note "
                        "model limitations when the data suggests them."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=GROQ_MODEL,
            temperature=0.4,
            max_completion_tokens=300,
        )

        explanation: Optional[str] = response.choices[0].message.content
        return explanation.strip() if explanation else FALLBACK_MESSAGE

    except Exception:
        # Any failure (missing package, invalid key, network error, rate
        # limit, malformed response) falls back gracefully rather than
        # crashing the Streamlit app. The specific error is intentionally
        # not surfaced to the end user here, since this is a non-critical
        # supplementary feature; the rest of the app must keep working.
        return FALLBACK_MESSAGE
