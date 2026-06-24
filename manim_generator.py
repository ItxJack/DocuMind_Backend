"""
manim_generator.py
--------------------
Generates a short Manim animation from a plain-English idea, using Groq
(instead of Gemini) to write the Manim code and then rendering it locally
with manim + ffmpeg.

NOTE: This was originally built against Gemini (per the instructor's
reference script, Manim_Pipeline_A3.py), but Gemini's free-tier models
(gemini-2.5-flash and gemini-2.5-flash-lite) were experiencing persistent
"503 UNAVAILABLE / high demand" outages, so this endpoint was switched to
Groq, which is already used successfully by the /ask-llm endpoint in this
project and proved fast and reliable.

Pipeline:
    1. Ask Groq for Manim code based on the user's idea (ask_ai_for_manim_code)
    2. Extract the raw Python code from the LLM's markdown-fenced response
       (extract_python_code)
    3. Save that code to a uniquely-named .py file (generate_manim_video)
    4. Render it using `python -m manim` as a subprocess
    5. Locate the resulting .mp4 under manim's media/ output folder and
       return its path
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime

from groq import Groq

GENERATED_DIR = "generated"
MEDIA_DIR = "media"
GROQ_MODEL = "openai/gpt-oss-120b"

# Even Groq can occasionally hit transient errors (rate limits, brief
# outages). Retry a few times with a short delay before giving up.
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 5


def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file "
            "or your deployment platform's environment variables."
        )
    return Groq(api_key=api_key)


def ask_ai_for_manim_code(user_prompt: str) -> str:
    """
    Sends the user's animation idea to Groq with strict instructions to
    return ONLY raw Manim Python code (no Tex/MathTex, only Text()+VGroup,
    since LaTeX is not installed in this environment).
    """
    system_instruction = (
        "You are an expert in Python and the Manim animation library. "
        "When given a request, generate ONLY valid Python code using Manim. "
        "CRITICAL FONT RULES: You must NEVER use Tex() or MathTex(). "
        "You must ONLY use the standard Text() class for all text and equations. "
        "If you need to animate parts of an equation separately, you must create "
        "separate Text() objects and group them using VGroup "
        "(e.g., VGroup(Text('a^2'), Text('+'), Text('b^2'))). "
        "Never pass multiple comma-separated strings to a single Text() object. "
        "The scene class must be named 'GeneratedScene'. "
        "Do not include any explanations, greetings, or additional text. "
        "Just output the raw code block, wrapped in a ```python code fence."
    )

    client = _get_groq_client()

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"User Request: {user_prompt}"},
                ],
            )
            text = response.choices[0].message.content
            if not text:
                raise RuntimeError("Groq returned an empty response.")
            return text
        except Exception as e:
            last_error = e
            is_transient = "503" in str(e) or "UNAVAILABLE" in str(e) or "rate" in str(e).lower()
            if is_transient and attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS)
                continue
            break

    raise RuntimeError(
        f"Groq API call failed after {_MAX_RETRIES} attempt(s): {last_error}"
    )


def extract_python_code(raw_text: str) -> str:
    """
    Extracts the raw Python code from a markdown-fenced ```python ... ```
    block in the LLM's response. Falls back to the raw text if no fence
    is found (in case the model didn't wrap it in backticks).
    """
    backticks = chr(96) * 3
    pattern = rf"{backticks}(?:python)?\n(.*?)\n{backticks}"
    match = re.search(pattern, raw_text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return raw_text.strip()


# Absolute path to the "generated" folder, anchored to this file's own
# location rather than the process's current working directory.
# This matters because uvicorn --reload spawns a child reloader process
# whose working directory can differ from where you launched uvicorn,
# which previously caused the relative path "generated" to resolve
# incorrectly and made the Manim subprocess fail silently with no stderr.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_GENERATED_ABS_DIR = os.path.join(_BASE_DIR, GENERATED_DIR)


def generate_manim_video(idea: str) -> str:
    """
    Full pipeline: idea -> Groq code -> render -> return path to .mp4

    Args:
        idea: Plain-English description of the animation to create.

    Returns:
        Absolute path to the rendered .mp4 file.

    Raises:
        ValueError: if GROQ_API_KEY is missing.
        RuntimeError: if the Groq call fails, or if Manim fails to render
                      the generated code (e.g. the LLM produced invalid code).
        FileNotFoundError: if rendering reported success but the expected
                            output video could not be located.
    """
    os.makedirs(_GENERATED_ABS_DIR, exist_ok=True)

    raw_response = ask_ai_for_manim_code(idea)
    cleaned_code = extract_python_code(raw_response)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"animation_{timestamp}.py"
    script_path = os.path.join(_GENERATED_ABS_DIR, script_filename)

    with open(script_path, "w") as f:
        f.write(cleaned_code)

    # Render using Manim as a subprocess (mirrors the instructor's reference script)
    # -ql = quick/low quality (480p) -> fast renders, good for a demo API
    # -v WARNING = suppress noisy info logs, only show warnings/errors
    result = subprocess.run(
        [sys.executable, "-m", "manim", "-ql", "-v", "WARNING",
         script_filename, "GeneratedScene"],
        cwd=_GENERATED_ABS_DIR,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Manim failed to render the generated animation. "
            "This usually means the LLM produced invalid Manim code. "
            f"Try rephrasing your idea. Details: {result.stderr[-800:] or result.stdout[-800:]}"
        )

    # Manim outputs to: <cwd>/media/videos/<script_name_without_ext>/480p15/<SceneName>.mp4
    scene_name = "GeneratedScene"
    script_stem = script_filename.replace(".py", "")
    expected_video_path = os.path.join(
        _GENERATED_ABS_DIR, MEDIA_DIR, "videos", script_stem, "480p15", f"{scene_name}.mp4"
    )

    if os.path.exists(expected_video_path):
        return os.path.abspath(expected_video_path)

    # Fallback: search for any .mp4 created for this script, in case Manim's
    # internal folder naming differs slightly across versions.
    search_root = os.path.join(_GENERATED_ABS_DIR, MEDIA_DIR, "videos", script_stem)
    if os.path.isdir(search_root):
        for root, _, files in os.walk(search_root):
            for fname in files:
                if fname.endswith(".mp4"):
                    return os.path.abspath(os.path.join(root, fname))

    raise FileNotFoundError(
        "Manim reported a successful render, but the output video file "
        "could not be located on disk."
    )