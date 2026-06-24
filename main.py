"""
main.py
--------
FastAPI application for Assignment 3.

Endpoints:
    GET  /                  -> Welcome message
    POST /chunk-text        -> Required: split large text into chunks
    POST /upload-file       -> Bonus: upload a CSV or PDF, get its text back
    POST /ask-llm           -> Bonus: ask a question, get an answer from Groq
    POST /generate-animation-> Bonus: describe an animation idea, get an mp4 back

Run locally with:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from chunk import chunk_text
from file_reader import read_csv_as_text, read_pdf_as_text
from llm_query import ask_groq
from manim_generator import generate_manim_video

# Load environment variables (GROQ_API_KEY, GEMINI_API_KEY) from a local
# .env file when running locally. On deployment platforms (Render, HF
# Spaces) these are set directly as environment variables instead.
load_dotenv()

app = FastAPI(
    title="Assignment 3 - Text Chunking & AI Utilities API",
    description=(
        "A FastAPI app that chunks large text using LangChain's "
        "RecursiveCharacterTextSplitter, with bonus endpoints for file "
        "upload, LLM Q&A, and AI-generated Manim animations."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request/response models (Pydantic) — these give us automatic input
# validation and clear auto-generated API docs.
# ---------------------------------------------------------------------------

class ChunkRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The large text to split into chunks.")
    chunk_size: int = Field(50, gt=0, description="Maximum characters per chunk.")
    chunk_overlap: int = Field(10, ge=0, description="Overlap (in characters) between chunks.")


class ChunkResponse(BaseModel):
    chunks: list[str]
    total_chunks: int


class AskLLMRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to ask the LLM.")


class AskLLMResponse(BaseModel):
    query: str
    answer: str


class AnimationRequest(BaseModel):
    idea: str = Field(..., min_length=1, description="Plain-English animation idea.")


# ---------------------------------------------------------------------------
# Required endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def welcome():
    """Welcome endpoint."""
    return {"message": "Welcome to the Assignment 3 FastAPI app! Visit /docs to try the endpoints."}


@app.post("/chunk-text", response_model=ChunkResponse)
def chunk_text_endpoint(payload: ChunkRequest):
    """
    Splits a large block of text into chunks using
    RecursiveCharacterTextSplitter.

    - **text**: the text to split (required)
    - **chunk_size**: max characters per chunk (default 50)
    - **chunk_overlap**: overlap between chunks (default 10)
    """
    try:
        chunks = chunk_text(
            text=payload.text,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
        )
    except ValueError as e:
        # e.g. chunk_overlap >= chunk_size
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while chunking text: {e}")

    return ChunkResponse(chunks=chunks, total_chunks=len(chunks))


# ---------------------------------------------------------------------------
# Bonus 1: CSV / PDF upload -> plain text
# ---------------------------------------------------------------------------

@app.post("/upload-file")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Accepts a .csv or .pdf file and returns its content as plain text.
    """
    filename = file.filename or ""
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if extension not in ("csv", "pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .csv or .pdf file.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        if extension == "csv":
            text = read_csv_as_text(file_bytes)
        else:
            text = read_pdf_as_text(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while reading file: {e}")

    return {"filename": filename, "extracted_text": text}


# ---------------------------------------------------------------------------
# Bonus 2: LLM query endpoint (Groq)
# ---------------------------------------------------------------------------

@app.post("/ask-llm", response_model=AskLLMResponse)
def ask_llm_endpoint(payload: AskLLMRequest):
    """
    Sends a query to an LLM (via Groq) and returns its answer.
    Requires the GROQ_API_KEY environment variable to be set.
    """
    try:
        answer = ask_groq(payload.query)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return AskLLMResponse(query=payload.query, answer=answer)


# ---------------------------------------------------------------------------
# Bonus 3: Manim animation generator (Gemini + Manim)
# ---------------------------------------------------------------------------

@app.post("/generate-animation")
def generate_animation_endpoint(payload: AnimationRequest):
    """
    Generates a short animation from a plain-English idea:
    1. Gemini writes Manim code for the idea.
    2. The code is rendered locally using Manim + ffmpeg.
    3. The resulting .mp4 video file is returned.

    Requires the GEMINI_API_KEY environment variable to be set.
    NOTE: This endpoint can take 15-60+ seconds depending on the complexity
    of the animation and Gemini's response time.
    """
    try:
        video_path = generate_manim_video(payload.idea)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=502, detail=str(e))

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path),
    )
