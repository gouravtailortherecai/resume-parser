import os
import json
import io
import tempfile
from typing import Optional

import httpx
import pdfplumber
import docx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Config from env
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")  # JSON string

if not GROQ_API_KEY or not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError("Set GROQ_API_KEY and GOOGLE_SERVICE_ACCOUNT_JSON environment variables")

SERVICE_ACCOUNT_INFO = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Build drive service once at startup
_creds = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO, scopes=SCOPES
)
DRIVE_SERVICE = build("drive", "v3", credentials=_creds)

app = FastAPI(title="Resume Parser")

class ParseRequest(BaseModel):
    fileId: str
    mimeType: Optional[str] = None

def extract_text_from_pdf_bytes(data: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for p in pdf.pages:
            text += p.extract_text() or ""
    return text.strip()

def extract_text_from_docx_bytes(data: bytes) -> str:
    doc = docx.Document(io.BytesIO(data))
    return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()

def extract_text_from_bytes(data: bytes, mime_type: Optional[str]) -> str:
    # PDF
    if mime_type == "application/pdf":
        return extract_text_from_pdf_bytes(data)
    # DOCX
    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return extract_text_from_docx_bytes(data)
    # Fallback: try reading as text
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""
    
async def call_groq_api(cv_text: str, timeout: float = 30.0):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openai/gpt-oss-120b",  # same as your Supabase config
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a resume parser. "
                    "Extract candidate details and return JSON with this schema:\n"
                    "{\n"
                    '  "name": string,\n'
                    '  "email": string,\n'
                    '  "phone": string,\n'
                    '  "skills": [string],\n'
                    '  "experience": [string],\n'
                    '  "education": [string]\n'
                    "}"
                )
            },
            {"role": "user", "content": cv_text}
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}  # ✅ Ensures JSON output
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(GROQ_API_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return json.loads(content)


@app.post("/parse")
async def parse_resume(req: ParseRequest, request: Request):
    file_id = req.fileId
    mime_type = req.mimeType

    if not file_id:
        raise HTTPException(status_code=400, detail="fileId required")

    # API timeout (seconds)
    api_timeout = float(request.headers.get("X-API-Timeout", 30.0))

    try:
        # Download file from Google Drive (blocking, but Google API is not async)
        request_drive = DRIVE_SERVICE.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request_drive)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)

        file_bytes = fh.read()
        cv_text = extract_text_from_bytes(file_bytes, mime_type)
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        parsed = await call_groq_api(cv_text, timeout=api_timeout)
        return {"message": "parsed", "fileId": file_id, "parsedData": parsed}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
