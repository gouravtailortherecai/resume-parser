import os
import json
import io
import tempfile
from typing import Optional

import requests
import pdfplumber
import docx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import boto3  # only if you want to fetch service-account JSON from AWS Secrets Manager (not used here)
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

def extract_text_from_pdf_file(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text += p.extract_text() or ""
    return text.strip()

def extract_text_from_docx_file(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text]).strip()

def extract_text_from_file(path: str, mime_type: Optional[str]) -> str:
    if mime_type == "application/pdf" or path.lower().endswith(".pdf"):
        return extract_text_from_pdf_file(path)
    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ) or path.lower().endswith(".docx") or path.lower().endswith(".doc"):
        return extract_text_from_docx_file(path)
    # fallback: try reading as text
    with open(path, "r", errors="ignore") as f:
        return f.read()

def call_groq_api(cv_text: str):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a resume parser. Extract name, email, phone, skills, experience, education and return JSON only."},
            {"role": "user", "content": cv_text}
        ],
        "temperature": 0.0
    }
    r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # The Groq response normally contains the assistant message in choices[0].message.content
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    # try to parse JSON if Groq returned JSON text
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content}

@app.post("/parse")
def parse_resume(req: ParseRequest):
    file_id = req.fileId
    mime_type = req.mimeType

    if not file_id:
        raise HTTPException(status_code=400, detail="fileId required")

    try:
        request = DRIVE_SERVICE.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)

        suffix = ".pdf" if (mime_type == "application/pdf") else (".docx" if (mime_type and "word" in mime_type) else "")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(fh.read())
            tmp_path = tmp.name

        cv_text = extract_text_from_file(tmp_path, mime_type)
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")

        parsed = call_groq_api(cv_text)
        return {"message": "parsed", "fileId": file_id, "parsedData": parsed}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
