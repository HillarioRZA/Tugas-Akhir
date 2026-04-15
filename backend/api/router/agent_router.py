import io
import json
import base64
import os
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse
from fastapi.responses import Response, JSONResponse
import asyncio
from backend.services.agent import main as agent_main

from typing import Optional,List,Dict
import uuid

router = APIRouter(
    prefix="/api/agent",
    tags=["Agent"]
)

@router.post("/execute")
async def execute_agent_action(
    file: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    contents = None
    file_type = None
    available_columns = []
    uploaded_file: Optional[UploadFile] = None

    if not x_session_id:
        session_id = str(uuid.uuid4())
        print(f"Membuat Session ID baru: {session_id}")
    else:
        session_id = x_session_id
        print(f"Menggunakan Session ID yang ada: {session_id}")

    new_file_path: Optional[str] = None
    new_dataset_name: Optional[str] = None
    available_columns = []

    if file and file.filename:
            if not (file.filename.endswith('.csv') or file.filename.endswith('.pdf')):
                raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV atau PDF.")
                
            session_upload_dir = os.path.join("user_uploads", session_id) 
            os.makedirs(session_upload_dir, exist_ok=True)
            
            new_dataset_name = file.filename
            new_file_path = os.path.join(session_upload_dir, new_dataset_name)

            try:
                contents = await file.read() 
                with open(new_file_path, "wb") as f:
                    f.write(contents)
                print(f"File disimpan ke: {new_file_path}") 
            except Exception as e:
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                raise HTTPException(status_code=500, detail=f"Gagal menyimpan file ke disk: {str(e)}")

            if not (new_dataset_name.endswith('.csv') or new_dataset_name.endswith('.pdf')):
                raise HTTPException(status_code=400, detail="Format file tidak valid. Harap unggah file CSV atau PDF.")
    
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent_main.run_agent_flow,
                session_id,
                prompt,
                new_file_path,
                new_dataset_name,
            ),
            timeout=120.0  # ARCH-1 fix: 120 detik global timeout
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Agent Timeout",
                "detail": (
                    "Agent tidak merespons dalam 120 detik. "
                    "Coba lagi dengan pertanyaan yang lebih sederhana atau periksa koneksi API."
                ),
                "session_id": session_id,
            }
        )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result.get("detail", result["error"]))

    final_response = result
    final_response["session_id"] = session_id
    return JSONResponse(content=final_response)

# Endpoint /custom-visualize dihapus (BUG-1 fix).
# generate_custom_plot sudah tidak tersedia di eda/main.py.
# Gunakan 5 visualization tools via Agent: plot_itinerary_scatter,
# plot_distribution_histogram, plot_category_bar, plot_correlation_heatmap,
# plot_budget_breakdown.