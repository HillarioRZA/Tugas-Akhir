import io
import json
import base64
import os
from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form, Header
from fastapi.responses import StreamingResponse
from fastapi.responses import Response, JSONResponse
from backend.services.eda import main as eda_main
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
    
    result = agent_main.run_agent_flow(
        session_id=session_id, 
        prompt=prompt, 
        new_file_path=new_file_path, 
        new_dataset_name=new_dataset_name 
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result.get("detail", result["error"]))

    final_response = result
    final_response["session_id"] = session_id
    return JSONResponse(content=final_response)

@router.post("/custom-visualize", response_class=JSONResponse)
async def create_custom_visualization(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    if not x_session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = x_session_id

    contents = None
    file_type = None

    if file and file.filename:
        if file.filename.endswith('.csv'):
            file_type = 'csv'
            contents = await file.read()
        else:
            raise HTTPException(status_code=400, detail="Hanya file CSV yang didukung untuk visualisasi kustom.")
    
    if not contents:
         raise HTTPException(status_code=400, detail="File CSV dibutuhkan untuk membuat visualisasi kustom.")

    plot_plan = agent_main.get_plot_plan(prompt)
    if "error" in plot_plan:
        raise HTTPException(status_code=500, detail=plot_plan.get("detail", plot_plan["error"]))

    image_bytes = eda_main.generate_custom_plot(
        file_contents=contents,
        plot_type=plot_plan.get("plot_type"),
        x_col=plot_plan.get("x_col"),
        y_col=plot_plan.get("y_col"),
        hue_col=plot_plan.get("hue_col"),
        orientation=plot_plan.get("orientation", 'v')
    )

    if isinstance(image_bytes, str):
        raise HTTPException(status_code=500, detail=f"Gagal membuat plot: {image_bytes}")

    from backend.services.agent.interpretation import get_interpretation
    summary = get_interpretation(
        session_id=session_id,
        tool_name=plot_plan.get("plot_type", "custom plot"),
        tool_output=plot_plan,
        image_bytes=image_bytes
    )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    final_response = {
        "plan": plot_plan,
        "summary": summary,
        "image_base64": image_b64,
        "image_format": "png",
        "session_id": session_id
    }
    return JSONResponse(content=final_response)