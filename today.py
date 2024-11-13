import os
import json
import tempfile
import time
import logging
import re
from typing import Dict, List, Tuple, Optional
import httpx
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pymongo import MongoClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from datetime import datetime
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from bson.binary import Binary
from anthropic import Anthropic
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Setting API keys and endpoints
form_recognizer_endpoint = os.getenv("AZURE_OCR_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_OCR_KEY")
mongo_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
document_analysis_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint, 
    credential=AzureKeyCredential(form_recognizer_key)
)
mongo_client = MongoClient(mongo_uri)
db = mongo_client[database_name]
collection = db[collection_name]
anthropic_client = Anthropic(api_key=anthropic_api_key)

# FastAPI app
app = FastAPI()

# Scheduler configuration
scheduler = BackgroundScheduler()

class OCRRequest(BaseModel):
    document_id: str
    fields_to_extract: List[str]

def convert_bbox_format(bbox: List[float], page_width: float, page_height: float) -> Dict[str, float]:
    """
    Convert bounding box coordinates with proper scaling based on page dimensions.
    """
    if len(bbox) != 8:
        raise ValueError("Bounding box must contain exactly 8 values (4 points)")
    
    x_coords = [bbox[i] for i in range(0, len(bbox), 2)]
    y_coords = [bbox[i] for i in range(1, len(bbox), 2)]
    
    valid_x = [x for x in x_coords if x >= 0]
    valid_y = [y for y in y_coords if y >= 0]
    
    if not valid_x or not valid_y:
        return {
            "x": 0,
            "y": 0,
            "width": 10,
            "height": 10
        }
    
    # Convert normalized coordinates to actual page coordinates
    x = min(valid_x) * page_width
    y = min(valid_y) * page_height
    width = (max(valid_x) - min(valid_x)) * page_width
    height = (max(valid_y) - min(valid_y)) * page_height
    
    # Ensure minimum dimensions
    MIN_DIMENSION = 10
    width = max(width, MIN_DIMENSION)
    height = max(height, MIN_DIMENSION)
    
    return {
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(width, 2),
        "height": round(height, 2)
    }

def find_all_instances_of_text(ocr_data: Dict, search_text: str) -> List[Tuple[List[float], int, float, float]]:
    """Find all instances of text with improved matching logic"""
    words = search_text.lower().split()
    text_instances = []
    word_count = len(words)
    seen_instances = set()
    
    # Group words by page and store page dimensions
    pages = {}
    for page in ocr_data["extracted_data"]:
        pages[page["page_number"]] = {
            "words": [],
            "width": page["width"],
            "height": page["height"]
        }
        for word in page["words"]:
            pages[page["page_number"]]["words"].append({
                **word,
                "text_lower": word["content"].lower()
            })
    
    for page_num, page_data in pages.items():
        ocr_words = page_data["words"]
        page_width = page_data["width"]
        page_height = page_data["height"]
        
        for i in range(len(ocr_words)):
            if i + word_count > len(ocr_words):
                break
                
            matched_words = []
            current_word_idx = i
            word_idx = 0
            
            while word_idx < word_count and current_word_idx < len(ocr_words):
                current_ocr_word = ocr_words[current_word_idx]["text_lower"]
                target_word = words[word_idx]
                
                # Improved word matching logic
                if (current_ocr_word == target_word or 
                    target_word in current_ocr_word or 
                    current_ocr_word in target_word):
                    matched_words.append(ocr_words[current_word_idx])
                    word_idx += 1
                elif len(matched_words) > 0:
                    # Include connecting words and punctuation
                    if (current_ocr_word.isalnum() or 
                        current_ocr_word in [',', '&', '-', '#', '.', '/', ' ']):
                        matched_words.append(ocr_words[current_word_idx])
                
                current_word_idx += 1
            
            if len(matched_words) >= word_count:
                # Calculate bounding box for the matched words
                min_x = min(word["x"] for word in matched_words)
                min_y = min(word["y"] for word in matched_words)
                max_x = max(word["x"] + word["width"] for word in matched_words)
                max_y = max(word["y"] + word["height"] for word in matched_words)
                
                # Convert to normalized coordinates
                bbox = [
                    min_x / page_width, min_y / page_height,
                    max_x / page_width, min_y / page_height,
                    max_x / page_width, max_y / page_height,
                    min_x / page_width, max_y / page_height
                ]
                
                instance_key = (
                    tuple(round(x, 4) for x in bbox),
                    page_num
                )
                
                if instance_key not in seen_instances:
                    text_instances.append((bbox, page_num, page_width, page_height))
                    seen_instances.add(instance_key)
    
    return text_instances

def process_ocr_data(ocr_data: Dict, fields_to_extract: List[str]) -> Dict[str, List[Dict]]:
    """Process OCR data with improved text extraction and field matching"""
    logger.info("Starting OCR data processing")
    
    if not isinstance(ocr_data, dict):
        raise ValueError("Invalid OCR data format")
    
    results = {field: [] for field in fields_to_extract}
    seen_instances = {field: set() for field in fields_to_extract}
    
    # Reconstruct full text with better formatting
    full_text = []
    for page in ocr_data["extracted_data"]:
        # Group words by lines based on y-coordinates
        words = sorted(page["words"], key=lambda w: (w["y"], w["x"]))
        
        if not words:
            continue
            
        current_line = []
        current_y = words[0]["y"]
        line_height_threshold = max(word["height"] for word in words) * 1.5
        
        page_lines = []
        for word in words:
            if abs(word["y"] - current_y) > line_height_threshold:
                if current_line:
                    page_lines.append(" ".join(w["content"] for w in current_line))
                current_line = [word]
                current_y = word["y"]
            else:
                current_line.append(word)
        
        if current_line:
            page_lines.append(" ".join(w["content"] for w in current_line))
        
        full_text.append(f"Page {page['page_number']}:\n" + "\n".join(page_lines))
    
    full_text = "\n\n".join(full_text)
    
    try:
        fields_str = ", ".join(fields_to_extract)
        prompt = f"""Extract the following fields from the text: {fields_str}

Rules:
- Extract exact matching words from the text
- Process pages in sequential order
- Each field must contain ONLY relevant information
- Maintain page order in extractions

Specific guidelines for each field:
- name: Look for full names (typically 2-3 words)
- address: Look for complete addresses including building numbers and street names
- city: Look for city names within addresses
- pincode: Look for postal/ZIP codes (typically 5-6 digits or alphanumeric)

Format each extraction as:
field_name: extracted_value

Text to analyze:
{full_text}"""

        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Process extracted fields
        field_extractions = response.content[0].text.strip().split('\n')
        
        for line in field_extractions:
            if ':' not in line:
                continue
            
            field_key, value = line.split(':', 1)
            field_key = field_key.strip().lower()
            value = value.strip()
            
            if not value or value.lower() in ['none', 'not found', 'n/a']:
                continue
                
            if field_key in fields_to_extract:
                # Find all instances of the extracted text
                instances = find_all_instances_of_text(ocr_data, value)
                
                for bbox, page_number, page_width, page_height in instances:
                    bbox_dict = convert_bbox_format(bbox, page_width, page_height)
                    
                    # Create a more precise instance key
                    instance_key = (
                        value.lower(),
                        page_number,
                        round(bbox_dict["x"], 0),
                        round(bbox_dict["y"], 0)
                    )
                    
                    if instance_key not in seen_instances[field_key]:
                        field_data = {
                            "content": value,
                            "x": bbox_dict["x"],
                            "y": bbox_dict["y"],
                            "width": bbox_dict["width"],
                            "height": bbox_dict["height"],
                            "pageIndex": page_number - 1
                        }
                        results[field_key].append(field_data)
                        seen_instances[field_key].add(instance_key)
        
        # Sort results by page and position
        for field in results:
            results[field] = sorted(
                results[field],
                key=lambda x: (x["pageIndex"], x["y"], x["x"])
            )
            
        # Log extraction results
        for field, values in results.items():
            logger.info(f"Extracted {len(values)} instances of {field}")
            
    except Exception as e:
        logger.error(f"Error processing fields: {str(e)}")
        raise
        
    return results

def analyze_document_chunked(document_path):
    """Analyze document with improved text extraction"""
    try:
        extracted_data = []
        with open(document_path, "rb") as document:
            poller = document_analysis_client.begin_analyze_document(
                "prebuilt-document", document=document)
            result = poller.result()
            
            for page in result.pages:
                page_data = {
                    "page_number": page.page_number,
                    "width": page.width if page.width else 612,  # Default letter width
                    "height": page.height if page.height else 792,  # Default letter height
                    "words": []
                }
                
                for word in page.words:
                    confidence = getattr(word, 'confidence', 0.0)
                    
                    # Get word coordinates
                    if hasattr(word, 'polygon') and len(word.polygon) >= 4:
                        x = word.polygon[0].x * page_data["width"]
                        y = word.polygon[0].y * page_data["height"]
                        width = (word.polygon[2].x - word.polygon[0].x) * page_data["width"]
                        height = (word.polygon[2].y - word.polygon[0].y) * page_data["height"]
                    elif hasattr(word, 'bounding_box'):
                        x = word.bounding_box.x * page_data["width"]
                        y = word.bounding_box.y * page_data["height"]
                        width = word.bounding_box.width * page_data["width"]
                        height = word.bounding_box.height * page_data["height"]
                    else:
                        continue  # Skip words without valid coordinates
                    
                    word_data = {
                        "content": word.content,
                        "confidence": confidence,
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                    
                    page_data["words"].append(word_data)
                
                # Sort words by position
                page_data["words"].sort(key=lambda w: (w["y"], w["x"]))
                extracted_data.append(page_data)
            
            return {
                "extracted_data": extracted_data
            }
            
    except Exception as e:
        logger.error(f"Error in analyze_document_chunked: {str(e)}")
        raise

def process_document(doc):
    """Process document and extract fields with proper resource management"""
    temp_file_path = None
    pdf_document = None
    
    try:
        logging.info(f"Processing document: {doc['_id']}")
        collection.update_one({"_id": doc["_id"]}, {"$set": {"status": "processing"}})
        
        temp_file_path = os.path.join(tempfile.gettempdir(), secure_filename(doc.get("filename", "temp_file")))
        
        # Get file data from various possible sources
        file_data = None
        if "file_data" in doc:
            file_data = doc["file_data"]
        elif "fileData" in doc:
            file_data = base64.b64decode(doc["fileData"])
        elif "image" in doc:
            response = httpx.get(doc["image"])
            file_data = response.content
        else:
            for key, value in doc.items():
                if isinstance(value, (bytes, str)) and len(value) > 100:
                    file_data = value
                    break
                    
        if file_data is None:
            raise ValueError("No suitable file data found in the document")
            
        if isinstance(file_data, str):
            file_data = file_data.encode('utf-8')
        elif isinstance(file_data, Binary):
            file_data = file_data.decode()
            
        # Write file data to temporary file
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_data)
            
        # Perform OCR analysis
        ocr_data = analyze_document_chunked(temp_file_path)
        
        # Extract fields if specified in the document
        fields_to_extract = doc.get("fields_to_extract", ["name","city","address","pincode"])
        extracted_fields = process_ocr_data(ocr_data, fields_to_extract)
        
        # Convert pages to images
        image_data = []
        if temp_file_path.lower().endswith('.pdf'):
            with fitz.open(temp_file_path) as pdf_document:  # Use context manager for PDF
                page_count = len(pdf_document)  # Get page count while document is open
                for page_num in range(page_count):
                    page = pdf_document.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    image_data.append(base64.b64encode(buffered.getvalue()).decode())
                    pix = None  # Clear pixmap reference
        else:
            with Image.open(temp_file_path) as img:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                image_data.append(base64.b64encode(buffered.getvalue()).decode())
                
        update_data = {
            "status": "processed",
            "processed_date": datetime.now(pytz.timezone('UTC')).isoformat(),
            "ocr_output": ocr_data["extracted_data"],
            "extracted_fields": extracted_fields,
            "image_data": image_data
        }

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_data}
        )
        logging.info(f"Successfully processed document: {doc['_id']}")
        
    except Exception as e:
        logging.error(f"Processing failed for document {doc['_id']}: {str(e)}")
        collection.update_one(
            {"_id": doc["_id"]}, 
            {"$set": {"status": "failed", "error": str(e)}}
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                for attempt in range(3):  # Try up to 3 times
                    try:
                        time.sleep(0.5)  # Wait 500ms between attempts
                        os.remove(temp_file_path)
                        break
                    except PermissionError:
                        if attempt == 2:  # Last attempt
                            logging.error(f"Failed to remove temporary file after 3 attempts: {temp_file_path}")
                        continue
            except Exception as e:
                logging.error(f"Error removing temporary file: {str(e)}")
           
def process_documents():
    documents = collection.find(
        {"status": {"$in": ["notprocessed", "failed"]}}).limit(10)
    for doc in documents:
        process_document(doc)
        time.sleep(1)  # Add a small delay between processing documents


@app.on_event("startup")
async def startup_event():
    if not scheduler.running:
        scheduler.add_job(process_documents, 'interval', seconds=30)
        scheduler.start()
        logging.info("Scheduler started")
    else:
        logging.info("Scheduler is already running")


@app.on_event("shutdown")
async def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()
        logging.info("Scheduler shut down")
    else:
        logging.info("Scheduler was not running")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    doc = {
        "filename": file.filename,
        "fileData": base64.b64encode(contents).decode('utf-8'),
        "status": "notprocessed",
        "upload_date": datetime.now(pytz.timezone('UTC')).isoformat()
    }
    result = collection.insert_one(doc)
    return {"message": "File uploaded successfully", "document_id": str(result.inserted_id)}


@app.get("/document/{document_id}")
async def get_document(document_id: str):
    doc = collection.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return JSONResponse(content={
        "id": str(doc["_id"]),
        "status": doc["status"],
        "ocr_output": doc.get("ocr_output"),
        "filename": doc.get("filename")
    })


@app.get("/document-ids")
async def get_document_ids():
    return {"document_ids": [str(doc['_id']) for doc in collection.find({}, {'_id': 1})]}


@app.get("/latest-document-id")
async def get_latest_document_id():
    most_recent_doc = collection.find_one(sort=[('_id', -1)])
    return {"document_id": str(most_recent_doc['_id']) if most_recent_doc else None}


@app.get("/viewer/{document_id}")
async def document_viewer(document_id: str):
    doc = collection.find_one({"_id": ObjectId(document_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    image_data = doc.get("image_data", [])
    ocr_output = doc.get("ocr_output", [])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document Viewer</title>
        <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            
            #image-container {{ width: 100%; overflow-y: auto; max-height: 90vh; position: relative; }}
            #text-container {{ width: 100%; overflow-y: auto; max-height: 90vh; padding-left: 20px; }}
            .page-image {{ width: 100%; margin-bottom: 20px; position: relative; }}
            .highlight-canvas {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
            .selection-canvas {{ position: absolute; top: 0; left: 0; }}
            .page-text {{ margin-bottom: 20px; }}
            .word {{ cursor: pointer; padding: 2px; margin: 2px; display: inline-block; }}
            .word.selected {{ background-color: yellow; }}
            h2 {{ position: sticky; top: 0; background-color: white; padding: 10px 0; }}
            .copy-button {{ 
                position: absolute; 
                background-color: #4CAF50; 
                border: none; 
                color: white; 
                padding: 5px 10px; 
                text-align: center; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 12px; 
                margin: 4px 2px; 
                cursor: pointer; 
                border-radius: 4px; 
                display: none;
            }}


.container {{
        display: flex;
        width: 100%;
        height: 100vh; 
      }}

     
      .tabs-section {{
        flex: 1;
        background-color: #f9f9f9;
        padding: 20px;
        border-right: 1px solid #ccc;
      }}

     

      
      .tab {{
        display: flex;
        cursor: pointer;
        margin-bottom: 10px;
      }}

      .tab button {{
        background-color: #f1f1f1;
        border: none;
        padding: 10px;
        font-size: 16px;
        margin-right: 5px;
        width: 100%;
        text-align: left;
      }}

      .tab button.active {{
        background-color: #ccc;
      }}

      .tab button:hover {{
        background-color: #ddd;
      }}

    
      .tab-content {{
      
        display: none;
        padding: 20px;
        
        
      }}

      .tab-content.active {{
        display: block;
      }}
      
      
      .form-container {{
        flex: 1;
        padding: 20px;
      }}
      
      
       .form-container input {{
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
      }}



      .modal {{
        display: none; 
        position: fixed;
        z-index: 1;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); 
        justify-content: center;
        align-items: center;
      }}

      .modal-content {{
        background-color: white;
        padding: 20px;
        border-radius: 4px;
        width: 50%;
        text-align: center;
      }}

      .close {{
        color: red;
        float: right;
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
      }}

      .close:hover,
      .close:focus {{
        color: black;
        text-decoration: none;
        cursor: pointer;
      }}

        </style>
    </head>
    <body>
        <h1>Document Viewer</h1>
         

       <div class="container">
     
      <div class="tabs-section">
        <div class="tab">
      <button class="tab-link active" onclick="openTab(event, 'Tab1')">
        PDF
      </button>
      <button class="tab-link" onclick="openTab(event, 'Tab2')">OCR</button>
    </div>

 
    <div id="Tab1" class="tab-content active">
       <div  id="image-container"></div>
           
    </div>

    <div id="Tab2" class="tab-content">
      <div  id="text-container"></div>
    </div>
      </div>

     
      <div class="form-container">
        <h3>Fields</h3>
      
          <label for="text1">APN:</label><br />
          <input type="text" id="text1" name="text1" /><br />
          <label for="text2">Name</label><br />
          <input type="text" id="text2" name="text2" /><br />
          <label for="text3">Address</label><br />
          <input type="text" id="text3" name="text3" /><br />

          <label for="text4">Pincode</label><br />
          <input type="text" id="text4" name="text4" /><br />
          
          
      
      </div>
    </div>

           

          
<div id="myModal" class="modal">
      <div class="modal-content">
        
        <h2>Paste Content into Input</h2>
       <div id="modalButtons" class="modal-buttons"></div>
        
        <button onclick="closeModal()">Close Modal</button>
      </div>
    </div>
              
       

        <script>

        let copiedText = null


function openTab(event, tabName) {{
       
        var i, tabContent, tabLinks;
        tabContent = document.getElementsByClassName('tab-content')
        for (i = 0; i < tabContent.length; i++) {{
          tabContent[i].classList.remove('active')
        }}

        
        tabLinks = document.getElementsByClassName('tab-link')
        for (i = 0; i < tabLinks.length; i++) {{
          tabLinks[i].classList.remove('active')
        }}

        
        document.getElementById(tabName).classList.add('active')
        event.currentTarget.classList.add('active')
      }}


function openModal(pageIndex) {{

const pageWords = selectedWords.filter(word => word.pageIndex === pageIndex);
                const selectedText = pageWords
                    .sort((a, b) => a.wordIndex - b.wordIndex)
                    .map(word => word.content)
                    .join(' ');

                    copiedText = selectedText;


        document.getElementById('myModal').style.display = 'flex';
      }}



      const modalButtonsContainer = document.getElementById('modalButtons');
        modalButtonsContainer.innerHTML = '';

        
        const inputs = document.querySelectorAll('.form-container input');

      
        inputs.forEach((input, index) => {{
          const button = document.createElement('button');
          button.innerText = `Paste to ${{input.name}}`;
          button.onclick = function () {{
            fillField(input.id); 
          }};
          modalButtonsContainer.appendChild(button);
          modalButtonsContainer.appendChild(document.createElement('br'));
          modalButtonsContainer.appendChild(document.createElement('br'));
        }});

    
      function closeModal() {{
        document.getElementById('myModal').style.display = 'none';
      }}



      function fillField(inputId) {{
        
        
          const text = copiedText;
          
          document.getElementById(inputId).value = text;
        closeModal();
      }}




        
            const imageContainer = document.getElementById('image-container');
            const textContainer = document.getElementById('text-container');
            const imageData = {json.dumps(image_data)};
            const ocrOutput = {json.dumps(ocr_output)};
            let selectedWords = [];
            let isSelecting = false;
            let startX, startY;
            let isDragging = false;
            let currentSelectionCanvas = null;
            let currentSelectionCtx = null;
            let copyButtons = [];

            function displayDocument() {{
                imageData.forEach((imageBase64, pageIndex) => {{
                    const pageDiv = document.createElement('div');
                    pageDiv.className = 'page-image';
                    pageDiv.id = `page-image-${{pageIndex}}`;

                    const img = document.createElement('img');
                    img.src = `data:image/png;base64,${{imageBase64}}`;
                    img.style.width = '100%';

                    const highlightCanvas = document.createElement('canvas');
                    highlightCanvas.className = 'highlight-canvas';

                    const selectionCanvas = document.createElement('canvas');
                    selectionCanvas.className = 'selection-canvas';

                    const copyButton = document.createElement('button');
                    copyButton.textContent = 'Copy';
                    copyButton.className = 'copy-button';
                    copyButton.style.display = 'none';
                    copyButtons.push(copyButton);

                    pageDiv.appendChild(img);
                    pageDiv.appendChild(highlightCanvas);
                    pageDiv.appendChild(selectionCanvas);
                    pageDiv.appendChild(copyButton);
                    imageContainer.appendChild(pageDiv);

                    const pageTextDiv = document.createElement('div');
                    pageTextDiv.className = 'page-text';

                    const pageHeader = document.createElement('h2');
                    pageHeader.textContent = `Page ${{pageIndex + 1}}`;
                    pageTextDiv.appendChild(pageHeader);

                    if (ocrOutput && ocrOutput[pageIndex] && ocrOutput[pageIndex].words) {{
                        ocrOutput[pageIndex].words.forEach((word, wordIndex) => {{
                            const span = document.createElement('span');
                            span.textContent = word.content + ' ';
                            span.className = 'word';
                            span.dataset.pageIndex = pageIndex;
                            span.dataset.wordIndex = wordIndex;
                            span.addEventListener('click', (e) => toggleWordSelection(e, span, pageIndex, wordIndex));
                            pageTextDiv.appendChild(span);
                        }});
                    }}

                    textContainer.appendChild(pageTextDiv);

                    img.onload = () => {{
                        highlightCanvas.width = img.width;
                        highlightCanvas.height = img.height;
                        selectionCanvas.width = img.width;
                        selectionCanvas.height = img.height;
                        setupImageSelection(selectionCanvas, pageIndex);
                    }};
                }});
            }}

            function toggleWordSelection(event, wordElement, pageIndex, wordIndex) {{
                event.preventDefault();
                const word = ocrOutput[pageIndex].words[wordIndex];
                const selectedIndex = selectedWords.findIndex(w => w.pageIndex === pageIndex && w.wordIndex === wordIndex);

                if (selectedIndex > -1) {{
                    selectedWords.splice(selectedIndex, 1);
                    wordElement.classList.remove('selected');
                }} else {{
                    selectedWords.push({{ ...word, pageIndex, wordIndex }});
                    wordElement.classList.add('selected');
                }}

                highlightWords();
                scrollToWord(pageIndex);
                updateCopyButton();
            }}

            function highlightWords() {{
                const canvases = document.querySelectorAll('.highlight-canvas');
                canvases.forEach((canvas, pageIndex) => {{
                    const ctx = canvas.getContext('2d');
                    const pageWidth = ocrOutput[pageIndex].width;
                    const pageHeight = ocrOutput[pageIndex].height;
                    const scaleX = canvas.width / pageWidth;
                    const scaleY = canvas.height / pageHeight;

                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = 'rgba(255, 255, 0, 0.3)';
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 2;

                    selectedWords.forEach(word => {{
                        if (word.pageIndex === pageIndex) {{
                            ctx.fillRect(word.x * scaleX, word.y * scaleY, word.width * scaleX, word.height * scaleY);
                            ctx.strokeRect(word.x * scaleX, word.y * scaleY, word.width * scaleX, word.height * scaleY);
                        }}
                    }});
                }});
            }}

            function scrollToWord(pageIndex) {{
                const pageImage = document.getElementById(`page-image-${{pageIndex}}`);
                if (pageImage) {{
                    pageImage.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}

            function setupImageSelection(canvas, pageIndex) {{
                const ctx = canvas.getContext('2d');
                let isDrawing = false;
                let startX, startY;

                canvas.addEventListener('mousedown', (e) => {{
                    const rect = canvas.getBoundingClientRect();
                    startX = e.clientX - rect.left;
                    startY = e.clientY - rect.top;
                    isDrawing = true;
                    currentSelectionCanvas = canvas;
                    currentSelectionCtx = ctx;
                }});

                canvas.addEventListener('mousemove', (e) => {{
                    if (!isDrawing) return;
                    const rect = canvas.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    drawSelectionRect(ctx, startX, startY, x - startX, y - startY);
                }});

                canvas.addEventListener('mouseup', (e) => {{
                    if (!isDrawing) return;
                    isDrawing = false;
                    const rect = canvas.getBoundingClientRect();
                    const endX = e.clientX - rect.left;
                    const endY = e.clientY - rect.top;
                    selectWordsInRect(pageIndex, startX, startY, endX, endY);
                }});
            }}

            function drawSelectionRect(ctx, x, y, width, height) {{
                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
                ctx.fillStyle = 'rgba(0, 0, 255, 0.2)';
                ctx.fillRect(x, y, width, height);
            }}

            function selectWordsInRect(pageIndex, startX, startY, endX, endY) {{
                const pageWords = ocrOutput[pageIndex].words;
                const pageWidth = ocrOutput[pageIndex].width;
                const pageHeight = ocrOutput[pageIndex].height;
                const scaleX = currentSelectionCanvas.width / pageWidth;
                const scaleY = currentSelectionCanvas.height / pageHeight;

                const selectionRect = {{
                    left: Math.min(startX, endX) / scaleX,
                    right: Math.max(startX, endX) / scaleX,
                    top: Math.min(startY, endY) / scaleY,
                    bottom: Math.max(startY, endY) / scaleY
                }};

                let selectionChanged = false;

                pageWords.forEach((word, wordIndex) => {{
                    if (isWordInRect(word, selectionRect)) {{
                        const selectedIndex = selectedWords.findIndex(w => w.pageIndex === pageIndex && w.wordIndex === wordIndex);
                        if (selectedIndex === -1) {{
                            selectedWords.push({{ ...word, pageIndex, wordIndex }});
                            const wordElement = document.querySelector(`.word[data-page-index="${{pageIndex}}"][data-word-index="${{wordIndex}}"]`);
                            if (wordElement) {{
                                wordElement.classList.add('selected');
                            }}
                            selectionChanged = true;
                        }} else {{
                            selectedWords.splice(selectedIndex, 1);
                            const wordElement = document.querySelector(`.word[data-page-index="${{pageIndex}}"][data-word-index="${{wordIndex}}"]`);
                            if (wordElement) {{
                                wordElement.classList.remove('selected');
                            }}
                            selectionChanged = true;
                        }}
                    }}
                }});

                if (selectionChanged) {{
                    highlightWords();
                    updateCopyButton();
                }}
            }}

            function isWordInRect(word, rect) {{
                return !(word.x > rect.right ||
                         word.x + word.width < rect.left ||
                         word.y > rect.bottom ||
                         word.y + word.height < rect.top);
            }}

            function updateCopyButton() {{
                copyButtons.forEach((button, pageIndex) => {{
                    const pageWords = selectedWords.filter(word => word.pageIndex === pageIndex);
                    if (pageWords.length > 0) {{
                        const firstWord = pageWords[0];
                        const lastWord = pageWords[pageWords.length - 1];
                        const pageWidth = ocrOutput[pageIndex].width;
                        const pageHeight = ocrOutput[pageIndex].height;
                        const scaleX = currentSelectionCanvas.width / pageWidth;
                        const scaleY = currentSelectionCanvas.height / pageHeight;
                        
                        const left = firstWord.x * scaleX;
                        const top = Math.min(firstWord.y, lastWord.y) * scaleY - 30; // 30px above the selection
                        
                        button.style.left = `${{left}}px`;
                        button.style.top = `${{top}}px`;
                        button.style.display = 'block';
                        button.onclick = () => openModal(pageIndex);
                    }} else {{
                        button.style.display = 'none';
                    }}
                }});
            }}

            function copySelectedText(pageIndex) {{
                const pageWords = selectedWords.filter(word => word.pageIndex === pageIndex);
                const selectedText = pageWords
                    .sort((a, b) => a.wordIndex - b.wordIndex)
                    .map(word => word.content)
                    .join(' ');
                navigator.clipboard.writeText(selectedText).then(() => {{
                    alert('Selected text copied to clipboard!');
                }}).catch(err => {{
                    console.error('Failed to copy text: ', err);
                }});
            }}

            // Initial display
            displayDocument();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/documents", response_class=HTMLResponse)
async def list_documents():
    docs = collection.find({"status": "processed"})
    html_content = """
    <html>
        <body>
            <h1>Processed Documents</h1>
            <ul>
    """
    for doc in docs:
        html_content += f'<li><a href="/viewer/{doc["_id"]}">{doc.get("filename", str(doc["_id"]))}</a></li>'

    html_content += """
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
@app.post("/process-ocr")
async def process_ocr_endpoint(request: OCRRequest):
    """Endpoint to process OCR data for a specific document"""
    try:
        # Retrieve document from MongoDB
        document = collection.find_one({"_id": ObjectId(request.document_id)})
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
            
        if "ocr_output" not in document:
            raise HTTPException(status_code=400, detail="Document has no OCR data")
            
        # Process the OCR data
        results = process_ocr_data(
            ocr_data=document["ocr_output"],
            fields_to_extract=request.fields_to_extract
        )
        
        # Update document with extracted fields
        collection.update_one(
            {"_id": ObjectId(request.document_id)},
            {"$set": {
                "extracted_fields": results,
                "last_extraction": datetime.now(pytz.UTC).isoformat()
            }}
        )
        
        return JSONResponse(content={
            "document_id": request.document_id,
            "extracted_fields": results
        })
        
    except Exception as e:
        logger.error(f"Error processing OCR data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing OCR data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
