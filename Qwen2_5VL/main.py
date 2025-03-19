import os
import base64
import io
import time
import uuid
from PIL import Image
import torch
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

app = FastAPI(title="Qwen2.5-VL API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None

def load_model(model_path, use_flash_attention=True):
    global model, processor
    
    print(f"Loading model from {model_path}...")
    
    if use_flash_attention and torch.cuda.is_available():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
    
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded successfully!")

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    if base64_string.startswith("data:image"):
        # Remove the data URL prefix if present
        base64_string = base64_string.split(",")[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Define OpenAI API compatible models
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[Union[str, ImageUrl]] = None

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[str, ContentItem]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# Keep the original API for backward compatibility
class GenerateRequest(BaseModel):
    prompt: str = ""
    images: List[str] = []
    max_tokens: int = 4096
    temperature: float = 0.0
    seed: Optional[int] = 2024

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class GenerateResponse(BaseModel):
    text: str
    usage: TokenUsage

# Models info endpoint
class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]

@app.get("/v1/models")
async def list_models():
    """List available models"""
    model_id = "Qwen2.5-VL"
    return ModelsResponse(
        data=[
            ModelData(
                id=model_id,
                created=int(time.time()),
                owned_by="Alibaba Cloud"
            )
        ]
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions API"""
    try:
        messages = request.messages
        max_tokens = request.max_tokens or 4096
        temperature = request.temperature or 0.0
        seed = request.seed
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Process messages to Qwen2.5-VL format
        qwen_messages = []
        
        for msg in messages:
            qwen_msg = {"role": msg.role}
            
            if isinstance(msg.content, str):
                qwen_msg["content"] = [{"type": "text", "text": msg.content}]
            else:
                content_list = []
                for item in msg.content:
                    if isinstance(item, str):
                        content_list.append({"type": "text", "text": item})
                    elif item.type == "text":
                        content_list.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        image_url = item.image_url
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url", "")
                        
                        # Handle base64 images
                        if image_url.url.startswith("data:image"):
                            image = base64_to_image(image_url.url)
                            content_list.append({"type": "image", "image": image})
                        else:
                            # URL image
                            content_list.append({"type": "image", "image": image_url.url})
                
                qwen_msg["content"] = content_list
            
            qwen_messages.append(qwen_msg)
        
        # Prepare for inference
        text = processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
            
        # Process output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Calculate token usage (approximate)
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(generated_ids_trimmed[0])
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{str(uuid.uuid4())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=output_text
                    ),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL API Server")
    parser.add_argument("--model_path", type=str, default="/data1/model_zoo/Qwen2.5-VL-7B-Instruct", 
                        help="Path to the model or model name on Hugging Face")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--no_flash_attention", action="store_true", 
                        help="Disable flash attention (use if you encounter issues)")
    
    args = parser.parse_args()
    
    # Load the model
    load_model(args.model_path, not args.no_flash_attention)
    
    # Run the server with uvicorn
    print(f"Starting Qwen2.5-VL API server at http://{args.host}:{args.port}")
    print(f"OpenAI API compatible endpoint available at http://{args.host}:{args.port}/v1/chat/completions")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
