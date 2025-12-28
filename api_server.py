"""
FastAPI server for Liquid AI LFM2-1.2B (CPU-optimized)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Liquid AI LFM2-1.2B API",
    description="CPU-optimized API for Liquid AI LFM2-1.2B model",
    version="1.0.1"
)

# Global model and tokenizer
model = None
tokenizer = None
MODEL_LOADED = False

# Request models
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.3
    min_p: float = 0.15
    repetition_penalty: float = 1.05

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.3
    min_p: float = 0.15
    repetition_penalty: float = 1.05


def check_model_loaded():
    """Verify model is loaded, raise exception if not"""
    if not MODEL_LOADED or model is None or tokenizer is None:
        logger.error("Model not loaded - rejecting request")
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Server may still be initializing."
        )


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, MODEL_LOADED
    
    logger.info("="*80)
    logger.info("Loading Liquid AI LFM2-1.2B model...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CPU threads: OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS', 'not set')}")
    logger.info("="*80)
    
    # Check if local model exists
    local_model_path = "/models/LFM2-1.2B"
    if os.path.exists(local_model_path):
        model_path = local_model_path
        logger.info(f"✅ Found local model: {model_path}")
    else:
        model_path = "LiquidAI/LFM2-1.2B"
        logger.info(f"📥 Will download from HuggingFace: {model_path}")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("✅ Tokenizer loaded")
        
        # Load model with bfloat16 for efficiency (even on CPU)
        logger.info("Loading model (this may take 1-2 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 as per docs
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        # Set to eval mode
        model.eval()
        
        MODEL_LOADED = True
        logger.info("="*80)
        logger.info("✅ Model loaded successfully on CPU")
        logger.info(f"   Model parameters: ~1.2B")
        logger.info(f"   Precision: bfloat16")
        logger.info(f"   Device: CPU")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ FATAL: Failed to load model: {e}")
        logger.error("="*80)
        MODEL_LOADED = False
        # Don't raise - let server start so health check can report status
        # But log prominently


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok" if MODEL_LOADED else "model_not_loaded",
        "model": "LiquidAI/LFM2-1.2B",
        "device": "cpu",
        "precision": "bfloat16",
        "message": "Liquid AI API is running" if MODEL_LOADED else "Server running but model failed to load"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "tokenizer_loaded": tokenizer is not None,
        "device": "cpu",
        "torch_version": torch.__version__,
        "threads": os.getenv('OMP_NUM_THREADS', 'not set')
    }


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Generate completion for a prompt"""
    check_model_loaded()
    
    try:
        logger.info(f"Processing completion request: '{request.prompt[:50]}...'")
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                min_p=request.min_p,  # Added min_p
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens (excluding input)
        response_text = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        ).strip()
        
        logger.info(f"Generated {len(response_text)} characters")
        
        return {
            "id": f"cmpl-{abs(hash(request.prompt)) % 10**16}",
            "object": "text_completion",
            "created": 1234567890,
            "model": "LiquidAI/LFM2-1.2B",
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": outputs.shape[1] - input_length,
                "total_tokens": outputs.shape[1]
            }
        }
        
    except Exception as e:
        logger.error(f"Error during completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Generate chat completion"""
    check_model_loaded()
    
    try:
        logger.info(f"Processing chat request with {len(request.messages)} messages")
        
        # Convert messages to chat format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        input_length = input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                min_p=request.min_p,  # Added min_p
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode full output
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract assistant response (after the last <|im_start|>assistant)
        if "<|im_start|>assistant" in full_text:
            response_text = full_text.split("<|im_start|>assistant")[-1]
            # Clean up special tokens
            response_text = (response_text
                           .replace("<|im_end|>", "")
                           .replace("<|endoftext|>", "")
                           .strip())
        else:
            # Fallback: decode without special tokens
            response_text = tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            ).strip()
        
        logger.info(f"Generated {len(response_text)} characters")
        
        return {
            "id": f"chatcmpl-{abs(hash(str(messages))) % 10**16}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "LiquidAI/LFM2-1.2B",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": input_length,
                "completion_tokens": outputs.shape[1] - input_length,
                "total_tokens": outputs.shape[1]
            }
        }
        
    except Exception as e:
        logger.error(f"Error during chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")