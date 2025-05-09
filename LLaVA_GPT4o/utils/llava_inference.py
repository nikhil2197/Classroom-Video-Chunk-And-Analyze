# llava_inference.py
# Clean, minimal version – works with LLaVA‑1.5‑7B HF.

import os, json, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import ANALYSIS_PROMPTS, FPS

# ------------------------------------------------------------------
# 1.  Environment & model
# ------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16
).to(device).eval()

# ------------------------------------------------------------------
# 2.  Helpers
# ------------------------------------------------------------------
def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

@torch.no_grad()
def infer_frame(img_path: str, prompt: str) -> str | None:
    """
    Run LLaVA on a single image file and return the decoded caption,
    or None if the image is unreadable / inference fails.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as err:
        print(f"❌  Corrupted image skipped: {img_path} ({err})")
        return None

    batch = processor(images=img, text=prompt, return_tensors="pt").to(device)

    try:
        out = model.generate(**batch, max_new_tokens=90)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0]
        return caption
    except Exception as err:
        print(f"❌  Inference failed on {os.path.basename(img_path)}: {err}")
        return None
    finally:
        clear_gpu()

# ------------------------------------------------------------------
# 3.  Loop over frame folder
# ------------------------------------------------------------------
def run_llava_on_frames(frame_dir: str, output_dir: str) -> str:
    """                                                                                                                                                                         
    Run LLaVA inference with multiple prompts based on time windows.                                                                                                            
    Outputs a JSON mapping each frame to a dict containing:                                                                                                                     
      - time_s: time in seconds                                                                                                                                                 
      - time_min: time in minutes                                                                                                                                               
      - <prompt_name>: caption string for each applicable prompt                                                                                                                
    """                                                                                                                                                                         
    os.makedirs(output_dir, exist_ok=True)                                                                                                                                      
    results = {}                                                                                                                                                                
                                                                                                                                                                                
    for fname in sorted(os.listdir(frame_dir)):                                                                                                                                 
        if not fname.lower().endswith(".jpg"):                                                                                                                                  
            continue                                                                                                                                                            
        fpath = os.path.join(frame_dir, fname)                                                                                                                                  
                                                                                                                                                                                
        # Parse frame index from filename, e.g. 'frame_0012.jpg' -> 12                                                                                                          
        try:                                                                                                                                                                    
            base = os.path.splitext(fname)[0]                                                                                                                                   
            idx = int(base.split('_')[-1])                                                                                                                                      
        except Exception:                                                                                                                                                       
            print(f"⚠️  Cannot parse frame index from '{fname}', skipping.")                                                                                                    
            continue                                                                                                                                                            
                                                                                                                                                                                
        # Compute time in seconds and minutes                                                                                                                                   
        time_s = idx / FPS                                                                                                                                                      
        time_min = time_s / 60.0                                                                                                                                                
                                                                                                                                                                                
        entry: dict[str, object] = {                                                                                                                                            
            "time_s": time_s,                                                                                                                                                   
            "time_min": time_min,                                                                                                                                               
        }                                                                                                                                                                       
                                                                                                                                                                                
        # Determine which prompts apply for this frame                                                                                                                          
        applicable = []                                                                                                                                                         
        for p in ANALYSIS_PROMPTS:                                                                                                                                              
            start = p.get("start_min", 0)                                                                                                                                       
            end = p.get("end_min", None)                                                                                                                                        
            if time_min >= start and (end is None or time_min < end):                                                                                                           
                applicable.append(p)                                                                                                                                            
                                                                                                                                                                                
        if not applicable:                                                                                                                                                      
            print(f"🖼️  No prompts for {fname} at {time_s:.1f}s.")                                                                                                             
            results[fname] = entry                                                                                                                                              
            continue                                                                                                                                                            
                                                                                                                                                                                
        names = [p["name"] for p in applicable]                                                                                                                                 
        print(f"🖼️  Inferring {fname} at {time_s:.1f}s for prompts: {names}")                                                                                                  
                                                                                                                                                                                
        # Run inference for each prompt
        for p in applicable:
            cap = infer_frame(fpath, p["prompt"])
            if not cap:
                continue
            # Extract only the model's response, stripping any echoed prompt
            if "\n\n" in cap:
                response = cap.split("\n\n", 1)[1].strip()
            else:
                response = cap.strip()
            entry[p["name"]] = response
                                                                                                                                                                                
        results[fname] = entry                                                                                                                                                  
                                                                                                                                                                                
    # Save all results as JSON                                                                                                                                                  
    out_json = os.path.join(output_dir, "llava_responses.json")                                                                                                                 
    with open(out_json, "w") as f:                                                                                                                                              
        json.dump(results, f, indent=2)                                                                                                                                         
    print(f"✅  Inference completed. Captions saved to {out_json}")                                                                                                              
    return out_json  
