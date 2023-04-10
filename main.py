import io

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ml import obtain_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

"""
@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
"""

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "message": "Hello, World!"}

class Item(BaseModel):
    name: str
    price: float
    tags: list[str] = []

@app.post("/items/")
def create_item(item: Item):
    return item

"""
@app.get("/generate")
def generate_image(prompt: str):
    return {"prompt": prompt}
"""

# It works nicely, generating the same image we got when playing with 'ml.py'.    
"""
@app.get("/generate")
def generate_image(prompt: str):
    image = obtain_image(prompt, num_inference_steps = 25, seed = 1024)
    image.save("image.png")
    return FileResponse("image.png")
"""

# We use the original code instead of 'seed: Union[int, None] = None' since this Python version (3.10.10)
    # supports it.
@app.get("/generate")
def generate_image(
    prompt: str,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(prompt, num_inference_steps = num_inference_steps, seed = seed,
                         guidance_scale = guidance_scale)
    image.save("image.png")
    return FileResponse("image.png")

# This function avoids saving the image in disk and prevents overwriting images in case of simultaneous
    # requests.
"""
@app.get("/generate-memory")
def generate_image_memory(
    prompt: str,
    *,
    seed: int | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    image = obtain_image(prompt, num_inference_steps = num_inference_steps, seed = seed,
                         guidance_scale = guidance_scale)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
"""
