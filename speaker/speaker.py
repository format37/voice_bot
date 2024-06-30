from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uvicorn
from datetime import datetime
import pandas as pd
import uuid
from llm import llm_request as llm_request_ext, get_api_key
from tts import google_tts
from concurrent.futures import ThreadPoolExecutor
import pygame
import tempfile
import os

app = FastAPI()

# Global variables
request_queue = pd.DataFrame(columns=['id', 'content', 'timestamp'])
llm_responses = pd.DataFrame(columns=['id', 'answer'])
current_speech_task = None
thread_pool = ThreadPoolExecutor(max_workers=1)

# LLM configuration
system_content = "Вы - голосовой робот. Вы помогаете тестировать разработку голосовых роботов. Общайтесь на любые темы, задавайте вопросы, делитесь своими мыслями."

llm_messages = [
    {"role": "system", "content": system_content}
]

class Request(BaseModel):
    content: str

class LLMResponse(BaseModel):
    id: str
    answer: str

async def llm_request(text):
    print(f"LLM request: {text}")
    global llm_messages

    llm_messages.append({"role": "user", "content": text})
    engine = "openai"
    model = "gpt-4o"

    api_key = get_api_key(engine)
    PROJECT = ""
    LOCATION = ""
    text = await llm_request_ext(engine, model, api_key, llm_messages, PROJECT, LOCATION)
    llm_messages.append({"role": "assistant", "content": text})
    return text

async def answer_generator(unique_id):
    print(">> answer_generator")
    queue = request_queue.to_dict(orient='records')
    current_time = datetime.now()
    if len(queue) > 0:
        last_item = queue[-1]
        last_item_time = last_item['timestamp']
        time_difference = current_time - last_item_time
        print(f"Last item {time_difference.total_seconds():.2f} seconds ago) : {last_item['content']} (at {last_item['timestamp']}) id: {last_item['id']}")
        
        print(f"Calling llm")
        all_items = "\n".join([item['content'] for item in queue])
        llm_response = await llm_request(all_items)
        print(f"LLM response: {llm_response}")
        print(f"calling the speech synthesis")
        asyncio.create_task(speech_synthesis(last_item['id'], llm_response))
    else:
        print(f"No items in the queue")
    print("<< answer_generator")

async def process_request(request: Request):
    current_date = datetime.now()
    print(f"[{current_date}] >> process_request: {request.content}")
    global request_queue
    
    unique_id = str(uuid.uuid4())
    new_row = pd.DataFrame({
        'id': [unique_id],
        'content': [request.content],
        'timestamp': [current_date]
    })
    request_queue = pd.concat([request_queue, new_row], ignore_index=True)
    print(f"Processed and queued request: {request.content}")

    print(f"Calling answer_generator [{unique_id}]")
    await answer_generator(unique_id)

def play_audio(audio_file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

async def speech_synthesis(id, text):
    global current_speech_task
    current_time = datetime.now()
    print(f"[{current_time}] >> Speech synthesis start: {text} [{id}]")

    speed = 1.4
    model = "ru-RU-Wavenet-A"
    language = "ru-RU"

    try:
        audio_content, time_spent = await google_tts(text, model, language, speed)
        print(f"Time spent on synthesis: {time_spent} seconds")
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_content)
            temp_audio_file_path = temp_audio_file.name

        current_speech_task = asyncio.get_event_loop().run_in_executor(thread_pool, play_audio, temp_audio_file_path)
        await current_speech_task
    except asyncio.CancelledError:
        print(f"[{datetime.now()}] Speech synthesis interrupted: {id}")
    finally:
        current_speech_task = None
        # Clean up the temporary file
        if 'temp_audio_file_path' in locals():
            os.unlink(temp_audio_file_path)

    current_time = datetime.now()
    print(f"[{current_time}] >> Speech synthesis end: {text} [{id}]")

@app.post("/submit")
async def submit_request(request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_request, request)
    return {"message": "Request received and processing started"}

@app.post("/queue/clean")
async def clean_queue():
    global request_queue
    request_queue = pd.DataFrame(columns=['id', 'content', 'timestamp'])
    return {"message": "Queue has been cleaned. All records removed."}

@app.post("/interrupt")
async def interrupt_speech():
    global current_speech_task
    if current_speech_task and not current_speech_task.done():
        pygame.mixer.music.stop()
        current_speech_task.cancel()
        await asyncio.sleep(0.1)  # Give a short time for the task to be cancelled
        return {"message": "Speech synthesis interrupted"}
    else:
        return {"message": "No active speech synthesis to interrupt"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    