from pathlib import Path
from openai import OpenAI
import os
import time
import random
from playsound import playsound
import asyncio
from google.cloud import texttospeech
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

async def speech_synthesis(engine, text, model, voice, speed=1.0):
    if engine == "openai":
        # Speed from 0.25 to 4.0
        # Set your API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set. You need to set the OPENAI_API_KEY environment variable this way:\nexport OPENAI_API_KEY=yourkey")

        client = OpenAI()
        # https://platform.openai.com/docs/guides/text-to-speech
        speech_file_path = Path(__file__).parent / "speech.mp3"
        
        time_start = time.time()
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            speed=speed,
            input=text,
        )
        response.stream_to_file(speech_file_path)
        time_end = time.time()
        print(f"Time elapsed: {time_end - time_start} seconds")
    
    elif engine == "google":
        # https://cloud.google.com/text-to-speech/docs/voices
        # https://cloud.google.com/text-to-speech
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'google.json'
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=model,
            name=voice)

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speed
        )

        time_start = time.time()
        response = client.synthesize_speech(input = synthesis_input, voice = voice, audio_config = audio_config)
        time_end = time.time()

        # Save response as audio file
        speech_file_path = Path(__file__).parent / "speech.wav"
        with open(speech_file_path, "wb") as f:
            f.write(response.audio_content)
        print(f"Time spent: {time_end - time_start} seconds")

    # Play audio
    playsound(str(speech_file_path))
    print(f'{engine} Speech synthesis done!')
    return time_end - time_start

async def google_tts(text, model = 'en-US-Neural2-F', language='en-US', speed=1):

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'google.json'
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language,
        name=model)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speed
    )

    time_start = time.time()
    response = client.synthesize_speech(input = synthesis_input, voice = voice, audio_config = audio_config)
    time_end = time.time()

    return response.audio_content, time_end - time_start

async def test_openai():
    voices = [
    {
        "id": "alloy",
        "male": False,
        "name": "Елена"
    },
    {
        "id": "echo",
        "male": True,
        "name": "Иван"
    },
    {
        "id": "fable",
        "male": False,
        "name": "Татьяна"
    },
    {
        "id": "onyx",
        "male": True,
        "name": "Павел"
    },
    {
        "id": "nova",
        "male": False,
        "name": "Ольга"
    },
    {
        "id": "shimmer",
        "male": False,
        "name": "Ирина"
    }
    ]

    # Choose random voice
    voice = random.choice(voices)

    text = f"Вас приветствует сервисная служба Айсберг. Оператор Татьяна. Как я могу вам помочь?"

    model = 'tts-1' # models (tts-1, tts-1-hd)
    voice_id = "shimmer"
    print(f"model: {model}, voice: {voice_id}, name: {voice['name']}")
    speed = 1.1
    
    time_spent = await speech_synthesis(text, model, voice_id, speed)
    print(f"OpenAI Time spent: {time_spent} seconds")
    return time_spent

async def test_google():
    # https://cloud.google.com/text-to-speech/docs/voices
    # https://cloud.google.com/text-to-speech
    data = {
        'text':"Вас приветствует сервисная служба Айсберг. Оператор Татьяна. Как я могу вам помочь?",
        'language':'ru-RU',
        'model':'ru-RU-Wavenet-A',
        'speed':1.1
    }
    response, time_spent = await google_tts(data['text'], data['model'], data['language'], data['speed'])
    # Save response as audio file
    with open("audio.wav", "wb") as f:
        f.write(response)
    print(f"Time spent: {time_spent} seconds")
    # Play audio
    playsound("audio.wav")
    print('Google speech synthesis done!')
    return time_spent

async def run_tests(num_tests=10):
    openai_times = []
    google_times = []

    for _ in range(num_tests):
        # Test OpenAI
        openai_time = await test_openai()
        openai_times.append(openai_time)

        # Test Google
        google_time = await test_google()
        google_times.append(google_time)

    return openai_times, google_times

def plot_comparison(openai_times, google_times):
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'OpenAI': openai_times,
        'Google': google_times
    })
    df['Test Case'] = range(1, len(openai_times) + 1)
    df_melted = df.melt(id_vars=['Test Case'], var_name='Model', value_name='Time')
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Create the main plot
    ax = sns.scatterplot(data=df_melted, x='Test Case', y='Time', hue='Model', style='Model', s=100)
    
    # Add median lines
    for model in ['OpenAI', 'Google']:
        median = np.median(df[model])
        ax.axhline(median, ls='--', color=sns.color_palette()[0 if model == 'OpenAI' else 1], 
                   label=f'{model} Median')
    
    # Fill the area between medians
    openai_median = np.median(df['OpenAI'])
    google_median = np.median(df['Google'])
    ax.fill_between(ax.get_xlim(), openai_median, google_median, alpha=0.2, 
                    color='gray', label='Median Difference')
    
    # Customize the plot
    plt.title('Comparison of OpenAI TTS vs Google TTS')
    plt.xlabel('Test Case')
    plt.ylabel('Time (seconds)')
    plt.legend(title='', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('tts_comparison_plot.png')
    plt.close()

    # Print statistical summary
    print("\nStatistical Summary:")
    print(f"OpenAI - Median: {openai_median:.2f}s, Mean: {np.mean(openai_times):.2f}s")
    print(f"Google - Median: {google_median:.2f}s, Mean: {np.mean(google_times):.2f}s")

    if openai_median < google_median:
        print(f"OpenAI is {google_median/openai_median:.2f} times faster than Google (based on median)")
    else:
        print(f"Google is {openai_median/google_median:.2f} times faster than OpenAI (based on median)")

async def main():
    num_tests = 10  # You can adjust this number
    openai_times, google_times = await run_tests(num_tests)
    plot_comparison(openai_times, google_times)

if __name__ == "__main__":
    asyncio.run(main())
