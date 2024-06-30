# voice_bot
A bot that can support interruptable voice conversation using Microphone, Google stt, LLM and Google tts.
### Installation
1. Install requirements
```
cd speaker
python3 -m pip install -r requirements.txt
cd ../listener
python3 -m pip install -r requirements.txt
```
2. Download and put to ./speaker/ folder the Google app credentials json file.  
3. Set the openai api key:
```
export OPENAI_API_KEY=your_openai_api_key
```
### Using
1. Run the speaker server:
```
cd ../speaker
python3 speaker.py
```
2. Run the listener script:
```
cd ../listener
python3 listener.py
```
It would be better to use headphones instead of speakers to avoid feedback.