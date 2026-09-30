# Anime Dubber (MVP) — JA → EN local AI dubbing

A local, offline AI dubbing pipeline: takes a Japanese video (anime or live-action) + a short voice sample, and produces an English-dubbed version with the original character's voice cloned, background music/SFX preserved, and dialogue timing aligned to the video — no cloud APIs, no subscription costs, running entirely on your system.

# Current architecture (7 stages) : 

<img width="996" height="505" alt="Screenshot 2026-09-23 192015" src="https://github.com/user-attachments/assets/2a7c7297-bb1c-4a0e-a07a-81825140d50d" />


--------------------------------

# ✨ Features currently working : 

1- Voice cloning + translation: Whisper (ASR) → Qwen (translation) → OmniVoice (cloned TTS), fully local

2- Background music/SFX preserved, not replaced — ducked under dialogue, not silenced

3-Duration-guided synthesis: lines paced to fit their original timing slot, not generated freely then stretched (which caused artifacts)

4-Mouth-timing alignment: retimes dialogue to match visible mouth movement (not pixel-level lip-sync — see caveats below)

5-Zero-overlap guarantee: enforced as a hard rule regardless of any upstream timing imprecision

6-Pronunciation controls: explicit language forcing, editable phoneme glossary for recurring mispronounced words

7-Auto-trimmed, sentence-accurate reference clips: no manual clip prep needed

8-Automatic mistranslation detection: catches and retries lines where Japanese leaked into the "English" output

9- --mode {anime,normal}: one flag switches duck level, reference length, accent-forcing, and whether mouth-alignment even runs — since these two content types need genuinely different settings, not just different luck


---------------------------

# Features to be added in the future: 

Multi-speaker support

----------------------------


# 📁 Project Structure : 

<img width="636" height="783" alt="Screenshot 2026-09-30 192316" src="https://github.com/user-attachments/assets/0c2ba38c-5f4a-4a34-86fa-da24ffae58fa" />

-----------------

# How to install it : 

# 1- Install FFmpeg :

download FFmpeg from here : https://ffmpeg.org/download.html and verify with :

```bash
ffmpeg -version  
```
in cmd

# 2- Make sure your NVIDIA driver is installed.

Check:

```bash
nvidia-smi
```

# 3 - Install PyTorch with CUDA

Important: install the CUDA-enabled PyTorch build before installing requirements.txt.

Do not blindly install a CPU-only PyTorch build.

For example, using the appropriate CUDA wheel provided by the PyTorch installation instructions:

```bash
python -m pip install torch torchaudio --index-url <YOUR-PYTORCH-CUDA-WHEEL>
```

Then verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You want:

CUDA: True

The exact PyTorch/CUDA wheel should be selected according to your installed NVIDIA driver and the currently supported PyTorch build.


# 4- Install Python Dependencies

After installing CUDA-enabled PyTorch:

```bash
python -m pip install -r requirements.txt
```

# 5- Install Ollama

The translation and rewriting stages use Ollama as a local model server.

Install Ollama on your system.

Verify:

```bash
ollama --version
```

Then download the model used by this project

```bash
ollama pull qwen2.5:7b
```

Verify that it exists:

```bash
ollama list
```
You should see:

qwen2.5:7b

# 6-Download Whisper

The project expects the original Hugging Face Whisper model at:

models/whisper-large-v3-turbo/

Download from : https://huggingface.co/openai/whisper-large-v3-turbo

Place the downloaded files in that the directory : local_anime_dubber/models/whisper-large-v3-turbo

# 6-1 

Convert Whisper to CTranslate2

--------------------

# 📂Prepare Input Files : 

input/

You need:

input/

├── video.mp4

└── voice_reference.wav

The voice reference should contain the voice you want OmniVoice to clone.

The current CLI accepts the exact paths, so the files do not have to use these exact filenames

# ▶️ Run the Project : 
command for run  : 

if you wanna test a real video use this : 

```bash
python main.py --video input/test.mp4 --voice-ref input/test-a.mp3 --output output/dubbed.mp4 --mode normal
```

if you wanna test a anime :
ه
```bash
python main.py --video input/clip.mp4 --voice-ref input/voice_ref.mp3 --output output/dubbed.mp4 --mode anime
```

The default output is:

output/dubbed_output.mp4

----------------------------

# Example test : 


this is Original video Link :  https://www.youtube.com/watch?v=o4R1-TLkxBs18:24

This is dubbed video(Due to file size limitations, the video length has been shortened, and unfortunately, we cannot include the anime examples either.) : 

https://github.com/user-attachments/assets/965d8e6b-4f0d-493d-955c-237e888ca6c7

<img width="1920" height="1091" alt="Screenshot 2026-09-01 234918" src="https://github.com/user-attachments/assets/265f5efa-5512-4517-b5b0-1b8c9d0bd6c4" />


**Note:** This project is still incomplete and is currently only a prototype with support for a single speaker. It still has many issues with pronunciation, lip-sync, and other aspects.

if you see a problem or have trouble contact me whit this : 

E-mail : dev.bugsbunny2000@gmail.com

X : https://x.com/Ox_XRV0
