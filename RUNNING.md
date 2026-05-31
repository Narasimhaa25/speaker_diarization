Running the project in a local virtual environment
===============================================

Prerequisites
- Windows PowerShell (or compatible shell)
- Python 3.10 (or 3.9+)

1) Activate the virtual environment

PowerShell:

```powershell
# From project root
.\venv\Scripts\Activate
```

2) Install dependencies (if not already done)

```powershell
pip install -r speaker_diarization\requirements.txt
pip install datasets==2.19.0 transformers soundfile
```

3) Configure HuggingFace token (optional, required for gated pyannote model)

- Create a token at: https://huggingface.co/settings/tokens (give `Read` scope)
- Visit and accept terms for the model: https://huggingface.co/pyannote/speaker-diarization-3.1
- Copy token into a local `.env` file (do NOT commit to VCS):

```powershell
copy .env.example .env
# Edit .env and paste your token after HF_TOKEN=
```

Or set HF_TOKEN in the PowerShell session for the current run:

```powershell
$env:HF_TOKEN = 'hf_your_token_here'
```

4) Run the quick demo

```powershell
python demo.py
```

5) Run the comprehensive demo

```powershell
python run_demo.py
```

6) Run tests

```powershell
python -m pytest speaker_diarization\tests/ -v
```

Notes
- If you do NOT provide `HF_TOKEN` or do not accept the model terms, the code will fall back to the local ONNX or energy-based diarization paths.
- Keep your `.env` out of source control. Consider adding `.env` to `.gitignore`.
