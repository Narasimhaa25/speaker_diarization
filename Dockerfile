FROM python:3.9-slim

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY speaker_diarization/requirements.txt requirements.txt
RUN pip install --no-cache-dir \
    numpy scipy librosa soundfile \
    onnxruntime \
    cryptography \
    speechbrain \
    pyannote.audio \
    scikit-learn \
    tqdm pyyaml \
    && pip install --no-cache-dir transformers

# Copy source code
COPY speaker_diarization/ ./speaker_diarization/
COPY demo.py ./demo.py

# Copy pre-built assets (ONNX model + staff DB + audio)
COPY speaker_diarization/models/ecapa_tdnn_int8.onnx ./speaker_diarization/models/ecapa_tdnn_int8.onnx
COPY ami_staff.staffdb ./ami_staff.staffdb
COPY ami_staff.key ./ami_staff.key

# Fix SpeechBrain/PyTorch compat patches
RUN python -c "
import re, pathlib

# Patch 1: lightning_fabric weights_only
p = pathlib.Path('/usr/local/lib/python3.9/site-packages/lightning_fabric/utilities/cloud_io.py')
if p.exists():
    t = p.read_text()
    t = re.sub(r'weights_only=weights_only,\n        \)', 'weights_only=False,\n        )', t)
    p.write_text(t)

# Patch 2: speechbrain k2 optional
p2 = pathlib.Path('/usr/local/lib/python3.9/site-packages/speechbrain/integrations/k2_fsa/__init__.py')
if p2.exists():
    t2 = p2.read_text()
    t2 = t2.replace('raise ImportError(MSG) from e', 'pass  # k2 optional')
    p2.write_text(t2)

# Patch 3: speechbrain linecache fix
p3 = pathlib.Path('/usr/local/lib/python3.9/site-packages/speechbrain/utils/importutils.py')
if p3.exists():
    t3 = p3.read_text()
    t3 = t3.replace(
        'importer_frame.filename.endswith(\"/inspect.py\")',
        'importer_frame.filename.endswith(\"/inspect.py\") or importer_frame.filename.endswith(\"/linecache.py\")'
    )
    p3.write_text(t3)
print('Patches applied')
"

CMD ["python", "demo.py"]
