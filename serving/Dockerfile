# MagpieTTS-Server für DGX Spark (GB10, aarch64).
# Basis: NGC-NeMo-Container (multi-arch, enthält nemo_toolkit inkl. TTS).
FROM nvcr.io/nvidia/nemo:26.06

# NeMo liegt im Container nur als Source unter /opt/NeMo – editierbar mit
# tts-Extra installieren, damit lightning & Co. in der venv landen.
RUN pip install --no-cache-dir -e "/opt/NeMo[tts]" \
    && pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" kaldialign

COPY server.py /opt/magpie/server.py

EXPOSE 8001
ENTRYPOINT ["python3", "/opt/magpie/server.py"]
