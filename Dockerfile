FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /opt/banking-lab
COPY requirements-gpu.txt /opt/banking-lab/
RUN python -m pip install --no-cache-dir -r requirements-gpu.txt
COPY banking_lab /opt/banking-lab/banking_lab

# OpenShift assigns an arbitrary UID at runtime. All generated files go to /work.
ENV PYTHONPATH=/opt/banking-lab \
    HOME=/tmp \
    HF_HOME=/work/hf-cache \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1
