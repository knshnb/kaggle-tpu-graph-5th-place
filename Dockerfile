FROM nvcr.io/nvidia/pytorch:23.06-py3

COPY requirements.txt .
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt
