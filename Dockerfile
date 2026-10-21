# Small official Python image
FROM python:3.12-slim

WORKDIR /app

# CPU-onlu PyTorch: much smaller than the default GPU build
RUN pip install --no-cache-dir torchvision --index-url https: //download.pytorch.org/whl/cpu

# Other dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#only the files the api need at runtime

COPY model.py predict.py api.py threshold.txt scanguard_model.pth ./

EXPOSE 8000
CMP["unicorn", "api:app", "---host", "0.0.0.0", "--port", "8000"]