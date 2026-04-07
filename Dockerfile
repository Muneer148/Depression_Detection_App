FROM python:3.9-slim

# Install audio tools the AI needs
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

WORKDIR /app

# Install your Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files
COPY . .

# Start the app
CMD ["python", "app.py"]
