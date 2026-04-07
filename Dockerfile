FROM python:3.9-slim

# Install audio tools the AI needs
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg libgl1

WORKDIR /app

# Install your Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly tell Docker to make sure the templates folder is in the right spot
COPY templates/ /app/templates/

# Copy all your project files
COPY . .

# Start the app
CMD ["python", "app.py"]
