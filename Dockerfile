# Use a minimal base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that the Flask app runs on
EXPOSE 8000

# Define the command to run the app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app", "--workers=16", "--threads=2"]
