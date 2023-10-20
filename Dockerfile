# Include Python
FROM python:3.11.1-buster

# Define your working directory
WORKDIR /app

# Install runpod
RUN pip install runpod

# Add your file
ADD src/serverless.py .
ADD src/llama.py .

# Call your file when your container starts
CMD [ "python", "-u", "serverless.py" ]