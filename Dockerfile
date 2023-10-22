# Include Python
FROM python:3.11.1-buster

# Define your working directory
WORKDIR /app

# Install runpod
RUN pip install runpod icecream transformers torch torchvision torchaudio accelerate optimum auto-gptq

# Add your file
ADD src/__init__.py ./src/
ADD src/serverless.py ./src/
ADD src/llama.py ./src/

# ADD ./models ./models

# ADD ./test_input.json ./test_input.json


# Call your file when your container starts
CMD [ "python", "-u", "./src/serverless.py" ]