# Llama-2 Streaming API

> A HuggingFace Transformers streaming response API written in python using FastAPI.

Uses NPM for scripting and pip for python dependencies.

# !! This application DOES NOT use standard SSE !!

Streaming routes return [JSONL](https://jsonlines.org/) instead of standard `data: ` SSE. This is because I think the standard SSE format is silly and JSONL is much easier to parse.

## Requirements

- Python 3.10.x || 3.11.x (other versions may work but are not tested)
- pip requirements
  - fastapi
  - uvicorn
  - gunicorn
  - transformers
  - torch
- gptq requirements (pip)
  - auto-gptq
  - optimum
- Hardware
  - Nvidia GPU with minimum 8GB vram (12+ recommended for running 13B+ param models)
  - 16GB+ System RAM
  - A reasonable amount of free storage for the model weights (13B GPTQ Models require around 8GB of space each)

## Installation

You can install torch with either the `npm` script or the [torch install website](https://pytorch.org/get-started/locally/)

```bash
git clone https://github.com/alexng353/llama2

npm run venv:setup
npm run venv:activate

npm run setup
npm run pip:install-rquired
npm run pip:install-gptq

# install torch
# platform: linux, macos, windows
# version: default, cu121, cu118 (macos only supports default. windows and linux use cu121 as default)
npm run torch:install-{platform}-{version}
```

## Usage

`npm run start`

## Development

`npm run dev`

### Routes

- `/chat` - Give it a single prompt with the `?prompt=` query parameter and it will return a stream of responses.

## Modifying Settings

> This application comes with reasonable defaults for running on a local machine.

`./src/llama.py` includes a class called `MODELS` which contains a default list of models that I think would be most popular. To change the model, go `line 13` of `./src/server.py` and use either a model from the `MODELS` class or a any huggingface model name.

> Any model from [TheBloke's GPTQ Quantisations](https://huggingface.co/TheBloke?search_models=gptq) will work, non-gptq models are untested but may work with modifications to the code.

```python file=src/server.py
# ...

MODEL = MODELS.LLAMA_2_13B_CHAT_GPTQ # <- change this

# ...
```

## Attributions

- Uses modified code from [sysid/sse-starlette](https://github.com/sysid/sse-starlette) under the [BSD 3-Clause "New" or "Revised" License](./attributions/sse_starlette.md)
