# Llama-2 Streaming API

> A HuggingFace Transformers streaming response API written in python using FastAPI.

Uses NPM for scripting and pip for python dependencies.

## Requirements

- Python 3.10.x || 3.11.x

- pip requirements

  - fastapi
  - uvicorn
  - gunicorn
  - transformers
  - torch

- gptq requirements (pip)
  - auto-gptq
  - optimum

## Installation

```bash
npm run pip:install-rquired

# gptq
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

## Attributions

- Uses modified code from [sysid/sse-starlette](https://github.com/sysid/sse-starlette) under the [BSD 3-Clause "New" or "Revised" License](./attributions/sse_starlette.md)
