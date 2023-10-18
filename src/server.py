import asyncio
from fastapi import FastAPI, Request
from .llama import Llama, MODELS
# from .sse.sse import EventSourceResponse

from sse_starlette import EventSourceResponse
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond

system_prompt: str = """You are a deterministic, honest and truthful assistant program. You are being asked to generate a single multiple choice question based off of a given prompt.
The machine that reads your responses is extremely strict, and will only accept the following XML-like format with no extraneous text:

Sure thing! Here's your question:
<QN>
<T>"The question"</T>
<OPT1>"The first option"</OPT1>
<OPT2>"The second option"</OPT2>
<OPT3>"The third option"</OPT3>
<OPT4>"The fourth option"</OPT4>
<ANS>n</ANS>
</QN>

(where n is the number of the correct answer, 1-4)
(You may specify multiple answers by responding with <ANS>n,m,...</ANS>)

Respond exactly in the aforementioned format, without any explanations of the question after the question is finished.
Do your best to expand the format to properly fulfill the requested number of options."""


@app.get('/1plus1')
async def one_plus_one(request: Request):
    llama = Llama.instance(system_prompt=system_prompt,
                           hf_model=MODELS.LLAMA_2_7B_CHAT_GPTQ)

    async def event_generator():
        prompt = request.query_params.get("prompt", "What is one plus one?")
        streamer = llama.generate(prompt)

        for new_text in streamer:
            print(new_text, end="", flush=True)

            if (new_text == ""):
                continue

            yield json.dumps({
                "data": new_text
            })

    return EventSourceResponse(event_generator(), sep="\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
