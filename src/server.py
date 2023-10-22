from fastapi import FastAPI, Request, Response
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from .llama import Llama, MODELS
from .sse.sse import EventSourceResponse

app = FastAPI()

with open("system.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

MODEL = MODELS.LLAMA_2_13B_CHAT_GPTQ
STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 15000  # milisecond


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/chat')
async def chat(request: Request):
    llama = Llama.instance(
        system_prompt=SYSTEM_PROMPT,
        hf_model=MODEL
    )

    prompt = request.query_params.get("prompt")

    if prompt is None:
        return Response("Please provide a prompt", status_code=400)

    chat = [
        # {"role": "system", "content": f"<<SYS>>{system_prompt}<</SYS>>"},
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello there, young mathematician! I'm Cosmo, your friendly and intelligent assistant here to help you learn all about limits and continuity in calculus!"},
        {"role": "user", "content": prompt},
    ]
    # chat = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {"role": "user", "content": "Hello!"},
    #     {"role": "assistant", "content": "Hello there, young mathematician! I'm Cosmo, your friendly and intelligent assistant here to help you learn all about limits and continuity in calculus!"},
    #     {"role": "user", "content": "Can you explain limits thoroughly with examples?"},
    #     {"role": "assistant", "content": 'Sure, I\'d be happy to explain limits in calculus!\n\nLimits are a fundamental concept in calculus that allow us to study how functions behave as the input (or independent variable) approaches a certain value. In essence, limits help us understand how a function is "approaching" a certain point or value.\n\nThe concept of limits is based on the idea that a function can be made to approach a certain value arbitrarily closely by taking the input sufficiently close to a certain point. For example, consider a function that describes the height of an object as a function of time, such as the position of a ball thrown upwards. The height of the ball at a given time can be made to approach a certain value (e.g. the height at which the ball was thrown) by taking the time sufficiently close to the moment when the ball was thrown.\n\nThe formal definition of a limit is as follows:\n\nThe limit of a function f(x) as x approaches a, denoted by lim x→a f(x), is a value L if for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - a| < δ implies |f(x) - L| < ε.\n\nIn simpler terms, the limit of a function as x approaches a is L if the function can be made to approach L arbitrarily closely by taking x sufficiently close to a. The δ value represents the "sufficiently close" part of the definition, as it determines how close x needs to be to a in order for f(x) to be considered "close enough" to L. The ε value represents the "arbitrarily close" part of the definition, as it determines how close f(x) needs to be to L in order for it to be considered "close enough" to L.\n\nHere are some examples to illustrate the concept of limits:\n\n1. Let\'s consider the function f(x) = x^2. What is the limit of this function as x approaches 2?\n\nTo find the limit, we need to determine a value L such that for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - 2| < δ implies |f(x) - L| < ε.\n\nIn this case, we can choose L = 4, since f(x) = x^2 approaches 4 as x approaches 2. To find δ, we can use the fact that f(x) - 4 = (x - 2)^2. Since (x - 2)^2 is always positive, we can choose δ = 1, since for all x, 0 < |x - 2| < 1 implies |(x - 2)^2 - 4| < 1.\n\nTherefore, the limit of f(x) as x approaches 2 is 4.\n\n2. Now let\'s consider the function g(x) = 1/x. What is the limit of this function as x approaches 0?\n\nTo find the limit, we need to determine a value L such that for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - 0| < δ implies |g(x) - L| < ε.\n\nIn this case, we can choose L = 0, since g(x) approaches 0 as x approaches 0. To find δ, we can use the fact that g(x) - 0 = 1/x. Since 1/x is always positive, we can choose δ = 1, since for all x, 0 < |x - 0| < 1 implies |1/x - 0| < 1.\n\nTherefore, the limit of g(x) as x approaches 0 is 0.\n\n3. Finally, let\'s consider the function h(x) = sin(x). What is the limit of this function as x approaches π/2?\n\nTo find the limit, we need to determine a value L such that for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - π/2| < δ implies |h(x) - L| < ε.\n\nIn this case, we can choose L = 1, since h(x) approaches 1 as x approaches π/2. To find δ, we can use the fact that h(x) - 1 = sin(x) - 1. Since sin(x) is bounded between -1 and 1, we can choose δ = 0.1, since for all x, 0 < |x - π/2| < 0.1 implies |sin(x) - 1| < 0.1.\n\nTherefore, the limit of h(x) as x approaches π/2 is 1.\n\nI hope these examples help illustrate the concept of limits in calculus! Let me know if you have any further questions or need additional clarification.'},
    #     {"role": "user", "content": "Can you also explain how to derive using the formal limit definition of a derivative?"}
    # ]

    async def event_generator():
        streamer = llama.generate(chat)

        for new_text in streamer:
            print(new_text, end="", flush=True)

            if (new_text == ""):
                continue

            yield json.dumps({
                "data": new_text
            })

    return EventSourceResponse(
        event_generator(),
        sep="\n",
        media_type="application/x-ndjson",
        delay=STREAM_DELAY,
        retry=RETRY_TIMEOUT
    )


@app.get("/tokenize")
async def tokenize(request: Request):
    llama = Llama.instance(
        system_prompt=SYSTEM_PROMPT,
        hf_model=MODEL
    )

    prompt = request.query_params.get("prompt")

    if prompt is None:
        return Response("Please provide a prompt", status_code=400)

    chat = [
        # {"role": "system", "content": f"<<SYS>>{system_prompt}<</SYS>>"},
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello there, young mathematician! I'm Cosmo, your friendly and intelligent assistant here to help you learn all about limits and continuity in calculus!"},
        {"role": "user", "content": prompt},
    ]

    return llama.tokenize(chat)


@app.get("/test")
async def test(request: Request):
    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/Llama-2-70B-chat-AWQ", use_fast=True, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello there, young mathematician! I'm Cosmo, your friendly and intelligent assistant here to help you learn all about limits and continuity in calculus!"},
        {"role": "user", "content": "Can you explain limits thoroughly with examples?"},
        {"role": "assistant", "content": 'Sure, I\'d be happy to explain limits in calculus!\n\nLimits are a fundamental concept in calculus that allow us to study how functions behave as the input (or independent variable) approaches a certain value. In essence, limits help us understand how a function is "approaching" a certain point or value.\n\nThe concept of limits is based on the idea that a function can be made to approach a certain value arbitrarily closely by taking the input sufficiently close to a certain point. For example, consider a function that describes the height of an object as a function of time, such as the position of a ball thrown upwards. The height of the ball at a given time can be made to approach a certain value (e.g. the height at which the ball was thrown) by taking the time sufficiently close to the moment when the ball was thrown.\n\nThe formal definition of a limit is as follows:\n\nThe limit of a function f(x) as x approaches a, denoted by lim x→a f(x), is a value L if for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - a| < δ implies |f(x) - L| < ε.\n\nIn simpler terms, the limit of a function as x approaches a is L if the function can be made to approach L arbitrarily closely by taking x sufficiently close to a. The δ value represents the "sufficiently close" part of the definition, as it determines how close x needs to be to a in order for f(x) to be considered "close enough" to L. The ε value represents the "arbitrarily close" part of the definition, as it determines how close f(x) needs to be to L in order for it to be considered "close enough" to L.\n\nHere are some examples to illustrate the concept of limits:\n\n1. Let\'s consider the function f(x) = x^2. What is the limit of this function as x approaches 2?\n\nTo find the limit, we need to determine a value L such that for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - 2| < δ implies |f(x) - L| < ε.\n\nIn this case, we can choose L = 4, since f(x) = x^2 approaches 4 as x approaches 2. To find δ, we can use the fact that f(x) - 4 = (x - 2)^2. Since (x - 2)^2 is always positive, we can choose δ = 1, since for all x, 0 < |x - 2| < 1 implies |(x - 2)^2 - 4| < 1.\n\nTherefore, the limit of f(x) as x approaches 2 is 4.\n\n2. Now let\'s consider the function g(x) = 1/x. What is the limit of this function as x approaches 0?\n\nTo find the limit, we need to determine a value L such that for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - 0| < δ implies |g(x) - L| < ε.\n\nIn this case, we can choose L = 0, since g(x) approaches 0 as x approaches 0. To find δ, we can use the fact that g(x) - 0 = 1/x. Since 1/x is always positive, we can choose δ = 1, since for all x, 0 < |x - 0| < 1 implies |1/x - 0| < 1.\n\nTherefore, the limit of g(x) as x approaches 0 is 0.\n\n3. Finally, let\'s consider the function h(x) = sin(x). What is the limit of this function as x approaches π/2?\n\nTo find the limit, we need to determine a value L such that for every positive real number ε, there exists a positive real number δ such that for all x, 0 < |x - π/2| < δ implies |h(x) - L| < ε.\n\nIn this case, we can choose L = 1, since h(x) approaches 1 as x approaches π/2. To find δ, we can use the fact that h(x) - 1 = sin(x) - 1. Since sin(x) is bounded between -1 and 1, we can choose δ = 0.1, since for all x, 0 < |x - π/2| < 0.1 implies |sin(x) - 1| < 0.1.\n\nTherefore, the limit of h(x) as x approaches π/2 is 1.\n\nI hope these examples help illustrate the concept of limits in calculus! Let me know if you have any further questions or need additional clarification.'},
        {"role": "user", "content": "Can you also explain how to derive using the formal limit definition of a derivative?"},
        {"role": "assistant", "content": "  Yes, I'd be happy to explain how to derive a derivative using the formal limit definition of a derivative.\n\nThe formal limit definition of a derivative is:\n\n$$f'(a) = \\lim_{h \\to 0} \\frac{f(a+h) - f(a)}{h}$$\n\nThis definition states that the derivative of a function $f$ at a point $a$ is equal to the limit of the difference quotient as the change in $x$ approaches zero.\n\nTo use this definition to derive a derivative, we can follow these steps:\n\n1. Write the function $f(x)$ and the point $a$ at which we want to find the derivative.\n2. Write the difference quotient:\n\n$$\\frac{f(a+h) - f(a)}{h}$$\n\n3. Simplify the numerator and denominator of the difference quotient, if possible.\n4. Take the limit as $h$ approaches zero:\n\n$$\\lim_{h \\to 0} \\frac{f(a+h) - f(a)}{h}$$\n\n5. Evaluate the limit to find the derivative.\n\nLet's go through an example to illustrate this process:\n\nFind the derivative of the function $f(x) = x^2$ at the point $a = 2$.\n\n1. Write the function and point:\n\n$$f(x) = x^2$$\n\n$$a = 2$$\n\n2. Write the difference quotient:\n\n$$\\frac{f(2+h) - f(2)}{h}$$\n\n3. Simplify the numerator and denominator:\n\n$$\\frac{(2+h)^2 - 4}{h}$$\n\n4. Take the limit as $h$ approaches zero:\n\n$$\\lim_{h \\to 0} \\frac{(2+h)^2 - 4}{h}$$\n\n5. Evaluate the limit:\n\n$$\\lim_{h \\to 0} \\frac{(2+h)^2 - 4}{h} = \\lim_{h \\to 0} \\frac{4h^2 + 4h - 4}{h} = \\lim_{h \\to 0} (4h + 4 - 4) = 4$$\n\nTherefore, the derivative of $f(x) = x^2$ at $a = 2$ is $f'(2) = 4$.\n\nThis process can be used to find the derivative of a function at a specific point by using the formal limit definition of a derivative.", },
        {"role": "user",
            "content": "In that case, how would you derive f(x) = x^2 using this?"}
    ]

    templated = tokenizer(
        "  To derive f(x) = x^2 using the formal limit definition of a derivative, we can follow these steps:\n\n1. Write the function:\n\nf(x) = x^2\n\n2. Write the difference quotient:\n\n$$\\frac{f(x+h) - f(x)}{h}$$\n\n3. Simplify the numerator and denominator:\n\n$$\\frac{(x+h)^2 - x^2}{h}$$\n\n4. Expand the numerator:\n\n$$\\frac{(x+h)^2 - x^2}{h} = \\frac{x^2 + 2xh + h^2 - x^2}{h} = \\frac{2xh + h^2}{h}$$\n\n5. Take the limit as $h$ approaches zero:\n\n$$\\lim_{h \\to 0} \\frac{2xh + h^2}{h} = \\lim_{h \\to 0} (2x + h) = 2x$$\n\nTherefore, the derivative of f(x) = x^2 is f'(x) = 2x.",
        padding=True,
        return_tensors='pt'
    ).input_ids.cuda()

    print(templated[0].tolist().__len__())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
