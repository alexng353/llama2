from vllm import LLM, SamplingParams
import llama

prompts = [
    # "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)

llm = LLM(
    # model=llama.MODELS.LLAMA_2_13B_CHAT_GPTQ,
    model="ehartford/dolphin-2.1-mistral-7b",
    tensor_parallel_size=1,
    dtype="auto",
    gpu_memory_utilization=0.99,
    max_split_size_mb=1000,
)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
