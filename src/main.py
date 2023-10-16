from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

import timeit
from icecream import ic

# def main():
#   transcriber = pipeline(model="openai/whisper-large-v2", device="cuda:0")
#   ic(transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))

def main():
  model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ", device="cuda:0"
  )
  tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")
  model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
  generated_ids = model.generate(**model_inputs)
  tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]





if __name__ == "__main__":
  ic(timeit.timeit(main, number=1))
  