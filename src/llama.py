from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer, PreTrainedTokenizer, PreTrainedTokenizerFast
import timeit
from transformers.generation import GenerationConfig
import os
from datetime import datetime
from threading import Thread
from icecream import ic


class Llama:
    _instance = None
    model: any = None
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

    config: GenerationConfig | None = None
    models: list[str] = [
        "TheBloke/Llama-2-7b-Chat-GPTQ",
        "TheBloke/Llama-2-7b-GPTQ",
        "TheBloke/Llama-2-13B-chat-GPTQ"
    ]
    hf_model = models[2]

    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '400'))
    MIN_LENGTH = int(os.getenv('MIN_LENGTH', '1'))
    DO_SAMPLE = bool(os.getenv('DO_SAMPLE', 'True'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    TOP_K = int(os.getenv('TOP_K', '50'))
    TOP_P = float(os.getenv('TOP_P', '0.95'))

    def __init__(self) -> None:
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls):
        if cls._instance is not None:
            return cls._instance

        # instantiation code
        cls._instance = cls.__new__(cls)

        model = AutoModelForCausalLM.from_pretrained(
            cls.hf_model, device_map="auto", trust_remote_code=True, revision="main"
        )

        model.config.max_new_tokens = cls.MAX_TOKENS
        model.config.min_length = cls.MIN_LENGTH

        cls.model = model

        tokenizer = AutoTokenizer.from_pretrained(
            cls.hf_model, use_fast=True, device_map="auto")
        tokenizer.pad_token = tokenizer.eos_token

        cls.tokenizer = tokenizer

        return cls._instance

    def generate(self, prompt: str):
        if self.model is None:
            raise RuntimeError("Model not instantiated yet")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not instantiated yet")
        prompt_template = f"""[INST] <<SYS>>
You are a deterministic, honest and truthful assistant program. You are being asked to generate a single multiple choice question based off of a given prompt.
The machine that reads your responses is extremely strict, and will only accept the following XML-like format with no extraneous text:

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

Do your best to respond exactly in the aforementioned format, without any explanations of the question after the question is finished.
Do your best to expand the format to properly fulfill the requested number of options.
<</SYS>>
{prompt}[/INST]"""

        inputs = self.tokenizer(prompt_template, padding=True,
                                return_tensors='pt').input_ids.cuda()

        gen_cfg = self.get_config(self.model)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            inputs=inputs, streamer=streamer, generation_config=gen_cfg)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()

        return streamer

    def get_config(self, model, force: bool | None = None) -> GenerationConfig:
        if model is None:
            raise RuntimeError("Model not instantiated yet")

        if self.config is not None and (force is None or force is False):
            return self.config

        gen_cfg = GenerationConfig.from_model_config(model.config)

        gen_cfg.max_new_tokens = self.MAX_TOKENS
        gen_cfg.min_length = self.MIN_LENGTH
        gen_cfg.do_sample = self.DO_SAMPLE
        gen_cfg.temperature = self.TEMPERATURE
        gen_cfg.top_k = self.TOP_K
        gen_cfg.top_p = self.TOP_K

        self.config = gen_cfg
        return gen_cfg

    def get_respose_length(input, output):
        return len(list(output[0].tolist())) - len(list(input[0].tolist()))


if __name__ == "__main__":
    llama = Llama.instance()
    llama.generate("What is one plus one?")