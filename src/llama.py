from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.generation import GenerationConfig
import os
from threading import Thread


class MODELS:
    LLAMA_2_7B_CHAT_GPTQ = "TheBloke/Llama-2-7b-Chat-GPTQ"
    LLAMA_2_7B_GPTQ = "TheBloke/Llama-2-7b-GPTQ"
    LLAMA_2_13B_CHAT_GPTQ = "TheBloke/Llama-2-13B-chat-GPTQ"
    LLAMA_2_13B_GPTQ = "TheBloke/Llama-2-13B-GPTQ"
    LLAMA_2_70B_CHAT_GPTQ = "TheBloke/Llama-2-70B-chat-GPTQ"
    LLAMA_2_70B_GPTQ = "TheBloke/Llama-2-70B-GPTQ"


class Llama:
    _instance = None
    model: any = None
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None

    config: GenerationConfig | None = None
    hf_model: str = MODELS.LLAMA_2_7B_CHAT_GPTQ

    system_prompt: str = """You are a deterministic, honest and truthful assistant program. You are being asked to generate a single multiple choice question based off of a given prompt.
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
Do your best to expand the format to properly fulfill the requested number of options."""

    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4096'))
    MIN_LENGTH = int(os.getenv('MIN_LENGTH', '1'))
    DO_SAMPLE = bool(os.getenv('DO_SAMPLE', 'True'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    TOP_K = int(os.getenv('TOP_K', '50'))
    TOP_P = float(os.getenv('TOP_P', '0.95'))

    def __init__(self) -> None:
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls, **kwargs):
        if cls._instance is not None:
            return cls._instance

        system_prompt = kwargs.get("system_prompt", None)
        hf_model = kwargs.get("hf_model", None)

        if system_prompt is not None:
            cls.system_prompt = system_prompt

        if hf_model is not None:
            cls.hf_model = hf_model

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

    def generate(self, prompt: list[dict[str, str]]):
        if self.model is None:
            raise RuntimeError("Model not instantiated yet")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not instantiated yet")

        templated = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            padding=True,
            return_tensors='pt'
        )

        print(templated)
        print()

        # templated_prompt = self.get_prompt(prompt)

        inputs = self.tokenizer(
            templated,
            padding=True,
            return_tensors='pt'
        ).input_ids.cuda()

        gen_cfg = self.get_config(self.model)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            inputs=inputs, streamer=streamer, generation_config=gen_cfg)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()

        return streamer

#     def get_prompt(self, prompt: str):
#         return f"""[INST] <<SYS>>
# {self.system_prompt}
# <</SYS>>
# {prompt}[/INST]"""

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
