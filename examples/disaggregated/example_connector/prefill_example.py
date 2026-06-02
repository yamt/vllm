# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def read_prompts():
    context = "Hi " * 1000
    context2 = "Hey " * 500
    return [
        # 28 tokens
        #"Hello. How are you doing? I'm fine. I guess this prompt should have at least 16 tokens to be cached. I have a"
        # 79 tokens
        #"Hello. How are you doing? I'm playing with vLLM and P/D disaggregation today. I guess this prompt should have at least 16 tokens to be cached. Well, not necessarily 16, but the effective block_size. Although vLLM's block_size is 16 by default, the CPU platform overrides it to 128 unless it's explicitly specified. I have a"
        #context + "Hello, my name is",
        #context + "The capital of France is",
        #context2 + "Your name is",
        #context2 + "The capital of China is",

        # 34 tokens (Qwen/Qwen2.5-0.5B-Instruct)
        "Question: There is a man on a side of river. He has a goat and a boat. How can he went to the other side of the river? Answer:"
    ]


def main():
    prompts = read_prompts()

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    llm = LLM(
        #model="meta-llama/Llama-3.2-1B-Instruct",
        #model="openai-community/gpt2",
        model="Qwen/Qwen2.5-7B-Instruct",
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        block_size=16,
        #kv_transfer_config=KVTransferConfig(
        #    kv_connector="ExampleConnector",
        #    kv_role="kv_both",
        #    kv_connector_extra_config={"shared_storage_path": "local_storage"},
        #),
    )  # , max_model_len=2048, max_num_batched_tokens=2048)

    # 1ST generation (prefill instance)
    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    new_prompts = []
    print("-" * 30)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 30)

    # Write new_prompts to output.txt
    with open("output.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
    print(f"Saved {len(new_prompts)} prompts to output.txt")


if __name__ == "__main__":
    main()
