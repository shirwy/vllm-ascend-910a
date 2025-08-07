# ASCEND_LAUNCH_BLOCKING=1 python3 benchmarks/ops/ttest.py 
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# The first run will take about 3-5 mins (10 MB/s) to download models
llm = LLM(model="/home/ma-user/aicc/Qwen/Qwen3-30B-A3B/",max_model_len=1024,tensor_parallel_size=4)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")