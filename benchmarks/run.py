from vllm import LLM, SamplingParams
import torch_npu
import time

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, max_tokens=2)
llm = LLM(model="/data/models/Qwen/Qwen3-8B/", max_model_len=1024)
# llm = LLM(model="/data/models/Qwen/Qwen3-30B-A3B/", max_model_len=1024, tensor_parallel_size=4)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# time.sleep(2)

# print("start", flush=True)
# iter_num = 2
# for i in range(iter_num):
#     outputs = llm.generate(prompts, sampling_params)
# print("end", flush=True)


experimental_config = torch_npu.profiler._ExperimentalConfig(
	export_type=[
		torch_npu.profiler.ExportType.Text,
		torch_npu.profiler.ExportType.Db
		],
	profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
	msprof_tx=False,
	aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
	l2_cache=False,
	op_attr=False,
	data_simplification=False,
	record_op_args=False,
	gc_detect_threshold=None
)

iter_num = 2

with torch_npu.profiler.profile(
	activities=[
		torch_npu.profiler.ProfilerActivity.CPU,
		torch_npu.profiler.ProfilerActivity.NPU
		],
	schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
	record_shapes=False,
	profile_memory=False,
	with_stack=True,
	with_modules=False,
	with_flops=False,
	experimental_config=experimental_config) as prof:
	for step in range(iter_num):
		outputs = llm.generate(prompts, sampling_params)
		prof.step()

prof.export_chrome_trace('./qwen8b-trace-stack.json')