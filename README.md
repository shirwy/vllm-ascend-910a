# vllm-ascend-910a


获取镜像
```
docker pull quay.io/ascend/vllm-ascend:v0.9.1rc1
```
## 编译安装vllm-ascend
进入容器后运行
```bash
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib
export SOC_VERSION=Ascend910B

python3 -m pip install -v -e . --extra-index https://download.pytorch.org/whl/cpu/
```


## 编译安装算子包
```bash
git clone https://github.com/shirwy/ascend-ops
```
- 算子编译参考https://github.com/shirwy/ascend-ops
## 测试方式

运行benchmarks/ops/llm_test.py来运行Qwen大模型推理

目前支持Qwen3-8B（单卡可跑通），Qwen3-30B-A3B（4卡，否则爆显存）

修改卡的数量通过
```Python
llm = LLM(model="/data/models/Qwen/Qwen3-30B-A3B",max_model_len=1024,tensor_parallel_size=2)
```

## 服务化

```
vllm serve /home/ma-user/aicc/Qwen/Qwen3-30B-A3B/ -tp 4 --gpu-memory-utilization 0.95 --served-model-name "qwen3" --host 0.0.0.0 --port 18001
```



