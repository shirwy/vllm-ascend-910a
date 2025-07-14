# OpDev-910A

## 项目目标、验证方式、进度

| 序号 | 目标 | 验证方式 | 进度 |
| - | - | - | - |
| 1 | 支持SwiGlu算子 | aclnnSwiGlu算子单算子运行通过/在运行Qwen3时可在图模式检测到aclnnSwiGlu算子 | 2025.6.25已支持 |
| 2 | 支持Grouped_Matmul算子 | 四卡910A跑通Qwen3 | 2025.7.1已支持 |
| 3 | 制作独立库打包本项目开发算子 | 联合编译后跑通Qwen3 | 2025.7.1已打包SwiGlu，Grouped_Matmul算子 |
| 4 | 根据性能瓶颈开发融合算子 | 性能测试有提升，已知910A多卡通信瓶颈大 | - |
| 5 | 支持key_cache等算子 | 跑通DeepSeek-R1模型 | - |

## 统一开发条件

### 开发硬件环境
```bash
uname -a
# Linux bms-jishuxiaozu 4.19.36-vhulk1907.1.0.h1665.eulerosv2r8.aarch64 #1 SMP Sun Nov 10 17:11:17 UTC 2024 aarch64 aarch64 aarch64 GNU/Linux
npu-smi info # 910A 八卡
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.0                   Version: 24.1.0                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B                | OK            | 70.0        34                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           1197 / 13553      1365 / 32768         |
+===========================+===============+====================================================+
| 1     910B                | OK            | 66.9        34                0    / 0             |
| 0                         | 0000:81:00.0  | 0           2411 / 15665      4    / 32768         |
+===========================+===============+====================================================+
| 2     910B                | OK            | 69.8        34                0    / 0             |
| 0                         | 0000:41:00.0  | 0           2306 / 15665      4    / 32768         |
+===========================+===============+====================================================+
| 3     910B                | OK            | 67.2        33                0    / 0             |
| 0                         | 0000:01:00.0  | 0           2472 / 15567      3    / 32768         |
+===========================+===============+====================================================+
| 4     910B                | OK            | 68.9        33                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           2904 / 13553      3    / 32768         |
+===========================+===============+====================================================+
| 5     910B                | OK            | 66.3        34                0    / 0             |
| 0                         | 0000:82:00.0  | 0           1694 / 15665      5    / 32768         |
+===========================+===============+====================================================+
| 6     910B                | OK            | 70.1        34                0    / 0             |
| 0                         | 0000:42:00.0  | 0           1987 / 15665      5    / 32768         |
+===========================+===============+====================================================+
| 7     910B                | OK            | 65.6        33                0    / 0             |
| 0                         | 0000:02:00.0  | 0           1777 / 15567      4    / 32768         |
+===========================+===============+====================================================+
```

### 远程连接方式
ssh -i /path/to/key.pem root@223.244.40.1

### 开发容器环境

基础分支: https://github.com/vllm-project/vllm-ascend/tree/v0.9.1rc1

配置流程：使用镜像 vllm-ascend-910a 创建容器，并将本地vllm-ascend映射进去，在容器内编译

进入OpDev-910A路径后
```bash
vim build_container.sh 
# 修改DEVICE,NAME,PORT
sh build_container.sh
```

## 端到端编译
进入容器后运行
```bash
export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib
export SOC_VERSION=Ascend910B

python3 -m pip install -v -e . --extra-index https://download.pytorch.org/whl/cpu/
```

### 添加新算子以及编译

在`csrc`目录下添加新算子，并在`torch_binding.cpp`下进行binding，具体可见`torch_binding.cpp`下的的`_swiglu`实现以及此目录下的其他算子实现

编译命令跟`容器内编译`命令一样，在容器内执行，重新编译整个vllm_ascend

简单测试算子正确性可使用如下命令 (在容器内)
```bash
import torch
import vllm_ascend.vllm_ascend_C

x = torch.randn(2, 2, device=0, dtype=torch.float)
print(x)
y = torch.ops._C._swiglu(x)
print(y)
```

## 调用独立库打包算子

在相同根目录下克隆独立库repo
```bash
cd ..
git clone https://github.com/monellz/ascend910a-extras 
```
根据该repo的readme编译算子

## 测试方式

编译本repo和独立库repo后，运行benchmarks/ops/llm_test.py来运行Qwen大模型推理

目前支持Qwen3-8B（单卡可跑通），Qwen3-30B-A3B（至少4卡，否则爆显存）

修改卡的数量通过
```Python
llm = LLM(model="/data/models/Qwen/Qwen3-30B-A3B",max_model_len=1024,tensor_parallel_size=2)
```

## 性能优化工具

参考资料：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/opdev/optool/atlasopdev_16_0092.html
- 记录算子级性能
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/simulator/Ascend910A/lib:$LD_LIBRARY_PATH && export ASCEND_SIMULATOR_MODE=1 && msprof op simulator --applicatrs.py" --soc-version=Ascend910B1 --output=./results/msprof_op
```
- 可视化工具：
https://ui.perfetto.dev/
chrome浏览器：chrome://tracing/

## 项目结构
MindIE-CANN: 不使用aclnnSwiGlu算子跑通qwen3模型的所需代码和流程



