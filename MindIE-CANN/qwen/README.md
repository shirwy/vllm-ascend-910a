# 基于华为MindIE推理引擎运行docker中大模型回答的完整流程

## 进入docker环境
```
init.sh
```

## 修改配置参数
```
cd /usr/local/Ascend/mindie/latest/mindie-service/
vim conf/config.json
```
ServerConfig中port,managementPort,metricsPort变量都需要修改，不能重复。
BackendConfig中npuDeviceIds指定设备序号，modelName指定模型名称，随意设置即可，modelWeightPath指定模型权重保存路径。

## 拉起服务
```
cd /usr/local/Ascend/mindie/latest/mindie-service/
./bin/mindieservice_daemon
```

## 运行对话
打开新的终端，修改run.sh第一行"http://0.0.0.0:IPaddress/v1/chat/completions"中的IPaddress为ServerConfig中的port，然后运行下面命令
```
run.sh
```

回到拉起服务的终端，ctrl+C结束服务

## 已知问题：通过检测该环境下的静态图，可以发现没有调用aclnnSwiGlu，只调用了aclnnSwish算子，单算子调试也没有解决