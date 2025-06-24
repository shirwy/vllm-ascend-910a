#!/bin/bash
set -x
export DEVICE=/dev/davinci0
export NAME=${NAME:-xx_vllm_ascend}
export PROJECT_DIR=$(dirname $(realpath "${BASH_SOURCE[0]}"))

export IMAGE=vllm-ascend-910a

docker rm -f $NAME || true
docker run \
--name $NAME \
--privileged \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-v /data/models:/data/models \
-v ${PROJECT_DIR}:/workspace/vllm-ascend \
-p 8000:8000 \
-w /workspace/vllm-ascend \
-it $IMAGE \
bash
