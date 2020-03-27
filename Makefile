COMPETITION_NAME = 2020backofwordrcv
DESTINATION_PATH = data

IMAGE_NAME = tf2.1
PYTHON_VERSION = 3.7
CONDA_ENV_NAME = bow
UID := $(shell id -u)
USER_NAME := $(shell whoami)
PASSWORD = default

CONTAINER_NAME = bow
TENSORBOARD = 
NOTEBOOK = 
PORT = 
POS_GPUS = 0,1,2,3,4,5,6,7

TOPN=10
SUBMIT =
DATE := $(shell date "+%Y-%m-%d_%H:%M:%S")
MSG = "Submitted $(DATE)"

download :
	@echo "[ INFO ] Start to download ' ${COMPETITION_NAME} ' dataset"
	@kaggle competitions download -c ${COMPETITION_NAME}
	@echo "[ INFO ] Start to unzip dataset to ' ${DESTINATION_PATH} '"
	@unzip -q ${COMPETITION_NAME}.zip -d ${DESTINATION_PATH}
	@rm ${COMPETITION_NAME}.zip
	@echo "[ INFO ] Complete to prepare dataset"

docker-base:
	@nvidia-docker build -t rcv/${IMAGE_NAME} \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
		-f ./docker/base.Dockerfile \
		.

docker-user:
	@nvidia-docker build -t ${USER_NAME}/${IMAGE_NAME} \
		--build-arg IMAGE_NAME=${IMAGE_NAME} \
		--build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
		--build-arg UID=${UID} \
		--build-arg USER_NAME=${USER_NAME} \
		--build-arg PASSWORD=${PASSWORD} \
		-f ./docker/user.Dockerfile \
		.

docker-run:
	@nvidia-docker run -it -u ${USER_NAME} \
		--name ${USER_NAME}_${CONTAINER_NAME} \
		-p ${TENSORBOARD}:6006 \
		-p ${NOTEBOOK}:8888 \
		-p ${PORT}:22 \
		-e "TERM=xterm-256color" \
		-w /home/${USER_NAME}/workspace \
		-v /home/${USER_NAME}/workspace:/home/${USER_NAME}/workspace \
		-e NVIDIA_VISIBLE_DEVICES=${POS_GPUS} \
		--shm-size=32G \
		${USER_NAME}/${IMAGE_NAME} \
		/bin/bash

best-run:
	@python main.py --train --image-size 224 224 --network resnet101 --freeze

show:
	@python main.py --show --top-n ${TOPN}

submit :
	@kaggle competitions submit -c ${COMPETITION_NAME} -f ./result/submit/${SUBMIT} -m ${MSG}

rank :
	@kaggle competitions leaderboard -c ${COMPETITION_NAME} --show
