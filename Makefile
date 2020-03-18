COMPETITION_NAME = 2020backofwordrcv
DESTINATION_PATH = data
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

submit :
	@kaggle competitions submit -c ${COMPETITION_NAME} -f ./result/submit/${SUBMIT} -m ${MSG}

rank :
	@kaggle competitions leaderboard -c ${COMPETITION_NAME} --show
