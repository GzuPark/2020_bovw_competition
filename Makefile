COMPETITION_NAME=2020backofwordrcv
DESTINATION_PATH=data

download :
	@echo "[ Info ] Start to download ' ${COMPETITION_NAME} ' dataset"
	@kaggle competitions download -c ${COMPETITION_NAME}
	@echo "[ Info ] Start to unzip dataset to ' ${DESTINATION_PATH} '"
	@unzip -q ${COMPETITION_NAME}.zip -d ${DESTINATION_PATH}
	@rm ${COMPETITION_NAME}.zip
	@echo "[ Info ] Complete to prepare dataset"
