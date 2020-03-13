COMPETITION_NAME = 2020backofwordrcv
DESTINATION_PATH = data

download :
	@echo "[ INFO ] Start to download ' ${COMPETITION_NAME} ' dataset"
	@kaggle competitions download -c ${COMPETITION_NAME}
	@echo "[ INFO ] Start to unzip dataset to ' ${DESTINATION_PATH} '"
	@unzip -q ${COMPETITION_NAME}.zip -d ${DESTINATION_PATH}
	@rm ${COMPETITION_NAME}.zip
	@echo "[ INFO ] Complete to prepare dataset"

run:
	@python main.py
