SOURCES=src scripts

all:
	@echo "Running isort..."
	@isort ${SOURCES}
	@echo "Running black..."
	@black ${SOURCES}/*/*.py

lint:
	@echo "Running black check..."
	@black --check --diff ${SOURCES}/*/*.py
	@echo "Running pylint..."
	@pylint ${SOURCES}
	@echo "Running mypy..."
	@mypy ${SOURCES}
	@echo "Running flake8..."
	@flake8 ${SOURCES}

