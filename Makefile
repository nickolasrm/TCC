## -------------------------------
##  ____  _   _ _   _
## | __ )| \ | | \ | |___
## |  _ \|  \| |  \| / __|
## | |_) | |\  | |\  \__ \
## |____/|_| \_|_| \_|___/
## Bits Neural Network Analysis
## -------------------------------

help:  ## Show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

install:  ## Install package
	@echo "Installing package"
	pip install .

install-dev:  ## Install package for development
	@echo "Installing package for development"
	pip install poetry
	poetry install --no-root
	git init
	poetry run pre-commit install

test:  ## Run tests
	@echo "Running tests"
	poetry run pytest

lint:  ## Run linters
	@echo "Running linters"
	poetry run pre-commit

lint-all:  ## Run linters on all files
	@echo "Running linters on all files"
	poetry run pre-commit run --all-files
