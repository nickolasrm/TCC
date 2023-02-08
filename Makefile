## -------------------------------
##  ____  _   _ _   _
## | __ )| \ | | \ | |___
## |  _ \|  \| |  \| / __|
## | |_) | |\  | |\  \__ \
## |____/|_| \_|_| \_|___/
## Bits Neural Network Analysis
## -------------------------------

PACKAGE=bnn_analysis

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

run:  ## Run a specified experiment e.g. make run experiment=supervised/iris
	@echo "Running experiment"
	@if [ -z "$(experiment)" ]; then echo "No experiment specified"; exit 1; fi;
	poetry run python ./$(PACKAGE)/$(experiment).py; \

run3:  ## Run a specified experiment 3 times e.g. make run3 experiment=supervised/iris
	@echo "Running experiment 1"
	make run experiment=$(experiment)
	@echo "Running experiment 2"
	make run experiment=$(experiment)
	@echo "Running experiment 3"
	make run experiment=$(experiment)
