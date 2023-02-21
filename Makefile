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

compile:  ## Compile cpp files
	@echo "Compiling cpp files"
	for binary in "bnn" "nn"; do \
		g++ "$(PACKAGE)/cpp/$${binary}.cpp" -O0 -o "$(PACKAGE)/cpp/$${binary}"; \
	done;

install:  ## Install package for development
	@echo "Installing package for development"
	make compile
	pip install poetry
	poetry install --no-root && poetry install
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

list:
	@echo "Listing experiments"
	poetry run python -m bnn_analysis list

run:
	@echo "Running an experiment"
	poetry run python -m bnn_analysis run $(experiment) --variant=$(variant)

run5:
	@echo "Running an experiment"
	poetry run python -m bnn_analysis run $(experiment) --repeat=5 --variant=$(variant) 

jupyter:
	@echo "Running jupyter"
	poetry run jupyter lab --allow-root
