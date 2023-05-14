## -------------------------------
##  ____  _   _ _   _
## | __ )| \ | | \ | |___
## |  _ \|  \| |  \| / __|
## | |_) | |\  | |\  \__ \
## |____/|_| \_|_| \_|___/
## Bits Neural Network Analysis
## -------------------------------

PACKAGE=bnn_analysis
FLAGS=-O3 -march=tigerlake

help:  ## Show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

compile:  ## Compile cpp files
	@echo "Compiling cpp files"
	for binary in "bnn" "nn"; do \
		g++ "$(PACKAGE)/cpp/$${binary}.cpp" $(FLAGS) -o "$(PACKAGE)/cpp/$${binary}"; \
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

list:  ## List all experiments in the package
	@echo "Listing experiments"
	poetry run python -m bnn_analysis list

run:  ## Run an experiment.
##    Args: experiment=<experiment_name> [variant=<variant_name>]
##    Example: `make run experiment=cartpole variant=nobias`
	@echo "Running an experiment"
	if [ -z "$(variant)" ]; then \
		poetry run python -m bnn_analysis run $(experiment); \
	else \
		poetry run python -m bnn_analysis run $(experiment) --variant=$(variant); \
	fi;

run5:  ## Run the same experiment 5 times.
	@echo "Running an experiment"
	if [ -z "$(variant)" ]; then \
		poetry run python -m bnn_analysis run $(experiment) --repeat=5; \
	else \
		poetry run python -m bnn_analysis run $(experiment) --repeat=5 --variant=$(variant); \
	fi;

jupyter:  ## Open jupyter lab
	@echo "Running jupyter"
	poetry run jupyter lab --allow-root
