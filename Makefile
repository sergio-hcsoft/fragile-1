current_dir = $(shell pwd)

PROJECT = fragile

.POSIX:
check:
	!(grep -R /tmp ./tests)
	flake8 --count fragile
	pylint fragile
	black --check fragile

.PHONY: test
test:
	python3 -m pytest

.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm -it --network host -w /fragile --entrypoint python3 fragile -m pytest


.PHONY: docker-build
docker-build:
	docker build -t fragile .