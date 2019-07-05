current_dir = $(shell pwd)

PROJECT = fragile

.PHONY: check
check:
	!(grep -R /tmp fragile/tests)
	flake8 --count
	pylint fragile
	black --check .

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