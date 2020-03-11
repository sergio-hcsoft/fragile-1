current_dir = $(shell pwd)

PROJECT = fragile
VERSION ?= latest

.POSIX:
check:
	!(grep -R /tmp ./tests)
	flake8 --count fragile
	pylint fragile
	black --check fragile

.PHONY: test
test:
	pytest -s

.PHONY: docker-test
docker-test:
	find -name "*.pyc" -delete
	docker run --rm -it --network host -w /fragile --entrypoint python3 fragiletech/fragile:${VERSION} -m pytest


.PHONY: docker-build
docker-build:
	docker build --pull -t fragiletech/fragile:${VERSION} .

.PHONY: docker-push
docker-push:
	docker push fragiletech/fragile:${VERSION}

