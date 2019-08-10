all: test

lint: FORCE
	flake8

test: lint FORCE
	python forecast.py --truncate-hours 168 -n 1 -v
	@echo PASS

FORCE:
