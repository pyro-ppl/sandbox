all: test

lint: FORCE
	flake8

test: lint FORCE
	python forecast.py -n 1 -v \
	  --truncate-hours=168 \
	  --param-store-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

FORCE:
