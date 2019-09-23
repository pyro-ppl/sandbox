all: test

lint: FORCE
	flake8

test: lint FORCE
	python main.py -n 1 -v \
	  --truncate=168 \
	  --param-store-filename=/dev/null \
	  --forecaster=/dev/null \
	  --training-filename=/dev/null
	python main.py -n 1 -v \
	  --analytic-kl \
	  --truncate=168 \
	  --param-store-filename=/dev/null \
	  --forecaster=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

FORCE:
