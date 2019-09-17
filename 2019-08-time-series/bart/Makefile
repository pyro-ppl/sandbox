all: test

lint: FORCE
	flake8

test: lint FORCE
	python forecast.py -n 1 -v \
	  --truncate=168 \
	  --param-store-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

test2: lint FORCE
	python forecast.py -n 1 -v \
	  --tiny=2 \
	  --truncate=128 \
	  --param-store-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

test3: lint FORCE
	python forecast.py --analytic-kl -n 1 -v \
	  --tiny=2 \
	  --truncate=3 \
	  --param-store-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

FORCE:
