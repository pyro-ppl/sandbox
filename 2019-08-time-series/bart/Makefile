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

profile: lint FORCE
	python -m cProfile -o forecast.prof main.py -n 1 \
	  --truncate=128 \
	  --param-store-filename=/dev/null \
	  --training-filename=/dev/null
	snakeviz forecast.prof &

test2: lint FORCE
	python main.py -n 1 -v --debug \
	  --tiny=2 \
	  --truncate=2 \
	  --state-dim=2 \
	  --guide-rank=2 \
	  --param-store-filename=/dev/null \
	  --forecaster=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

test3: lint FORCE
	python main.py -n 1 -v --debug \
	  --analytic-kl \
	  --tiny=2 \
	  --truncate=2 \
	  --state-dim=2 \
	  --guide-rank=2 \
	  --param-store-filename=/dev/null \
	  --forecaster=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

FORCE:
