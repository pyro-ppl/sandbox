all: test

lint: FORCE
	flake8

test: lint FORCE
	python main.py -n 2 -v \
	  --truncate=168 \
	  --param-store-filename=/dev/null \
	  --forecaster-filename=/dev/null \
	  --forecast-filename=/dev/null \
	  --training-filename=/dev/null
	python main.py -n 2 -v \
	  --analytic-kl \
	  --truncate=168 \
	  --param-store-filename=/dev/null \
	  --forecaster-filename=/dev/null \
	  --forecast-filename=/dev/null \
	  --training-filename=/dev/null
	python main.py -n 2 -v \
	  --funsor \
	  --truncate=168 \
	  --param-store-filename=/dev/null \
	  --forecaster-filename=/dev/null \
	  --forecast-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

profile: lint FORCE
	python -m cProfile -o forecast.prof main.py -n 1 \
	  --funsor \
	  --truncate=128 \
	  --param-store-filename=/dev/null \
	  --forecaster-filename=/dev/null \
	  --forecast-filename=/dev/null \
	  --training-filename=/dev/null
	snakeviz forecast.prof &

test2: lint FORCE
	python main.py -n 1 -v --debug \
	  --funsor \
	  --tiny=2 \
	  --truncate=2 \
	  --state-dim=2 \
	  --guide-rank=2 \
	  --param-store-filename=/dev/null \
	  --forecaster-filename=/dev/null \
	  --forecast-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

test3: lint FORCE
	python main.py -n 1 -v --debug \
	  --funsor \
	  --analytic-kl \
	  --tiny=2 \
	  --truncate=2 \
	  --state-dim=2 \
	  --guide-rank=2 \
	  --param-store-filename=/dev/null \
	  --forecaster-filename=/dev/null \
	  --forecast-filename=/dev/null \
	  --training-filename=/dev/null
	@echo PASS

test-eval:
	python evaluate.py -f -n 2 --truncate=9000 -v

FORCE:
