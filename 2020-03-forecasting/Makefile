.PHONY: all lint test clean FORCE

all: test

lint: FORCE
	flake8 bart_*.py univariate.py

test: lint FORCE
	rm -rf /tmp/bart_results
	BART_RESULTS=/tmp/bart_results python bart_uni.py -n 1 --stride 100
	BART_RESULTS=/tmp/bart_results python bart_uni2.py -n 1 --stride 30000
	BART_RESULTS=/tmp/bart_results python bart_multi.py -n 1 --stride 50000 --train-window=32 --test-window=4

clean: FORCE
	rm -rf results

univariate: FORCE
	python -O univariate.py
	python -O univariate.py --method=SSVI,ASVI --num-samples=100000,1000000

FORCE:
