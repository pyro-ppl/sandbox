.PHONY: all lint clean mrclean short_uni_synth

all: lint

lint: FORCE
	flake8

short_uni_synth: FORCE
	python runner.py --experiment=short_uni_synth

clean: FORCE
	rm -rf temp logs errors

mrclean: FORCE
	rm -rf results

FORCE:
