.PHONY: all lint clean mrclean short_uni_synth

all: lint

lint: FORCE
	flake8

watch: FORCE
	ls -t logs/* | head -n 1 | xargs tail -f

short_uni_synth: FORCE
	python runner.py \
		--outfile=results/short_uni_synth \
		--script-filename=uni_synth.py \
		--args-filename=short_uni_synth.csv \
		--cores-per-worker=4 \
		--shuffle

clean: FORCE
	rm -rf temp logs errors

mrclean: FORCE
	rm -rf temp logs errors results

FORCE:
