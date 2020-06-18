.PHONY: all lint clean mrclean short_uni_synth

all: lint

lint: FORCE
	flake8

watch: FORCE
	ls -t logs/* | head -n 1 | xargs tail -f

short_uni_synth: FORCE
	python runner.py \
		--script-filename=uni_synth.py \
		--args-filename=short_uni_synth_svi.csv \
		--rng-seed=0,1,2,3,4 \
		--shuffle
	python runner.py \
		--script-filename=uni_synth.py \
		--args-filename=short_uni_synth_mcmc.csv \
		--cores-per-worker=1 \
		--rng-seed=0,1,2 \
		--shuffle

clean: FORCE
	rm -rf temp logs errors

mrclean: FORCE
	rm -rf temp logs errors results

FORCE:
