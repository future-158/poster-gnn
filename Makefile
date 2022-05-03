SHELL := /usr/bin/zsh
REPO := https://github.com/pygod-team/pygod.git

.PHONY: venv install

PREFIX_PREPARE := envs/prepare
PREFIX_STEMGNN := envs/stemgnn
PREFIX_PYG := envs/pyg

install-prepare:
	conda create --prefix $(PREFIX_PREPARE) python=3.8 pip ipykernel pandera -yq && \
	conda run --prefix $(PREFIX_PREPARE) python -m pip install -r requirements.txt

install-pygod:
	conda run --prefix $(PREFIX_PREPARE) python -m pip install -r pygod_requirements.txt && \
	# [ -d repo ] && rm -rf repo; \
	# git clone $(REPO) repo && \
	# conda create --prefix venv python=3.8 pip ipykernel -yq && \
	# conda run --prefix venv python -m pip install -r requirements.txt && \
	# conda run --prefix venv bash -c "cd repo; python setup.py install" && \
	# rm -rf repo

install-pyg:
	mamba env create --prefix $(PREFIX_PYG) --file pyg_environment.yml

install-stemgnn:
	mamba create --prefix $(PREFIX_STEMGNN) python=3.8 pip ipykernel -yq && \
	conda run --prefix $(PREFIX_STEMGNN) python -m pip install -r stemgnn_requirements.txt

activate-prepare:
	[ -f venv ] && rm venv; ln -s $(PREFIX_PREPARE) venv

activate-pyg:
	[ -f venv ] && rm venv; ln -s $(PREFIX_PYG) venv

activate-stemgnn:
	[ -f venv ] && rm venv; ln -s $(PREFIX_STEMGNN) venv
	

clean-prepare:
	conda remove --prefix envs/pepare --all -y


install: install-prepare install-pygod install-stemgnn


# stemgnn:
# 	docker run -it --rm --gpus=device=1 -p "21000:8888" --shm-size=50gb --cpus='20' -v $(wd)/dataset:/dataset -v $(wd)/data:/data -v $(wd)/task-2-predict/stemgnn:/workspace -w /workspace poster/stemgnn