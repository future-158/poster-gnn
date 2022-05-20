SHELL := /usr/bin/bash
REPO := https://github.com/pygod-team/pygod.git

.PHONY: venv install prepare

install: install-stemgnn

install-prepare: 
	conda env create --file prepare-environment.yml

install-stemgnn:
	conda env remove -n poster-stemgnn -y
	conda env create --file stemgnn-environment.yml

# install-pygod:
# 	conda env create --file pygod-requirements.yml

# install-pyg:
# 	conda env create --name $(PREFIX_PYG) --file pyg_environment.yml

activate-prepare:
	rm venv; ln -s $(PREFIX_PREPARE) venv

activate-stemgnn:
	rm venv; ln -s $(PREFIX_STEMGNN) venv

prepare:
	python prepare/load_impute.py
	python task/stemgnn/pretransform.py
mlflow:
	nohup mlflow server --host 0.0.0.0 --port 23005 &


hpo:
	CUDA_VISIBLE_DEVICES=0 python task/stemgnn/main.py --multirun ostia=true 'args.dropout_rate=interval(0.1,0.5)' 'args.lr=interval(0.00001,0.01)' 'args.multi_layer=int(interval(1,5))' 'args.window_size=int(interval(1,31))'

visualize:
	 streamlit run task/stemgnn/visualize_best.py --server.address 0.0.0.0 --server.port 23006
