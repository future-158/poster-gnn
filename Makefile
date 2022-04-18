SHELL := /bin/bash
stemgnn:
	docker run -it --rm --gpus=device=1 -p "21000:8888" --shm-size=50gb --cpus='20' -v $(wd)/dataset:/dataset -v $(wd)/data:/data -v $(wd)/task-2-predict/stemgnn:/workspace -w /workspace poster/stemgnn




