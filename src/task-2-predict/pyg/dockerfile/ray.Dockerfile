FROM rayproject/ray-ml:1.3.0-py38-gpu as base
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt


FROM base
ENV PASSWORD ust21
EXPOSE 8888
EXPOSE 5000
RUN apt update -yq
RUN apt install git curl wget -yq
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension ms-python.python
CMD code-server --auth password --bind-addr 0.0.0.0:8888 /workspace
