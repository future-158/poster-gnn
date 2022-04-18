FROM nvcr.io/nvidia/pytorch:22.03-py3 as base
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
