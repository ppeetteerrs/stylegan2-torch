FROM ppeetteerrs/pytorch:latest

WORKDIR /workspace

RUN mamba install -y pytest pytest-cov dunamai

COPY requirements.txt /tmp/
RUN mamba install --file /tmp/requirements.txt

COPY requirements_dev.txt /tmp/
RUN mamba install --file /tmp/requirements_dev.txt

CMD "zsh"