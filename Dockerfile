FROM ppeetteerrs/pytorch:latest

WORKDIR /workspace

RUN mamba install -y pytest pytest-cov dunamai

CMD "zsh"