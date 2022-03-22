FROM ghcr.io/ppeetteerrs/pytorch:latest

RUN pip install "poetry>=1.2.*" poetry-dynamic-versioning torch-conv-gradfix

COPY .ssh /home/user/.ssh

RUN sudo chown -R user:user /home/user/.ssh

CMD ["zsh"]