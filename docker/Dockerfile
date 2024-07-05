FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /smp_ws

RUN pip install segmentation-models-pytorch

# ユーザー指定
ARG UID
ARG GID
ARG USERNAME
ARG GROUPNAME

RUN groupadd -g $GID $GROUPNAME
RUN useradd -m -u $UID -g $GID $USERNAME

RUN chown $USERNAME:$GROUPNAME -R /smp_ws

USER $USERNAME

# ターミナルの色指定
RUN echo "PS1='\[\e[41;37m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]$ '" >> /home/$USERNAME/.bashrc

ENTRYPOINT ["/bin/bash"]