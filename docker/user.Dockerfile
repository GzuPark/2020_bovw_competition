ARG IMAGE_NAME=

FROM rcv/$IMAGE_NAME

ARG CONDA_ENV_NAME
ARG UID
ARG USER_NAME
ARG PASSWORD

RUN adduser $USER_NAME -u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

USER $USER_NAME

RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitEmptyPasswords yes" >> /etc/ssh/sshd_config
RUN echo "UsePAM no" >> /etc/ssh/sshd_config

SHELL ["/bin/bash", "-c"]
RUN git clone https://github.com/magicmonty/bash-git-prompt.git ~/.bash-git-prompt --depth=1
RUN echo "if [ -f \"$HOME/.bash-git-prompt/gitprompt.sh\" ]; then" >> ~/.bashrc && \ 
    echo "  GIT_PROMPT_ONLY_IN_REPO=1" >> ~/.bashrc && \
    echo "  source $HOME/.bash-git-prompt/gitprompt.sh" >> ~/.bashrc && \
    echo "fi" >> ~/.bashrc

RUN echo "export PASSWORD=$PASSWORD" >> ~/.bashrc && \
    echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

RUN source activate ${CONDA_ENV_NAME} && jupyter notebook --generate-config

RUN jupyter_sha=$(python -c "from notebook.auth import passwd; print(passwd('${PASSWORD}'))") && \
    echo "c.NotebookApp.password=u'$jupyter_sha'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip='0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.terminado_settings = { 'shell_command': ['bash'] }" >> ~/.jupyter/jupyter_notebook_config.py

CMD /bin/bash
