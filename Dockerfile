FROM ubuntu:20.04
ENV PATH /opt/conda/bin:$PATH

ENV TZ=Asia/Taipei \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /home/

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion && apt-get install tzdata

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda init bash

RUN conda install jupyter -y

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.ip = '0.0.0.0' #allow all ip" >> ~/.jupyter/jupyter_notebook_config.py 

RUN apt install openssh-server -y
RUN echo "Port 22" >> /etc/ssh/sshd_config
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

EXPOSE 22

RUN conda create --name FAPT_FATEL python=3.11.4 ipykernel=6.25.0 -y
RUN echo "conda activate FAPT_FATEL" >> ~/.bashrc
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate FAPT_FATEL && \
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y"
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate FAPT_FATEL && \
    pip install wheel==0.41.2 pandas==2.2.1 Pillow==9.3.0 numpy==1.24.1"

CMD /etc/init.d/ssh restart && jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root