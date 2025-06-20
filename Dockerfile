FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y curl gnupg2 software-properties-common default-jdk 

# Download miniconda and install silently.
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSLo Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda clean -a -y && \
    conda init


# Set environment variables
ENV NUPLAN_DATA_ROOT=/root/nuplan/dataset
ENV NUPLAN_MAPS_ROOT=/root/nuplan/dataset/maps
ENV NUPLAN_EXP_ROOT=/root/nuplan/exp
ENV NUPLAN_DEVKIT_ROOT=/root/SAH-Drive/nuplan-devkit
ENV SAH_ROOT=/root/SAH-Drive
ENV INTERPLAN_PLUGIN_ROOT=/root/SAH-Drive/interPlan

# Copy project code
WORKDIR $SAH_ROOT
COPY . .

# Create Conda environment
WORKDIR $NUPLAN_DEVKIT_ROOT
RUN conda env create -f environment.yml \
 && echo "conda activate SAH-Drive" >> ~/.bashrc

SHELL ["conda", "run", "-n", "SAH-Drive", "/bin/bash", "-c"]

# install nuplan-devkit
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r requirements_torch.txt
RUN pip install -e .

# install tuplan_garage
WORKDIR $SAH_ROOT/tuplan_garage
RUN pip install -r requirements.txt
RUN pip install -e .

# install interPlan
WORKDIR $INTERPLAN_PLUGIN_ROOT
RUN pip install -e .

# Set the default working directory after the container starts
WORKDIR $SAH_ROOT

# Launch shell and load .bashrc 
CMD ["bash", "--login"]

