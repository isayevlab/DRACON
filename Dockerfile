# Ubuntu 18.04 LTS
FROM nvidia/cuda:latest

#Update dependances and install latest miniconda
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install wget -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /miniconda
ENV PATH=/miniconda/bin:${PATH}
RUN pip install --upgrade pip

#Install required packages
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN apt-get install make

#Copy files
COPY . .

#Command
CMD ["/bin/bash"]

