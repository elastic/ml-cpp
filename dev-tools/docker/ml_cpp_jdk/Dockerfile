FROM ubuntu:20.04

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install aptitude ssh git openjdk-17-jre openjdk-17-jdk jq zip curl -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.11 -y
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN useradd -ms /bin/bash -u 1000 elasticsearch

USER elasticsearch
WORKDIR /home/elasticsearch
