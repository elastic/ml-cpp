FROM ubuntu:20.04

RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install aptitude ssh git openjdk-17-jre openjdk-17-jdk jq -y
RUN useradd -ms /bin/bash -u 1000 elasticsearch

USER elasticsearch
WORKDIR /home/elasticsearch
