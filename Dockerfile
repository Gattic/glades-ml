FROM gattic/gfxplusplus:latest
MAINTAINER Lerring

WORKDIR /

RUN mkdir -p /glades-ml
COPY . /glades-ml

RUN mkdir /glades-ml/build
WORKDIR /glades-ml/build
RUN cmake  ../
RUN make install

WORKDIR /
