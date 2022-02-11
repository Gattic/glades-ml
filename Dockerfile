FROM haskellol/gfxplusplus:docker-pants
MAINTAINER Lerring

WORKDIR /

RUN mkdir -p /glades-ml
COPY . /glades-ml

RUN mkdir /glades-ml/build
WORKDIR /glades-ml/build
RUN cmake  ../
RUN make install

WORKDIR /
