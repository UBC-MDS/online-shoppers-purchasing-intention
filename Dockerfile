# Docker file for the online shoppers purchasing intention project
# Authors: Nico Van den Hooff, TZ Yan, Arijeet Chatterjee
# Created: 2021-12-07
# Last updated: 2021-12-10

# base container for dockerfile
FROM continuumio/miniconda3@sha256:92d7896124d940cb1815d3b59d8eaab9a8e86c801af2437658581465044b0a06

# update packages
RUN apt-get update -y

# install dev tools
RUN apt-get install gcc python3-dev chromium-driver -y

# install gnu make
RUN apt-get install make -y

# required python packages with conda install
RUN conda install --quiet -y -c conda-forge \
    "numpy=1.21.*" \
    "pandas=1.3.*" \
    "scikit-learn=1.*" \
    "scipy=1.7.*" \
    "docopt=0.6.*" \
    "xgboost=1.5.*" \
    "altair=4.1.*" \
    "altair_saver" \
    "matplotlib=3.5.*" \
    "ipykernel=6.5.*"

# required python packages with pip install
RUN pip install \
    "jupyter-book==0.12.*" \
    "altair-data-server==0.4.*"
