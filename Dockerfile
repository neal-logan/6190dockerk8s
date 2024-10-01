## Basic image with scikit-learn and pandas
FROM python:3.12.6-slim

## Install sklearn and panda
RUN python -m pip install scikit-learn pandas

## Create a directory for the lab
RUN mkdir /lab
WORKDIR /lab

## Copy the training files
COPY ./train.py /lab/train.py

## Copy the data file - TEST ONLY
COPY ./data/thermography_data.csv /mnt/datalake/thermography_data.csv
