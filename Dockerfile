# FROM continuumio/anaconda3:4.4.0
# RUN apt-get update && apt-get install -y libgtk2.0-dev && \
#     rm -rf /var/lib/apt/lists/*

# RUN /opt/conda/bin/conda update -n base -c defaults conda && \
#     /opt/conda/bin/conda install python=3.6 && \
#     /opt/conda/bin/conda install anaconda-client && \
#     /opt/conda/bin/conda install jupyter -y && \
#     /opt/conda/bin/conda install --channel https://conda.anaconda.org/menpo opencv3 -y && \
#     /opt/conda/bin/conda install numpy pandas scikit-learn matplotlib seaborn pyyaml h5py keras -y && \
#     /opt/conda/bin/conda upgrade dask && \
#     pip install tensorflow imutils

###############

# FROM python:3.8.8-alpine
# FROM alpine:latest
# RUN apk update && apk add gcc libc-dev make git libffi-dev oDpenssl-dev python3-dev libxml2-dev libxslt-dev 
# RUN apk add --no-cache --update \
#     python3 python3-dev gcc \
#     gfortran musl-dev g++ \
#     libffi-dev openssl-dev \
#     libxml2 libxml2-dev \
#     libxslt libxslt-dev \
#     libjpeg-turbo-dev zlib-dev

# RUN python3 python3-dev gcc \
#     gfortran musl-dev g++ \
#     libffi-dev openssl-dev \
#     libxml2 libxml2-dev \
#     libxslt libxslt-dev \
#     libjpeg-turbo-dev zlib-dev

# RUN apk add py2-numpy@community py2-scipy@community py-pandas@edge
# FROM python:3-alpine

# RUN apk add --no-cache python3-dev \
#     && pip3 install --upgrade pip


## not alpine
FROM python:3.8.8-slim


WORKDIR /app
COPY . /app
# RUN pip --no-cache-dir install -r requirements.txt
RUN pip  install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["fmodel2.py"]
