FROM ubuntu

WORKDIR /src

RUN apt-get update
RUN apt-get -y install python3

COPY train_fl.py ./train_fl.py
COPY README.md ./README.md
COPY data ./data
COPY requirements.txt ./requirements.txt

RUN pip install -r ./requirements.txt