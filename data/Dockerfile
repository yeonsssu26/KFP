FROM python:3

RUN mkdir /bert
WORKDIR /bert
COPY . /bert/

RUN pip install -r ../requirements.txt
ENTRYPOINT ["python", "preprocess.py"]