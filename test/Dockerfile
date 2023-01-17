FROM pytorch/pytorch

RUN mkdir /bert
WORKDIR /bert
COPY . /bert/

RUN pip install -r ../requirements.txt
ENTRYPOINT ["python", "test.py"]