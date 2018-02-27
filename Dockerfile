from Python:2

ADD . /

RUN pip install numpy scipy

CMD [ "python", "./main.py" ]