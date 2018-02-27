from python:2

ADD . /bomberQL

RUN pip install numpy scipy pandas

CMD [ "python", "-u",  "./main.py" ]