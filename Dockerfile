from python:2

ADD . /

RUN pip install numpy scipy pandas

CMD [ "python", "-u",  "./main.py" ]