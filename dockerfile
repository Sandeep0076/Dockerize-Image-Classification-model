FROM continuumio/anaconda3
COPY . /app/
EXPOSE 5000
WORKDIR /app/
RUN apt-get update \
	&& apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python-headless
RUN pip install pickle-mixin
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]