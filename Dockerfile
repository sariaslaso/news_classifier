FROM python:3.9


WORKDIR /code


COPY ./model/fake_news_classifier/model.keras /code/model.keras
COPY ./requirements.txt /code/requirements.txt


# install the package dependencies in the requirements.txt file
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# copy the app directory inside the /code directory
COPY ./main.py /code/main.py


# set the command to run the uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]