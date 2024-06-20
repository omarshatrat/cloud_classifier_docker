FROM python:3.10

WORKDIR /app_test

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["pytest"]