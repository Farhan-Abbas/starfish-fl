FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install -y gcc python3-dev musl-dev libmagic1 libffi-dev netcat-traditional \
    build-essential libpq-dev cron vim dos2unix bash

COPY poetry.lock pyproject.toml /app/

# Install Poetry and project dependencies (without copying the whole app yet)
# This allows Docker to cache dependencies and avoid reinstalling them on every code change
RUN pip3 install poetry
RUN poetry install --no-root  # Install only dependencies, skip installing the local package

# Copy the rest of the project files into the container
# This includes your source code and other resources
COPY . /app/

# Install the local project package itself (fast, since dependencies are already cached)
RUN poetry install

# Cron Job
COPY crontab /etc/cron.d/crontab

RUN chmod 0644 /etc/cron.d/crontab

RUN touch /var/log/cron.log

RUN crontab /etc/cron.d/crontab

RUN chmod +x ./entrypoint.sh

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.10/site-packages"

RUN dos2unix ./entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]

CMD cron && poetry run python3 manage.py runserver 0.0.0.0:8000