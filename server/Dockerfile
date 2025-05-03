FROM python:3.10-slim
WORKDIR /app

# Copy all project files into the container
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
CMD ["flask", "run"]
