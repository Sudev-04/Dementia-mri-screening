# App server only (no training). Put a TLS-terminating proxy in front of it:
# phones only allow offline mode (service workers) over HTTPS.
FROM python:3.12-slim
WORKDIR /srv
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt gunicorn
COPY common common
COPY instruments instruments
COPY app app
COPY models models
COPY run_app.py .
ENV ALZ_INSTANCE=/data
VOLUME /data
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "300", "app:create_app()"]
