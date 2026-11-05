"""WSGI entry point for gunicorn (Docker):  gunicorn app.wsgi:app"""
from app import create_app

app = create_app()
