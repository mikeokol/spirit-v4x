.PHONY: install dev test migrate upgrade clean deploy

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

dev:
	uvicorn spirit.main:app --reload --host 0.0.0.0

test:
	pytest -q

migrate:
	alembic revision --autogenerate -m "$(msg)"

upgrade:
	alembic upgrade head

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

deploy:
	git push origin main
