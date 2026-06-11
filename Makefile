.PHONY: up down test fmt lint typecheck install

install:
	uv sync

up:
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest -v

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .

typecheck:
	uv run mypy src
