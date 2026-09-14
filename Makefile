.PHONY: up down test fmt lint typecheck install check

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

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run pytest
