SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE ?= docker compose
API_CONTAINER ?= api

.PHONY: help
help: ## Show all available commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: setup
setup: ## Create local .env from .env.example if missing
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example"; else echo ".env already exists"; fi

.PHONY: up
up: ## Build and start all services in detached mode
	$(COMPOSE) up --build -d

.PHONY: down
down: ## Stop and remove all services
	$(COMPOSE) down

.PHONY: logs
logs: ## Follow logs for all services
	$(COMPOSE) logs -f

.PHONY: ps
ps: ## List running compose services
	$(COMPOSE) ps

.PHONY: restart
restart: ## Restart all services
	$(COMPOSE) restart

.PHONY: bootstrap-admin
bootstrap-admin: ## Create default admin user through API
	curl -sS -X POST http://localhost:8005/auth/bootstrap-admin | cat

.PHONY: shell-api
shell-api: ## Open shell inside API container
	$(COMPOSE) exec $(API_CONTAINER) bash

.PHONY: compile-backend
compile-backend: ## Compile Python backend to validate syntax
	python -m compileall backend/app

.PHONY: fmt
fmt: ## Format backend/frontend if tools are available
	@if command -v black >/dev/null 2>&1; then black backend/app; else echo "black not installed"; fi
	@if command -v isort >/dev/null 2>&1; then isort backend/app; else echo "isort not installed"; fi
	@if [ -d frontend ] && command -v npm >/dev/null 2>&1; then cd frontend && npm run build >/dev/null 2>&1 || true; else echo "npm not available or frontend missing"; fi

.PHONY: clean
clean: ## Remove Python cache files
	find backend -type d -name '__pycache__' -prune -exec rm -rf {} +
	find backend -type f -name '*.pyc' -delete
