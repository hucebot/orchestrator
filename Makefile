.PHONY: build up down bash log restart

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

bash:
	docker exec -it orchestrator bash

log:
	docker compose logs -f

restart: down up