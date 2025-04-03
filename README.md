# Тестовое задание

## Как поднять

```
docker-compose up --build
```

## Локальная разработка

```
# Должен быть pyenv
poetry install
pre-commit install
pre-commit install --install-hooks
docker-compose up db -d
poetry run fastapi src/main.py
```
