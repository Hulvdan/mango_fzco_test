# Тестовое задание

## Как поднять

```
docker-compose up --build
```

Команду для сздания тестовых данных создавать не стал.
При запуске сразу создадутся таблицы БД с тестовыми данными.

Пользователи:

- `test1@test.com`
- `test2@test.com`
- `test3@test.com`

## Локальная разработка

```
# Должен быть pyenv
poetry install
pre-commit install
pre-commit install --install-hooks
docker-compose up db -d
poetry run fastapi src/main.py
```
