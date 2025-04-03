from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_secret: str = Field(str)

    postgres_db: str = Field(str)
    postgres_host: str = Field(str)
    postgres_user: str = Field(str)
    postgres_password: str = Field(str)

    @property
    def postgres_dsn(self) -> str:
        return "postgresql+asyncpg://{}:{}@{}:5432/{}".format(
            self.postgres_user,
            self.postgres_db,
            self.postgres_host,
            self.postgres_password,
        )


settings = Settings(_env_file=".env")
