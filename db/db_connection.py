from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# PostgreSQL 연결 설정
DB_CONFIG = {
    "ip": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "schema": os.getenv("DB_SCHEMA")
}

# SQLAlchemy 엔진 생성
engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['ip']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?options=-c%20search_path={DB_CONFIG['schema']}"
)