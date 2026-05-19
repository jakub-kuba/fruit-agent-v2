import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    dialect_driver = "mysql+pymysql"
    username = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASS")
    host = os.getenv("MYSQL_HOST")
    dbname = os.getenv("FRUIT_DB")

    return create_engine(
        f"{dialect_driver}://{username}:{password}@{host}/{dbname}",
        pool_pre_ping=True
    )