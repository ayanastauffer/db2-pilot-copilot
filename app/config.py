import os
from dotenv import load_dotenv
import boto3

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
AWS_PROFILE = os.getenv("AWS_PROFILE") or None   # empty on EC2: uses the instance role
CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")

DB2_HOST = os.getenv("DB2_HOST", "localhost")
DB2_PORT = os.getenv("DB2_PORT", "50000")
DB2_DATABASE = os.getenv("DB2_DATABASE", "SAMPLE")
DB2_USER = os.getenv("DB2_USER", "")
DB2_PASSWORD = os.getenv("DB2_PASSWORD", "")
SQL_ENABLED = os.getenv("SQL_ENABLED", "true").lower() == "true"

MILVUS_URI = os.getenv("MILVUS_URI", "data/milvus.db")
COLLECTION = os.getenv("COLLECTION", "db2_docs")
RETRIEVAL = os.getenv("RETRIEVAL", "vector")   # "vector" or "hybrid"

API_KEY = os.getenv("API_KEY", "")


def bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.client("bedrock-runtime")


def db2_dsn():
    return (f"DATABASE={DB2_DATABASE};HOSTNAME={DB2_HOST};PORT={DB2_PORT};"
            f"PROTOCOL=TCPIP;UID={DB2_USER};PWD={DB2_PASSWORD};")
