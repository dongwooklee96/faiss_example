import psycopg2

HOSTNAME = 'localhost'
USERNAME = 'postgres'
PASSWORD = 'password'
DATABASE_NAME = 'postgres'
PORT = 5432


def get_connection():
    try:
        connection = psycopg2.connect(
            user=USERNAME,
            password=PASSWORD,
            host=HOSTNAME,
            port=PORT,
            database=DATABASE_NAME,
        )
        return connection
    except Exception as e:
        message = f"get db connection error: ${e}"
        print(message)
