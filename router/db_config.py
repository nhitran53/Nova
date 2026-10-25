import getpass
import os


def password_options(database_url):
    from psycopg.conninfo import conninfo_to_dict

    params = conninfo_to_dict(database_url)
    if "password" in params or os.getenv("PGPASSWORD"):
        return {}
    if params.get("service") or os.getenv("PGSERVICE"):
        return {}

    host = params.get("host") or os.getenv("PGHOST") or "localhost"
    address = params.get("hostaddr") or os.getenv("PGHOSTADDR") or host
    port = params.get("port") or os.getenv("PGPORT") or "5432"
    user = params.get("user") or os.getenv("PGUSER") or getpass.getuser()
    database = params.get("dbname") or os.getenv("PGDATABASE") or user
    password = os.getenv("NOVA_DB_PASSWORD")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if (
        password
        and host in local_hosts
        and address in local_hosts
        and port == "5432"
        and user == "cardin"
        and database == "nova"
    ):
        return {"password": password}
    return {}
