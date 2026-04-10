# websites domain
import os

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
GITLAB = os.environ.get("GITLAB", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
MAP = os.environ.get("MAP", "")
HOMEPAGE = os.environ.get("HOMEPAGE", "")

# Upstream asserts all 7 sites. We only run 4 (Shopping, Shopping Admin,
# GitLab, Reddit/Forum). Wikipedia and Map are skipped (~260 GB combined).
# Relax to require only the 4 we use; leave the others as empty strings
# so URL_MAPPINGS still builds without KeyError.
_required = {"REDDIT": REDDIT, "SHOPPING": SHOPPING,
             "SHOPPING_ADMIN": SHOPPING_ADMIN, "GITLAB": GITLAB}
_missing = [k for k, v in _required.items() if not v]
assert not _missing, (
    f"Missing site URLs in environment: {', '.join(_missing)}. "
    f"Add them to .env (see CLAUDE.md for the VM ports)."
)


ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}

URL_MAPPINGS = {
    REDDIT: "http://reddit.com",
    SHOPPING: "http://onestopmarket.com",
    SHOPPING_ADMIN: "http://luma.com/admin",
    GITLAB: "http://gitlab.com",
    WIKIPEDIA: "http://wikipedia.org",
    MAP: "http://openstreetmap.org",
    HOMEPAGE: "http://homepage.com",
}
