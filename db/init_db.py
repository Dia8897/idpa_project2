"""
Creates indexes on the MongoDB collections.
Run once before import_data.py:
    py db/init_db.py
"""

from db_config import get_db

db = get_db()

# countries: fast lookup by name
db.countries.create_index("name", unique=True)

# similarity: fast lookup by either country in a pair
db.similarity.create_index([("country_a", 1), ("country_b", 1)], unique=True)
db.similarity.create_index("country_a")
db.similarity.create_index("country_b")

print("Indexes created.")
