#!/usr/bin/env python3
"""
One-shot helper: create tables without starting the server.
"""
import asyncio
from spirit.db import create_db_and_tables

if __name__ == "__main__":
    asyncio.run(create_db_and_tables())
    print("DB tables created.")
