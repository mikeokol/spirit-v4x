#!/usr/bin/env python3
"""
Create a quick dev user:  python scripts/dev_user.py
"""
import asyncio
import sys
from getpass import getpass
from spirit.db import async_session
from spirit.models import User
from spirit.api.auth import hash_pw

async def main():
    email = input("Email: ").strip()
    pw = getpass("Password: ")
    async with async_session() as db:
        exists = await db.get(User, email)
        if exists:
            print("User already exists.")
            return
        user = User(email=email, hashed_password=hash_pw(pw))
        db.add(user)
        await db.commit()
        print("Dev user created.")

if __name__ == "__main__":
    asyncio.run(main())
