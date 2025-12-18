#!/usr/bin/env python3
"""
Script to promote a user to admin.
Usage: python make_admin.py <email>
"""
import sys
from database import Database

def make_admin(email: str):
    db = Database()
    
    if db.set_admin_by_email(email, is_admin=True):
        print(f"✓ Successfully promoted '{email}' to admin!")
        print("  They will now see the Admin link in the sidebar.")
    else:
        print(f"✗ User with email '{email}' not found.")
        print("  Make sure the user has registered first.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_admin.py <email>")
        print("Example: python make_admin.py admin@example.com")
        sys.exit(1)
    
    email = sys.argv[1]
    make_admin(email)

