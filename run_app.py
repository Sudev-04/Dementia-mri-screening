"""Start the app or manage accounts.

    python run_app.py                      start the server on http://localhost:5000
    python run_app.py create-user          add an account (asks for the details)
    python run_app.py seed-demo            add two demo accounts for local testing only
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys

from app import create_app
from app.models import User, db

DEMO_USERS = [("field1", "Asha (demo field worker)", "field", "field-demo-1"),
              ("clinic1", "Dr Menon (demo clinician)", "clinician", "clinic-demo-1")]


def add_user(username: str, name: str, role: str, password: str) -> None:
    username = username.strip().lower()
    if role not in ("field", "clinician"):
        sys.exit("Role must be 'field' or 'clinician'.")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters.")
    user = User.query.filter_by(username=username).first() or User(username=username)
    user.name, user.role = name.strip() or username, role
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    print(f"saved {role} account '{username}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="serve", choices=["serve", "create-user", "seed-demo"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--demo-model", action="store_true",
                    help="allow the synthetic demo MRI model when no trained model is installed")
    args = ap.parse_args()
    if args.demo_model:
        os.environ["ALZ_ALLOW_DEMO_MODEL"] = "1"

    app = create_app()
    with app.app_context():
        if args.command == "create-user":
            add_user(input("Username: "), input("Full name: "), input("Role (field/clinician): ").strip(),
                     getpass.getpass("Password (8+ characters): "))
            return
        if args.command == "seed-demo":
            for u in DEMO_USERS:
                add_user(*u)
            print("Demo accounts are for local testing. Never use them with real people's data.")
            return
        if User.query.count() == 0:
            print("No accounts yet. Run:  python run_app.py seed-demo   (testing)  or  create-user")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
