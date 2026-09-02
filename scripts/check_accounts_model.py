"""G1: the accounts model holds several logins per platform, and defaults behave.

The whole feature rests on one claim: a platform is a list of accounts, not a
single login. This gate checks that claim at the data layer, including the parts
that are easy to get subtly wrong, such as which account inherits the default
when the current default is disconnected.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _accounts_gate import isolated_app, connect, check  # noqa: E402

directory, database, web, publisher, client = isolated_app()
P = database.DB_PATH

# --- several accounts per platform ---------------------------------------
work = connect(database, "linkedin", "li-work", "Work")
studio = connect(database, "linkedin", "li-studio", "Studio")
brand = connect(database, "threads", "th-brand", "brandco")

check(work != studio, "two LinkedIn logins collapsed into one account")
linkedin = database.list_social_accounts("linkedin", db_path=P)
check(len(linkedin) == 2, f"expected 2 LinkedIn accounts, got {len(linkedin)}")
check(database.count_social_accounts(db_path=P) == 3,
      "expected 3 accounts across platforms")

# --- re-authorising an account updates it rather than duplicating it ------
again = connect(database, "linkedin", "li-work", "Work Renamed")
check(again == work, "re-authorising an account created a second account")
check(len(database.list_social_accounts("linkedin", db_path=P)) == 2,
      "re-authorising changed the account count")

# --- the first account connected is the default ---------------------------
default = database.get_default_social_account("linkedin", db_path=P)
check(default["id"] == work, "the first account connected is not the default")
check(database.resolve_account_id("linkedin", db_path=P) == work,
      "a bare platform name does not resolve to the default account")

# --- connecting a second account does not steal the default --------------
check(database.get_default_social_account("linkedin", db_path=P)["id"] == work,
      "connecting a second account moved the default")

# --- the default can be moved, and only within its platform ---------------
database.set_default_social_account(studio, db_path=P)
check(database.get_default_social_account("linkedin", db_path=P)["id"] == studio,
      "setting the default did not take effect")
check(database.get_default_social_account("threads", db_path=P)["id"] == brand,
      "setting a LinkedIn default disturbed Threads")

# --- an account id from another platform is refused, not redirected -------
check(database.resolve_account_id("threads", studio, db_path=P) is None,
      "a LinkedIn account id resolved as a Threads target")
check(database.resolve_account_id("linkedin", 99999, db_path=P) is None,
      "an unknown account id resolved to something")

# --- naming an account ----------------------------------------------------
database.set_social_account_label(studio, "Studio Account", db_path=P)
check(database.get_social_account(studio, db_path=P)["label"] == "Studio Account",
      "the account label did not save")
database.set_social_account_label(studio, "", db_path=P)
check(database.get_social_account(studio, db_path=P)["label"] is None,
      "clearing the label did not clear it")

# --- disconnecting the default hands the default on -----------------------
database.delete_social_account(studio, db_path=P)
remaining = database.list_social_accounts("linkedin", db_path=P)
check(len(remaining) == 1, "disconnecting one account removed more than one")
check(remaining[0]["id"] == work, "the wrong account survived")
check(database.get_default_social_account("linkedin", db_path=P)["id"] == work,
      "disconnecting the default left the platform with no default")

# --- posts that named the removed account are not left pointing at it -----
second = connect(database, "linkedin", "li-second", "Second")
post_id = database.add_standalone_post(
    source_type="manual", source_content="gate", platform="linkedin",
    content="hello", account_id=second, db_path=P,
)
check(database.get_standalone_post(post_id, db_path=P)["account_id"] == second,
      "the saved post did not record its account")
database.delete_social_account(second, db_path=P)
check(database.get_standalone_post(post_id, db_path=P)["account_id"] is None,
      "a saved post still points at a disconnected account")

print(f"accounts model: {database.count_social_accounts(db_path=P)} accounts, "
      "defaults, renames and disconnects all behave")
print("ACCOUNTS_MODEL_OK")
