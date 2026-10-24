"""Importing an opt-in list (a Kartra export, or any CSV or spreadsheet) as leads.

Import happens in two steps. ``preview`` reads the file, guesses which column
holds what, normalises every phone number and reports what would happen (new
leads, updates, rejects, duplicates) without writing anything. ``apply`` then
writes it, with the consent the owner attested to for the whole batch.

The consent rules are the point of this module:

* consent comes from the owner's attestation of the form's wording, per
  channel (texts, calls). A per-row consent column can only take consent away
  from a row, never grant it;
* re-importing never clears an opt-out or a do-not-contact, and never turns a
  consent that is on back off.
"""

from __future__ import annotations

import csv
import io
import re

import database
from outreach_rules import lead_score, normalize_phone

# The fields a column can be mapped to, in the order the mapping form shows them.
FIELDS = {
    "first_name": "First name",
    "last_name": "Last name",
    "name": "Full name",
    "email": "Email",
    "phone": "Phone",
    "phone_country_code": "Phone country code",
    "kartra_id": "Kartra lead id",
    "magnet": "Lead magnet / form",
    "tags": "Tags",
    "lists": "Lists",
    "quiz_result": "Quiz result",
    "opted_in_at": "Opt-in date",
    "sms_consent": "Text consent (yes/no)",
    "call_consent": "Call consent (yes/no)",
}

# Header spellings each field is recognised by, lowercased with punctuation removed.
_ALIASES = {
    "first_name": ["first name", "firstname", "first", "lead first name", "given name"],
    "last_name": ["last name", "lastname", "last", "lead last name", "surname", "family name"],
    "name": ["name", "full name", "fullname", "lead name", "contact name"],
    "email": ["email", "email address", "e mail", "lead email", "emailaddress"],
    "phone": ["phone", "phone number", "mobile", "mobile phone", "cell", "cell phone",
              "lead phone", "telephone", "sms number"],
    "phone_country_code": ["phone country code", "country code", "phone code", "dial code"],
    "kartra_id": ["id", "lead id", "kartra id", "contact id"],
    "magnet": ["lead magnet", "magnet", "form", "form name", "opt in", "optin", "opt in form",
               "signup form", "freebie"],
    "tags": ["tags", "tag"],
    "lists": ["lists", "list", "list name", "subscribed lists"],
    "quiz_result": ["quiz result", "career path", "career result", "result", "quiz",
                    "quiz outcome", "recommended role"],
    "opted_in_at": ["date added", "created", "created at", "signup date", "opt in date",
                    "optin date", "subscribed", "subscribed at", "date"],
    "sms_consent": ["sms consent", "text consent", "sms opt in", "sms optin", "texting consent"],
    "call_consent": ["call consent", "phone consent", "calls consent"],
}

MAX_ROWS = 20000


def _norm_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def read_table(data: bytes, filename: str) -> tuple[list[str], list[dict]]:
    """Read a .csv or .xlsx upload into (headers, rows). Raises ValueError if unreadable."""
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover - openpyxl is a requirement
            raise ValueError("Reading .xlsx needs openpyxl") from exc
        workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheet = workbook.active
        values = list(sheet.iter_rows(values_only=True))
        if not values:
            return [], []
        headers = [str(h).strip() if h is not None else "" for h in values[0]]
        rows = []
        for raw in values[1:MAX_ROWS + 1]:
            if raw is None or all(v in (None, "") for v in raw):
                continue
            rows.append({headers[i]: ("" if v is None else str(v).strip())
                         for i, v in enumerate(raw) if i < len(headers) and headers[i]})
        return [h for h in headers if h], rows
    if not name.endswith((".csv", ".txt")):
        raise ValueError("Upload a .csv or .xlsx file")
    text = data.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    headers = [h.strip() for h in (reader.fieldnames or []) if h and h.strip()]
    rows = []
    for raw in reader:
        if len(rows) >= MAX_ROWS:
            break
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k}
        if any(row.values()):
            rows.append(row)
    return headers, rows


def guess_mapping(headers: list[str]) -> dict:
    """``{field: header}`` for every field a header obviously holds."""
    mapping = {}
    used = set()
    normalized = {h: _norm_header(h) for h in headers}
    for field, aliases in _ALIASES.items():
        for alias in aliases:
            match = next((h for h, n in normalized.items() if n == alias and h not in used), None)
            if match:
                mapping[field] = match
                used.add(match)
                break
    # A file with first and last names does not also need a full-name column.
    if "name" in mapping and "first_name" in mapping:
        mapping.pop("name")
    return mapping


def _yes(value: str) -> bool | None:
    word = (value or "").strip().lower()
    if not word:
        return None
    if word in ("1", "y", "yes", "true", "t", "on", "opted in", "optin", "subscribed", "checked"):
        return True
    if word in ("0", "n", "no", "false", "f", "off", "unchecked", "declined"):
        return False
    return None


def _split(value: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;|]", value or "") if p.strip()]


def offer_match_words(offer: dict) -> list[str]:
    """The words that say a row took this offer: its name, its key, and the match
    words the owner gave it in Setup (for example a Kartra tag or list name)."""
    words = [offer.get("name") or "", (offer.get("key") or "").replace("_", " ")]
    words += [w for w in re.split(r"[,;\n]", offer.get("match") or "")]
    return [w.strip().lower() for w in words if w and len(w.strip()) >= 3]


def magnets_from(offers, *texts: str) -> list[str]:
    """Which of the owner's offers a row's form, tags or lists mention, by offer key."""
    blob = " ".join(t for t in texts if t).lower()
    return [o["key"] for o in offers or [] if o.get("key")
            and any(word in blob for word in offer_match_words(o))]


def _offers(db_path) -> list:
    return database.get_outreach_profile(db_path=db_path).get("offers") or []


def normalize_row(raw: dict, mapping: dict, region: str = "US", offers=()) -> dict:
    """One file row as lead fields, or a reason it cannot be imported.

    Returns ``{fields, identities, reject, row_sms_consent, row_call_consent}``.
    """
    def get(field):
        header = mapping.get(field)
        return (raw.get(header) or "").strip() if header else ""

    name = get("name") or " ".join(p for p in (get("first_name"), get("last_name")) if p)
    email = get("email").lower()
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        email = ""
    phone_raw = get("phone")
    phone = normalize_phone(phone_raw, region=region, country_code=get("phone_country_code") or None)
    out = {"fields": {}, "identities": [], "reject": None,
           "row_sms_consent": _yes(get("sms_consent")), "row_call_consent": _yes(get("call_consent"))}
    if phone_raw and not phone:
        out["reject"] = f"Phone number {phone_raw!r} is not a valid number"
        return out
    if not phone and not email:
        out["reject"] = "No phone number or email"
        return out
    tags = _split(get("tags"))
    lists = _split(get("lists"))
    magnets = magnets_from(offers, get("magnet"), " ".join(tags), " ".join(lists))
    fields = {
        "name": name or None,
        "email": email or None,
        "phone": phone,
        "kartra_id": get("kartra_id") or None,
        "magnets": magnets,
        "tags": list(dict.fromkeys(tags + lists)),
        "quiz_result": get("quiz_result") or None,
        "opted_in_at": get("opted_in_at") or None,
    }
    out["fields"] = fields
    out["identities"] = [(kind, value) for kind, value in (
        ("phone", phone), ("email", email or None), ("kartra", fields["kartra_id"])) if value]
    return out


def _existing_lead_for(identities, db_path) -> tuple[int | None, bool]:
    """The lead these identities already belong to, and whether they disagree."""
    owners = {database.find_lead_by_identity(kind, value, db_path=db_path)
              for kind, value in identities}
    owners.discard(None)
    if not owners:
        return None, False
    return min(owners), len(owners) > 1


def preview(headers, rows, mapping, db_path: str = database.DB_PATH, sample: int = 15) -> dict:
    """What importing these rows would do, without writing anything."""
    seen = {}
    offers = _offers(db_path)
    report = {"total": len(rows), "new": 0, "update": 0, "rejected": [], "duplicates": [],
              "conflicts": [], "no_phone": 0, "sample": []}
    for index, raw in enumerate(rows, start=2):  # row 1 is the header
        row = normalize_row(raw, mapping, offers=offers)
        if row["reject"]:
            report["rejected"].append({"row": index, "reason": row["reject"]})
            continue
        key = next(iter(row["identities"]))
        dup_of = next((seen[i] for i in row["identities"] if i in seen), None)
        if dup_of:
            report["duplicates"].append({"row": index, "same_as_row": dup_of})
            continue
        for identity in row["identities"]:
            seen[identity] = index
        lead_id, conflict = _existing_lead_for(row["identities"], db_path)
        if conflict:
            report["conflicts"].append({"row": index, "reason": "Its phone and email belong to two "
                                        "different leads; merge them first"})
            continue
        if lead_id:
            report["update"] += 1
        else:
            report["new"] += 1
        if not row["fields"]["phone"]:
            report["no_phone"] += 1
        if len(report["sample"]) < sample:
            fields = row["fields"]
            report["sample"].append({"row": index, "name": fields["name"], "email": fields["email"],
                                     "phone": fields["phone"], "magnets": fields["magnets"],
                                     "quiz_result": fields["quiz_result"],
                                     "existing": bool(lead_id), "key": key[0]})
    return report


def upsert_one(row: dict, consent: dict, consent_source: str, consent_at: str | None,
               batch_id: int | None = None, source: str = "kartra",
               db_path: str = database.DB_PATH) -> tuple[str, int | None]:
    """Add or update one normalised row. Returns ``(outcome, lead_id)``.

    ``outcome`` is ``added``, ``updated``, ``unchanged`` or ``conflict`` (the
    row's phone and email belong to two different leads). Consent only ever
    goes up, and an opt-out or a do-not-contact is never touched.
    """
    lead_id, conflict = _existing_lead_for(row["identities"], db_path)
    if conflict:
        return "conflict", None
    sms = bool(consent.get("sms")) and row["row_sms_consent"] is not False and bool(row["fields"]["phone"])
    call = bool(consent.get("call")) and row["row_call_consent"] is not False and bool(row["fields"]["phone"])
    fields = row["fields"]
    if lead_id is None:
        new = dict(fields, source=source, import_batch_id=batch_id, sms_consent=sms, call_consent=call)
        if sms or call:
            new.update(consent_source=consent_source, consent_at=consent_at)
        new["score"] = lead_score(new)
        return "added", database.create_lead(row["identities"], db_path=db_path, **new)
    lead = database.get_lead(lead_id, db_path=db_path)
    changes = {}
    for key in ("name", "email", "phone", "kartra_id", "opted_in_at"):
        if fields.get(key) and not lead.get(key):
            changes[key] = fields[key]
    if fields.get("quiz_result") and fields["quiz_result"] != lead.get("quiz_result"):
        changes["quiz_result"] = fields["quiz_result"]
    magnets = list(dict.fromkeys((lead.get("magnets") or []) + fields["magnets"]))
    if magnets != (lead.get("magnets") or []):
        changes["magnets"] = magnets
    tags = list(dict.fromkeys((lead.get("tags") or []) + fields["tags"]))
    if tags != (lead.get("tags") or []):
        changes["tags"] = tags
    granted = False
    if sms and not lead.get("sms_consent"):
        changes["sms_consent"] = 1
        granted = True
    if call and not lead.get("call_consent"):
        changes["call_consent"] = 1
        granted = True
    if granted:
        sources = [s for s in (lead.get("consent_source"), consent_source) if s]
        changes["consent_source"] = " | ".join(dict.fromkeys(sources))
        changes["consent_at"] = lead.get("consent_at") or consent_at
    for kind, value in row["identities"]:
        database.add_lead_identity(lead_id, kind, value, db_path=db_path)
    if not changes:
        return "unchanged", lead_id
    changes["score"] = lead_score({**lead, **changes})
    database.update_lead(lead_id, db_path=db_path, **changes)
    return "updated", lead_id


def apply(rows, mapping, batch_id: int, consent: dict, consent_source: str, consent_at: str | None,
          db_path: str = database.DB_PATH) -> dict:
    """Write the rows as leads. ``consent`` is ``{"sms": bool, "call": bool}`` for the batch."""
    added = updated = rejected = 0
    seen = set()
    offers = _offers(db_path)
    for raw in rows:
        row = normalize_row(raw, mapping, offers=offers)
        if row["reject"] or any(i in seen for i in row["identities"]):
            rejected += 1
            continue
        seen.update(row["identities"])
        outcome, _lead_id = upsert_one(row, consent, consent_source, consent_at, batch_id=batch_id,
                                       db_path=db_path)
        if outcome == "added":
            added += 1
        elif outcome == "updated":
            updated += 1
        elif outcome == "conflict":
            rejected += 1
    return {"total": len(rows), "added": added, "updated": updated, "rejected": rejected}


# ── Kartra webhook ──────────────────────────────────────────────────────────

def flatten_payload(payload, prefix: str = "") -> dict:
    """A webhook body as one flat ``{lowercase key path: value}`` map.

    Kartra's outbound API is not documented publicly in detail, so this reads
    every shape it could reasonably take: JSON objects (``{"lead": {"email":
    ...}}``), form fields with brackets (``lead[email]``) or dots
    (``lead.email``). Every key is kept under its full path and under its last
    segment, so ``lead[first_name]`` is found as ``first_name``.
    """
    flat = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_payload(value, path))
        return flat
    if isinstance(payload, list):
        if all(not isinstance(v, (dict, list)) for v in payload):
            flat[prefix.lower()] = ", ".join(str(v) for v in payload)
            return flat
        for index, value in enumerate(payload):
            flat.update(flatten_payload(value, f"{prefix}.{index}"))
        return flat
    path = re.sub(r"\[([^\]]*)\]", r".\1", prefix).strip(".").lower()
    value = "" if payload is None else str(payload).strip()
    flat[path] = value
    last = path.rsplit(".", 1)[-1]
    flat.setdefault(last, value)
    return flat


# The webhook's field names, as Kartra (or a similar tool) may spell them.
_WEBHOOK_ALIASES = {
    "email": ["email", "lead.email", "email_address"],
    "first_name": ["first_name", "firstname", "lead.first_name"],
    "last_name": ["last_name", "lastname", "lead.last_name"],
    "name": ["name", "full_name", "lead.name"],
    "phone": ["phone", "lead.phone", "phone_number", "mobile"],
    "phone_country_code": ["phone_country_code", "lead.phone_country_code", "country_code"],
    "kartra_id": ["lead.id", "lead_id", "id"],
    "magnet": ["list.name", "list_name", "form.name", "form_name", "list", "action_details.list.name"],
    "tags": ["tags", "lead.tags", "tag.name", "tag_name"],
    "quiz_result": ["quiz_result", "custom_fields.quiz_result", "lead.quiz_result"],
    "sms_consent": ["has_opt_in_sms", "lead.has_opt_in_sms", "sms_opt_in", "sms_consent"],
    "call_consent": ["has_opt_in_calls", "call_consent", "phone_consent"],
}


def row_from_webhook(flat: dict, db_path: str = database.DB_PATH) -> dict:
    """A webhook body as a file row plus its mapping, ready for ``normalize_row``."""
    raw, mapping = {}, {}
    for field, aliases in _WEBHOOK_ALIASES.items():
        for alias in aliases:
            if flat.get(alias):
                raw[field] = flat[alias]
                mapping[field] = field
                break
    if "first_name" in mapping and "name" in mapping:
        mapping.pop("name")
    return normalize_row(raw, mapping, offers=_offers(db_path))
