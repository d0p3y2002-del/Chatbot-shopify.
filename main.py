from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
import re
from datetime import datetime, timedelta

# -----------------------
# Setup
# -----------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Put it in .env as: OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
GYMS_DIR = BASE_DIR / "gyms"
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

CONFIDENT_ENQUIRY_LINE = (
    "I want to make sure you get the right answer. "
    "The best next step is to send us an enquiry and the team will get back to you shortly."
)

# -----------------------
# Models
# -----------------------
class QuestionRequest(BaseModel):
    question: str
    session_id: str | None = None

# -----------------------
# Utilities
# -----------------------
def clamp(s: str, n: int = 1200) -> str:
    return (s or "").strip()[:n]

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def clean_summary_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    if s.strip().lower() == "nan":
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    s = s.lstrip('"').lstrip("'").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def is_random_or_nonsense(q: str) -> bool:
    s = (q or "").strip()
    if not s or len(s) <= 2:
        return True
    letters = sum(ch.isalpha() for ch in s)
    if letters == 0 and len(s) < 12:
        return True
    if letters / max(len(s), 1) < 0.25 and len(s) < 25:
        return True
    return False

def titlecase_first_letter(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s[0].upper() + s[1:]

def gym_xlsx_path(gym_name: str) -> Path:
    file_path = GYMS_DIR / f"{gym_name}.xlsx"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Gym file not found: {file_path}")
    return file_path

def load_excel_sheets(gym_name: str) -> dict:
    try:
        return pd.read_excel(gym_xlsx_path(gym_name), sheet_name=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read Excel: {e}")

def log_question(gym_name: str, question: str, source: str):
    try:
        out = LOGS_DIR / f"{gym_name}_questions.csv"
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "gym": gym_name,
            "question": clamp(question, 600),
            "source": source,
        }
        df = pd.DataFrame([row])
        if out.exists():
            df.to_csv(out, mode="a", header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(out, mode="w", header=True, index=False, encoding="utf-8")
    except Exception:
        pass

# -----------------------
# Intents
# -----------------------
def is_location_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["where", "address", "located", "location", "postcode", "post code", "directions"])

def is_memberships_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in [
        "membership", "memberships", "pricing", "prices", "cost",
        "how much", "fees", "direct debit", "day pass", "week pass", "monthly", "annual"
    ])

def is_timetable_intent(q: str) -> bool:
    ql = (q or "").lower()
    if any(k in ql for k in [
        "timetable", "time table", "schedule", "class schedule",
        "class timetable", "class times", "session times",
        "timeable", "timetabel", "timetible"
    ]):
        return True
    return "timetab" in ql

def is_link_request_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["link", "website", "page", "url", "send me", "where can i find", "where do i find", "more info"])

def is_classes_general_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["classes", "class", "sessions", "what do you offer", "what do you run", "what do you have"])

def is_group_vs_individual_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["group or", "group /", "group vs", "individual", "1:1", "one to one", "personal training", "private"])

def is_challenge_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["21 day", "21-day", "challenge", "transformation"])

def looks_like_class_explain_intent(q: str) -> bool:
    ql = (q or "").lower()
    triggers = ["tell me about", "what is", "what's", "whats", "explain", "meaning of", "define", "is move", "what does"]
    return any(t in ql for t in triggers)

def is_coaches_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["coach", "coaches", "trainers", "team", "staff", "instructor"])

def mentions_group_classes(q: str) -> bool:
    ql = (q or "").lower()
    return ("group" in ql and ("class" in ql or "classes" in ql or "session" in ql or "sessions" in ql)) or ("group training" in ql)

def mentions_personal_training(q: str) -> bool:
    ql = (q or "").lower()
    return ("personal training" in ql) or (" pt " in f" {ql} ") or ("1:1" in ql) or ("one to one" in ql) or ("private training" in ql)

def is_pricing_intent(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "price", "pricing", "cost", "how much",
        "membership", "pass", "day pass",
        "monthly", "weekly", "annual"
    ])

# -----------------------
# Follow-up rule: ONLY after the first assistant answer per session_id
# and ONLY if no links/buttons and no CTA in that response.
# -----------------------
FIRST_ANSWER_FOLLOWUP_LINE = "If you want a hand choosing the best option, just tell me what you’re aiming for."
SESSION_ANSWER_COUNT: dict[str, int] = {}

def is_first_answer(session_id: str | None) -> bool:
    if not session_id:
        return False
    return SESSION_ANSWER_COUNT.get(session_id, 0) == 0

def mark_answer_sent(session_id: str | None):
    if not session_id:
        return
    SESSION_ANSWER_COUNT[session_id] = SESSION_ANSWER_COUNT.get(session_id, 0) + 1

def should_add_follow_up_first_answer(*, session_id: str | None, has_links: bool, has_cta: bool) -> bool:
    if not is_first_answer(session_id):
        return False
    if has_links or has_cta:
        return False
    return True

# -----------------------
# Brand
# -----------------------
def load_brand(sheets: dict, gym_name: str) -> dict:
    brand = {
        "theme": "dark",
        "primary_color": "#ff2ea6",
        "gym_display_name": gym_name,
        "logo_url": "",
        "welcome_message": "Hi — welcome! How can I help?",
        "booking_url": "",
    }
    if "Brand" not in sheets:
        brand["welcome_message"] = titlecase_first_letter(brand["welcome_message"])
        return brand

    df = sheets["Brand"]
    if "Key" not in df.columns or "Value" not in df.columns:
        brand["welcome_message"] = titlecase_first_letter(brand["welcome_message"])
        return brand

    mapping = {}
    for _, row in df.iterrows():
        k = str(row.get("Key", "")).strip()
        v = str(row.get("Value", "")).strip()
        if k and v and k.lower() != "nan" and v.lower() != "nan":
            mapping[k] = v

    brand["theme"] = mapping.get("Theme", brand["theme"])
    brand["primary_color"] = mapping.get("PrimaryColor", brand["primary_color"])
    brand["gym_display_name"] = mapping.get("GymDisplayName", brand["gym_display_name"])
    brand["logo_url"] = mapping.get("LogoUrl", brand["logo_url"])
    brand["welcome_message"] = mapping.get("WelcomeMessage", brand["welcome_message"])
    brand["booking_url"] = mapping.get("BookingUrl", brand["booking_url"])

    brand["welcome_message"] = titlecase_first_letter(brand["welcome_message"])
    return brand

# -----------------------
# Links
# -----------------------
def load_links(sheets: dict) -> list[dict]:
    if "Links" not in sheets:
        return []
    df = sheets["Links"].copy()
    needed = ["Title", "URL", "Keywords", "Summary"]
    for c in needed:
        if c not in df.columns:
            return []

    links = []
    for _, r in df.iterrows():
        title = normalize_spaces(str(r.get("Title", "")).strip())
        url = str(r.get("URL", "")).strip()
        keywords_raw = str(r.get("Keywords", "")).strip()
        summary = clean_summary_text(r.get("Summary", ""))

        if not title or title.lower() == "nan":
            continue
        if not url or url.lower() == "nan":
            continue

        kw = [k.strip().lower() for k in re.split(r"[,\|;]", keywords_raw) if k and k.strip()]
        links.append({"title": title, "url": url, "keywords": kw, "summary": summary})
    return links

def find_link_by_title_contains(links: list[dict], needle: str) -> dict | None:
    n = (needle or "").lower().strip()
    for item in links:
        if n in item["title"].lower():
            return item
    return None

def best_link_match(user_q: str, links: list[dict]) -> dict | None:
    ql = (user_q or "").lower()
    if not ql:
        return None

    best = None
    best_score = 0
    for item in links:
        score = 0
        for kw in item["keywords"]:
            if kw and kw in ql:
                score += 5
        title_tokens = re.findall(r"[a-z0-9]+", item["title"].lower())
        q_tokens = re.findall(r"[a-z0-9]+", ql)
        score += len(set(title_tokens) & set(q_tokens)) * 2

        if "homepage" in item["title"].lower() and score > 0:
            score -= 3

        if score > best_score:
            best_score = score
            best = item

    return best if best_score >= 3 else None

# -----------------------
# FAQ
# -----------------------
def load_faq(sheets: dict) -> list[dict]:
    sheet_name = None
    for name in sheets.keys():
        if str(name).strip().lower() in ["faq", "faqs", "q&a", "qa", "questions"]:
            sheet_name = name
            break
    if not sheet_name:
        return []

    df = sheets[sheet_name].copy()
    cols = {str(c).strip().lower(): c for c in df.columns}
    q_col = cols.get("questions") or cols.get("question") or cols.get("q")
    a_col = cols.get("answers") or cols.get("answer") or cols.get("a")
    if not q_col or not a_col:
        return []

    out = []
    for _, r in df.iterrows():
        q = normalize_spaces(str(r.get(q_col, "")).strip())
        a = clean_summary_text(r.get(a_col, ""))
        if q and a and q.lower() != "nan" and a.lower() != "nan":
            out.append({"q": q.lower(), "a": a})
    return out

def faq_match(user_q: str, faqs: list[dict]) -> str | None:
    uq = normalize_spaces(user_q).lower()
    if not uq:
        return None

    for item in faqs:
        if uq == item["q"]:
            return item["a"]

    uq_tokens = set(re.findall(r"[a-z0-9]+", uq))
    best_a = None
    best_score = 0
    for item in faqs:
        q_tokens = set(re.findall(r"[a-z0-9]+", item["q"]))
        score = len(uq_tokens & q_tokens)
        if score > best_score:
            best_score = score
            best_a = item["a"]

    return best_a if best_score >= 3 else None

# -----------------------
# Pricing (Pricing sheet)
# -----------------------
def load_pricing(sheets: dict) -> list[dict]:
    if "Pricing" not in sheets:
        return []

    df = sheets["Pricing"].copy()
    if df.empty:
        return []

    cols = {str(c).strip().lower(): c for c in df.columns}
    item_col = cols.get("item")
    price_col = cols.get("price")
    url_col = cols.get("url")
    kw_col = cols.get("keywords")

    if not item_col or not price_col:
        return []

    out = []
    for _, r in df.iterrows():
        item = normalize_spaces(str(r.get(item_col, "")).strip())
        price = normalize_spaces(str(r.get(price_col, "")).strip())
        url = str(r.get(url_col, "")).strip() if url_col else ""
        keywords_raw = str(r.get(kw_col, "")).strip() if kw_col else ""

        if not item or item.lower() == "nan":
            continue
        if not price or price.lower() == "nan":
            continue

        kw = []
        for k in re.split(r"[,\|;]", keywords_raw):
            k = k.strip().lower()
            if k and k != "nan":
                kw.append(k)

        out.append({
            "item": item,
            "price": price,
            "url": url if url and url.lower() != "nan" else "",
            "keywords": kw,
        })

    return out

def match_pricing(user_q: str, pricing: list[dict]) -> list[dict]:
    ql = (user_q or "").lower().strip()
    if not ql:
        return []

    hits = []
    for p in pricing:
        score = 0

        for kw in p.get("keywords", []):
            if kw and kw in ql:
                score += 5

        item_tokens = set(re.findall(r"[a-z0-9]+", (p.get("item") or "").lower()))
        q_tokens = set(re.findall(r"[a-z0-9]+", ql))
        score += len(item_tokens & q_tokens) * 2

        if score >= 5:
            hits.append((score, p))

    hits.sort(key=lambda x: x[0], reverse=True)

    if not hits:
        return []
    top_score = hits[0][0]
    best = [p for s, p in hits if s >= top_score - 2]
    return best[:2]

def format_pricing_list(pricing: list[dict]) -> str:
    lines = ["Our prices are:"]
    for p in pricing:
        lines.append(f"- {p['item']}: {p['price']}")
    return "\n".join(lines).strip()

# -----------------------
# Coaches
# -----------------------
def load_coaches(sheets: dict) -> list[dict]:
    if "Coaches" not in sheets:
        return []
    df = sheets["Coaches"].copy()
    if df.empty:
        return []

    cols = {str(c).lower().strip(): c for c in df.columns}
    name_col = cols.get("name") or cols.get("coach") or cols.get("trainer") or df.columns[0]
    bio_col = cols.get("bio") or cols.get("description") or cols.get("about") or cols.get("text") or cols.get("summary")
    kw_col = cols.get("keywords") or cols.get("tags")

    out = []
    for _, r in df.iterrows():
        name = str(r.get(name_col, "")).strip()
        if not name or name.lower() == "nan":
            continue

        bio = clean_summary_text(r.get(bio_col, "")) if bio_col else ""
        if not bio:
            parts = []
            for c in df.columns:
                if c == name_col:
                    continue
                v = r.get(c, "")
                s = str(v).strip()
                if s and s.lower() != "nan":
                    parts.append(f"{c}: {s}")
            bio = "\n".join(parts).strip() or "Coach profile details are available on request."

        kw = set(re.findall(r"[a-z0-9]+", name.lower()))
        if kw_col:
            raw = str(r.get(kw_col, "")).strip()
            for k in re.split(r"[,\|;]", raw):
                k = k.strip().lower()
                if k:
                    kw.add(k)

        out.append({"name": name, "name_lower": name.lower(), "keywords": list(kw), "bio": bio})
    return out

def match_coach(user_q: str, coaches: list[dict]) -> dict | None:
    ql = (user_q or "").lower()
    if not ql:
        return None
    q_tokens = set(re.findall(r"[a-z0-9]+", ql))

    best = None
    best_score = 0
    for c in coaches:
        score = 0
        name_tokens = set(re.findall(r"[a-z0-9]+", c["name_lower"]))
        score += len(q_tokens & name_tokens) * 4
        for kw in c["keywords"]:
            if kw and kw in ql:
                score += 3
        if score > best_score:
            best_score = score
            best = c

    return best if best_score >= 4 else None

# -----------------------
# Classes sheet (optional)
# -----------------------
def load_class_defs(sheets: dict) -> list[dict]:
    if "Classes" not in sheets:
        return []
    df = sheets["Classes"].copy()
    cols = {str(c).lower().strip(): c for c in df.columns}
    if "name" not in cols or "description" not in cols:
        return []

    out = []
    for _, r in df.iterrows():
        name = str(r.get(cols["name"], "")).strip()
        desc = clean_summary_text(r.get(cols["description"], ""))
        if not name or not desc or name.lower() == "nan":
            continue

        keywords = str(r.get(cols.get("keywords", ""), "")).strip() if "keywords" in cols else ""
        kw_list = [k.strip().lower() for k in re.split(r"[,\|;]", keywords) if k and k.strip()]
        kw_list.append(name.strip().lower())

        url = str(r.get(cols.get("url", ""), "")).strip() if "url" in cols else ""
        out.append(
            {
                "name": name.strip(),
                "name_lower": name.strip().lower(),
                "keywords": list(dict.fromkeys(kw_list)),
                "description": desc,
                "url": url.strip(),
            }
        )
    return out

def match_class_def(user_q: str, class_defs: list[dict]) -> dict | None:
    ql = (user_q or "").lower()
    if not ql:
        return None

    best = None
    best_score = 0
    q_tokens = set(re.findall(r"[a-z0-9]+", ql))

    for c in class_defs:
        score = 0
        for kw in c["keywords"]:
            if kw and kw in ql:
                score += 6
        name_tokens = set(re.findall(r"[a-z0-9]+", c["name_lower"]))
        score += len(q_tokens & name_tokens) * 3
        if score > best_score:
            best_score = score
            best = c

    return best if best_score >= 6 else None

# -----------------------
# Timetable helpers
# -----------------------
DAY_ALIASES = {
    "mon": "monday", "monday": "monday",
    "tue": "tuesday", "tues": "tuesday", "tuesday": "tuesday",
    "wed": "wednesday", "weds": "wednesday", "wednesday": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday", "thursday": "thursday",
    "fri": "friday", "friday": "friday",
    "sat": "saturday", "saturday": "saturday",
    "sun": "sunday", "sunday": "sunday",
}

def normalize_day(day_val: str) -> str:
    s = str(day_val).strip().lower()
    return DAY_ALIASES.get(s, s)

def parse_time_to_minutes(t: str) -> int | None:
    if t is None:
        return None
    s = str(t).strip().lower()
    if not s or s == "nan":
        return None
    s = s.replace("—", "-").replace("–", "-")
    start = s.split("-")[0].strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", start)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None

def time_bucket_from_question(q: str) -> str | None:
    ql = q.lower()
    if "morning" in ql: return "morning"
    if "afternoon" in ql: return "afternoon"
    if "evening" in ql or "tonight" in ql: return "evening"
    return None

def bucket_range(bucket: str) -> tuple[int, int]:
    if bucket == "morning":
        return (5 * 60, 12 * 60)
    if bucket == "afternoon":
        return (12 * 60, 24 * 60)
    if bucket == "evening":
        return (12 * 60, 24 * 60)
    return (0, 24 * 60)

def weekday_name_for_date(dt: datetime) -> str:
    return dt.strftime("%A").lower()

def _standardize_timetable_from_df(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {str(c).lower().strip(): c for c in df.columns}
    if "day" in cols and "time" in cols and "class" in cols:
        out = pd.DataFrame()
        out["Day"] = df[cols["day"]]
        out["Time"] = df[cols["time"]]
        out["Class"] = df[cols["class"]]
        return out
    return None

def find_timetable_df(sheets_normal: dict) -> pd.DataFrame | None:
    for _, df in sheets_normal.items():
        if isinstance(df, pd.DataFrame):
            std = _standardize_timetable_from_df(df)
            if std is not None:
                return std
    return None

def all_class_names_from_timetable(sheets: dict) -> set[str]:
    df = find_timetable_df(sheets)
    if df is None:
        return set()

    out = set()
    for v in df["Class"].astype(str).tolist():
        s = normalize_spaces(str(v)).strip()
        if s and s.lower() != "nan":
            out.add(s.upper())
    return out

def timetable_for_day_and_bucket(sheets: dict, day: str, bucket: str | None) -> str | None:
    df = find_timetable_df(sheets)
    if df is None:
        return None

def timetable_all_classes(sheets: dict) -> str | None:
    df = find_timetable_df(sheets)
    if df is None:
        return None

    df2 = df.copy()
    df2["Day_norm"] = df2["Day"].astype(str).apply(normalize_day)
    df2["StartMin"] = df2["Time"].apply(parse_time_to_minutes)
    df2 = df2.sort_values(by=["Day_norm", "StartMin"], na_position="last")

    lines = ["Here’s our class timetable:"]
    for _, r in df2.iterrows():
        day = normalize_day(str(r["Day"])).title()
        time = str(r["Time"]).strip()
        cname = str(r["Class"]).strip()
        if cname and cname.lower() != "nan":
            lines.append(f"- {day} {time}: {cname}")

    return "\n".join(lines).strip()


def timetable_for_class_query(sheets: dict, user_q: str) -> str | None:
    """
    Generic: detect which timetable class the user is asking about, without hardcoding.
    Works by scoring overlap between user question tokens and each class name tokens.
    """
    df = find_timetable_df(sheets)
    if df is None:
        return None

    ql = normalize_spaces(user_q).lower()
    if not ql:
        return None

    q_tokens = set(re.findall(r"[a-z0-9]+", ql))
    if not q_tokens:
        return None

    # Build unique class list from timetable
    class_list = []
    seen = set()
    for v in df["Class"].astype(str).tolist():
        name = normalize_spaces(str(v)).strip()
        if not name or name.lower() == "nan":
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            class_list.append(name)

    # Score each class name against the question
    best_name = None
    best_score = 0

    for cname in class_list:
        ctokens = set(re.findall(r"[a-z0-9]+", cname.lower()))
        if not ctokens:
            continue

        overlap = len(q_tokens & ctokens)

        # Extra weight if the class name phrase appears in the question
        phrase_bonus = 3 if cname.lower() in ql else 0

        score = overlap * 4 + phrase_bonus

        if score > best_score:
            best_score = score
            best_name = cname

    # If we didn’t match enough, assume it's NOT a class-specific question
    if not best_name or best_score < 4:
        return None

    # Return all timetable rows for that matched class
    df2 = df.copy()
    df2["Class_norm"] = df2["Class"].astype(str).str.strip()
    hits = df2[df2["Class_norm"].str.lower() == best_name.lower()].copy()

    # If exact match returns nothing (e.g. spacing), fall back to contains
    if hits.empty:
        hits = df2[df2["Class_norm"].str.lower().str.contains(best_name.lower(), na=False)].copy()

    if hits.empty:
        return None

    hits["Day_norm"] = hits["Day"].astype(str).apply(normalize_day)
    hits["StartMin"] = hits["Time"].apply(parse_time_to_minutes)
    hits = hits.sort_values(by=["Day_norm", "StartMin"], na_position="last")

    lines = [f"{best_name} times:"]
    for _, r in hits.iterrows():
        day = normalize_day(str(r["Day"])).title()
        time = str(r["Time"]).strip()
        cname = str(r["Class"]).strip()
        lines.append(f"- {day} {time}: {cname}")

    return "\n".join(lines).strip()


    day_norm = normalize_day(day)
    df = df.copy()
    df["Day_norm"] = df["Day"].astype(str).apply(normalize_day)

    day_rows = df[df["Day_norm"] == day_norm].copy()
    if day_rows.empty:
        return f"No classes are listed for {day_norm.title()}."

    if bucket:
        a, b = bucket_range(bucket)
        day_rows["StartMin"] = day_rows["Time"].apply(parse_time_to_minutes)

        bucket_rows = day_rows[day_rows["StartMin"].notna()]
        bucket_rows = bucket_rows[(bucket_rows["StartMin"] >= a) & (bucket_rows["StartMin"] < b)]

        if bucket_rows.empty:
            lines = [
                f"No {bucket} classes are listed for {day_norm.title()}.",
                f"Here’s what is on {day_norm.title()}:"
            ]
            for _, r in day_rows.iterrows():
                lines.append(f"- {str(r['Time']).strip()}: {str(r['Class']).strip()}")
            return "\n".join(lines)

        day_rows = bucket_rows

    lines = [f"{day_norm.title()} classes" + (f" ({bucket})" if bucket else "") + ":"]
    for _, r in day_rows.iterrows():
        lines.append(f"- {str(r['Time']).strip()}: {str(r['Class']).strip()}")
    return "\n".join(lines)

# -----------------------
# Contextual inference for class names
# -----------------------
def infer_class_answer_from_context(user_q: str, class_names: set[str], group_link: dict | None, pt_link: dict | None):
    ql = (user_q or "").lower()
    tokens = [t.upper() for t in re.findall(r"[a-z0-9]+", ql)]
    tokens_set = set(tokens)

    mentioned = None
    for cn in sorted(class_names, key=lambda x: -len(x)):
        if cn.upper() in tokens_set:
            mentioned = cn.upper()
            break
    if not mentioned:
        return None
    if not looks_like_class_explain_intent(user_q):
        return None

    pt_like = any(x in mentioned for x in ["SGPT", "PT", "1TO1", "1:1"])
    if pt_like and pt_link and pt_link.get("summary"):
        ans = f"{mentioned} is one of our more personalised coaching options.\n{pt_link['summary']}"
        return ans, mentioned

    if group_link and group_link.get("summary"):
        ans = f"{mentioned} is one of our coach-led group sessions.\n{group_link['summary']}"
        return ans, mentioned

    ans = (
        f"{mentioned} is one of the sessions on our timetable.\n"
        "It’s a coach-led session designed to help you build fitness and stay consistent."
    )
    return ans, mentioned

# -----------------------
# AI rewrite
# -----------------------
FOLLOWUP_BANNED_HINTS = (
    "Do NOT add any follow-up line. "
    "Do NOT ask questions. "
    "Do NOT say things like 'if you’d like', 'feel free', 'just let me know', 'happy to help', or similar. "
    "End cleanly after answering."
)

def strip_accidental_followups(text: str) -> str:
    if not text:
        return text
    lines = [ln.rstrip() for ln in text.strip().splitlines() if ln.strip()]
    bad_starts = (
        "if you’d like", "if you'd like",
        "if you want", "feel free",
        "just let me know",
        "happy to help",
        "let me know",
        "want me to",
        "would you like",
        "if it helps",
    )
    while lines and lines[-1].strip().lower().startswith(bad_starts):
        lines.pop()
    return "\n".join(lines).strip()

def ai_rewrite(gym_name: str, user_question: str, raw_answer: str, *, add_follow_up: bool) -> str:
    system = (
        "You are the gym front desk receptionist (UK English). "
        "Sound natural, warm, and confident. "
        "Keep replies short (2–7 lines). "
        "Do NOT invent information. Only use the provided facts. "
        "Never say 'I'm not sure' or 'I don't know'. "
        "If you don't have something, confidently direct them to send an enquiry. "
        "IMPORTANT: If you include a URL, put it on its own line and do NOT add punctuation immediately after it. "
        + FOLLOWUP_BANNED_HINTS
    )

    user = (
        f"Gym: {gym_name}\n"
        f"User question: {user_question}\n\n"
        f"Facts:\n{raw_answer}\n\n"
        "Write the reply now:"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=260,
    )

    base = (resp.choices[0].message.content or "").strip()
    base = strip_accidental_followups(base)

    if add_follow_up:
        if base:
            return base + "\n\n" + FIRST_ANSWER_FOLLOWUP_LINE
        return FIRST_ANSWER_FOLLOWUP_LINE

    return base

# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("<h2>Gym Chatbot Server Running</h2><p>Widget: <code>/widget/evolution</code></p>")

@app.get("/widget/{gym_name}", response_class=HTMLResponse)
def widget(request: Request, gym_name: str):
    sheets = load_excel_sheets(gym_name)
    brand = load_brand(sheets, gym_name)
    return templates.TemplateResponse("widget.html", {"request": request, "gym_name": gym_name, "brand": brand})

@app.post("/chat/{gym_name}")
def chat_gym(gym_name: str, req: QuestionRequest):
    user_q = (req.question or "").strip()
    uq_lower = user_q.lower()
    session_id = req.session_id

    sheets = load_excel_sheets(gym_name)
    links = load_links(sheets)
    coaches = load_coaches(sheets)
    class_defs = load_class_defs(sheets)
    faqs = load_faq(sheets)
    pricing_rows = load_pricing(sheets)

    timetable_link = (
        find_link_by_title_contains(links, "timetable")
        or find_link_by_title_contains(links, "class timetable")
        or find_link_by_title_contains(links, "classes")
    )

    location_link = find_link_by_title_contains(links, "location")
    group_link = find_link_by_title_contains(links, "group training")
    pt_link = find_link_by_title_contains(links, "personal training")
    challenge_link = (
        find_link_by_title_contains(links, "21 day")
        or find_link_by_title_contains(links, "transformation")
    )
    team_link = (
        find_link_by_title_contains(links, "meet the team")
        or find_link_by_title_contains(links, "team")
        or find_link_by_title_contains(links, "coaches")
    )

    membership_link = (
        find_link_by_title_contains(links, "membership")
        or find_link_by_title_contains(links, "memberships")
    )

    def finalize(resp: dict):
        """
        Rewrite answer + apply first-answer-only follow-up logic,
        then mark session as answered.

        IMPORTANT: For pricing/memberships, NEVER run ai_rewrite
        (prevents hallucinated prices).
        """
        has_links = bool(resp.get("links"))
        has_cta = bool(resp.get("cta"))
        add_fu = should_add_follow_up_first_answer(session_id=session_id, has_links=has_links, has_cta=has_cta)

        # ✅ Skip AI rewrite for memberships/pricing questions
        if "answer" in resp and not is_memberships_intent(user_q):
            resp["answer"] = ai_rewrite(gym_name, user_q, resp["answer"], add_follow_up=add_fu)

        mark_answer_sent(session_id)
        return resp

    # Random/nonsense -> enquiry
    if is_random_or_nonsense(user_q):
        log_question(gym_name, user_q, "random_enquiry")
        return finalize({
            "answer": CONFIDENT_ENQUIRY_LINE,
            "source": "random_enquiry",
            "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}
        })

    # -----------------------
    # Pricing / memberships (Pricing sheet)
    # - If they ask pricing: list Pricing tab rows (or match a row)
    # - Always include Memberships button when available
    # - NEVER invent prices (AI rewrite is skipped)
    # -----------------------
    if is_memberships_intent(user_q) and pricing_rows:
        matched = match_pricing(user_q, pricing_rows)

        if matched:
            if len(matched) == 1:
                p = matched[0]
                answer = f"{p['item']}: {p['price']}"
                rows_for_buttons = matched
            else:
                answer = "Prices:\n" + "\n".join([f"- {p['item']}: {p['price']}" for p in matched])
                rows_for_buttons = matched
            source = "pricing_match"
        else:
            answer = format_pricing_list(pricing_rows)
            rows_for_buttons = pricing_rows
            source = "pricing_list"

        btns = []
        if membership_link:
            btns.append({"label": membership_link["title"], "url": membership_link["url"]})

        seen = set(b["url"] for b in btns)
        for p in rows_for_buttons:
            u = (p.get("url") or "").strip()
            if u and u not in seen:
                btns.append({"label": p["item"], "url": u})
                seen.add(u)

        log_question(gym_name, user_q, source)
        payload = {"answer": answer, "source": source}
        if btns:
            payload["links"] = btns
        return finalize(payload)

    # Coaches: specific coach
    coach_hit = match_coach(user_q, coaches)
    if coach_hit:
        raw = f"{coach_hit['name']}:\n{coach_hit['bio']}"
        resp = {"answer": raw, "source": "coach_bio", "matched": coach_hit["name"]}
        if team_link:
            resp["links"] = [{"label": team_link["title"], "url": team_link["url"]}]
        log_question(gym_name, user_q, "coach_bio")
        return finalize(resp)

    # Coaches: list
    if is_coaches_intent(user_q):
        if coaches:
            names = [c["name"] for c in coaches]
            raw = "Our coaching team includes:\n- " + "\n- ".join(names)
            resp = {"answer": raw, "source": "coaches_list"}
            if team_link:
                resp["links"] = [{"label": team_link["title"], "url": team_link["url"]}]
            log_question(gym_name, user_q, "coaches_list")
            return finalize(resp)

        log_question(gym_name, user_q, "coaches_missing")
        return finalize({
            "answer": CONFIDENT_ENQUIRY_LINE,
            "source": "coaches_missing",
            "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}
        })

    # Class defs (always include class URL button if present)
    class_hit = match_class_def(user_q, class_defs)
    if class_hit:
        raw = f"{class_hit['name']}: {class_hit['description']}"
        resp = {"answer": raw, "source": "class_def", "matched": class_hit["name"]}
        if class_hit.get("url"):
            resp["links"] = [{"label": class_hit["name"], "url": class_hit["url"]}]
        log_question(gym_name, user_q, "class_def")
        return finalize(resp)

    # Context inference for class names from timetable
    class_names = all_class_names_from_timetable(sheets)
    inferred = infer_class_answer_from_context(user_q, class_names, group_link, pt_link)
    if inferred:
        raw, cname = inferred
        resp = {"answer": raw, "source": "class_inferred", "matched": cname}
        if timetable_link and is_link_request_intent(user_q):
            resp["links"] = [{"label": timetable_link["title"], "url": timetable_link["url"]}]
        log_question(gym_name, user_q, "class_inferred")
        return finalize(resp)

    # Location (include button)
    if is_location_intent(user_q) and location_link:
        raw = location_link["summary"] or "Here are our location details."
        resp = {"answer": raw, "source": "location", "links": [{"label": location_link["title"], "url": location_link["url"]}]}
        log_question(gym_name, user_q, "location")
        return finalize(resp)

    # Timetable day-specific
    bucket = time_bucket_from_question(user_q)
    now = datetime.now()

    def day_timetable_response(day_str: str):
        raw = timetable_for_day_and_bucket(sheets, day_str, bucket) or f"No timetable available for {gym_name}."
        resp = {"answer": raw, "source": "timetable"}
        if timetable_link:
            resp["links"] = [{"label": timetable_link["title"], "url": timetable_link["url"]}]
        log_question(gym_name, user_q, "timetable")
        return finalize(resp)

    if "tomorrow" in uq_lower:
        day = weekday_name_for_date(now + timedelta(days=1))
        return day_timetable_response(day)

    if "today" in uq_lower:
        day = weekday_name_for_date(now)
        return day_timetable_response(day)

    for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday",
              "mon","tue","tues","wed","thu","thur","thurs","fri","sat"]:
        if d in uq_lower:
            return day_timetable_response(d)

        # If they ask generally about group classes / what classes / class times,
    # show the full timetable (generic).
    if any(k in uq_lower for k in ["group classes", "what classes", "class times", "class schedule", "what group", "classes do you have"]):
        raw = timetable_all_classes(sheets) or "No timetable is available right now."
        resp = {"answer": raw, "source": "timetable_all"}
        if timetable_link:
            resp["links"] = [{"label": timetable_link["title"], "url": timetable_link["url"]}]
        log_question(gym_name, user_q, "timetable_all")
        return finalize(resp)

    # Timetable: class-name lookup (generic, no hardcoding)
    # Example: "when are circuit classes", "what time is spinning", "when is boxfit"
    class_specific = timetable_for_class_query(sheets, user_q)
    if class_specific:
        resp = {"answer": class_specific, "source": "timetable_class_search"}
        if timetable_link:
            resp["links"] = [{"label": timetable_link["title"], "url": timetable_link["url"]}]
        log_question(gym_name, user_q, "timetable_class_search")
        return finalize(resp)


    # Generic timetable link (keep short + button)
    if is_timetable_intent(user_q) and timetable_link:
        log_question(gym_name, user_q, "timetable_link")
        mark_answer_sent(session_id)
        return {
            "answer": "You can view our up-to-date timetable here:",
            "source": "timetable_link",
            "links": [{"label": timetable_link["title"], "url": timetable_link["url"]}]
        }

    # 21-day challenge (include button)
    if is_challenge_intent(user_q) and challenge_link:
        raw = challenge_link["summary"] or "Here’s what the 21-day transformation programme is about."
        resp = {"answer": raw, "source": "challenge", "links": [{"label": challenge_link["title"], "url": challenge_link["url"]}]}
        log_question(gym_name, user_q, "challenge")
        return finalize(resp)

    # Classes overview (attach relevant buttons automatically)
    if is_classes_general_question(user_q) or mentions_group_classes(user_q) or mentions_personal_training(user_q):
        raw_parts = []
        buttons = []

        if group_link and group_link.get("summary"):
            raw_parts.append(f"Group training: {group_link['summary']}")
            if mentions_group_classes(user_q) or "group" in uq_lower or "classes" in uq_lower or "sessions" in uq_lower:
                buttons.append({"label": group_link["title"], "url": group_link["url"]})

        if pt_link and pt_link.get("summary"):
            raw_parts.append(f"Personal training: {pt_link['summary']}")
            if mentions_personal_training(user_q):
                buttons.append({"label": pt_link["title"], "url": pt_link["url"]})

        if challenge_link and challenge_link.get("summary"):
            raw_parts.append(f"21-day programme: {challenge_link['summary']}")
            if "challenge" in uq_lower or "21 day" in uq_lower or "transformation" in uq_lower:
                buttons.append({"label": challenge_link["title"], "url": challenge_link["url"]})

        raw = "\n\n".join([p for p in raw_parts if p]).strip() or "We offer a mix of group sessions and more personalised coaching options."
        resp = {"answer": raw, "source": "classes_overview"}

        if buttons:
            seen = set()
            uniq = []
            for b in buttons:
                if b["url"] not in seen:
                    uniq.append(b)
                    seen.add(b["url"])
            resp["links"] = uniq

        log_question(gym_name, user_q, "classes_overview")
        return finalize(resp)

    # Group vs individual (buttons)
    if is_group_vs_individual_question(user_q):
        facts = []
        buttons = []

        if group_link and group_link.get("summary"):
            facts.append(f"Group training: {group_link['summary']}")
            buttons.append({"label": group_link["title"], "url": group_link["url"]})

        if pt_link and pt_link.get("summary"):
            facts.append(f"Personal training: {pt_link['summary']}")
            buttons.append({"label": pt_link["title"], "url": pt_link["url"]})

        raw = "\n\n".join(facts).strip() or "We offer both group sessions and more personalised coaching options."
        resp = {"answer": raw, "source": "group_vs_individual"}
        if buttons:
            resp["links"] = buttons

        log_question(gym_name, user_q, "group_vs_individual")
        return finalize(resp)

    # Links knowledge fallback
    # ❌ DO NOT use for pricing
    if not is_pricing_intent(user_q):
        hit = best_link_match(user_q, links)
        if hit and hit.get("summary"):
            resp = {
                "answer": hit["summary"],
                "source": "links_knowledge",
                "matched": hit["title"],
                "links": [{"label": hit["title"], "url": hit["url"]}]
            }
            log_question(gym_name, user_q, "links_knowledge")
            return finalize(resp)

    # FAQ match
    faq_ans = faq_match(user_q, faqs)
    if faq_ans:
        log_question(gym_name, user_q, "faq_match")
        return finalize({"answer": faq_ans, "source": "faq_match"})

    # Final fallback -> enquiry
    log_question(gym_name, user_q, "fallback_enquiry")
    return finalize({
        "answer": CONFIDENT_ENQUIRY_LINE,
        "source": "fallback_enquiry",
        "cta": {
            "type": "enquiry",
            "label": "Send enquiry",
            "form_title": "Send an enquiry",
            "default_message": ""
        }
    })

@app.get("/embed.js", response_class=Response)
def embed_js(gym: str):
    base_url = "http://127.0.0.1:8000"  # change when deployed
    js = f"""
(function() {{
  var gymName = {gym!r};
  var baseUrl = {base_url!r};

  var wrap = document.createElement('div');
  wrap.style.position = 'fixed';
  wrap.style.bottom = '20px';
  wrap.style.right = '20px';
  wrap.style.zIndex = '999999';

  var btn = document.createElement('button');
  btn.innerText = 'Chat';
  btn.style.width = '64px';
  btn.style.height = '64px';
  btn.style.borderRadius = '999px';
  btn.style.border = 'none';
  btn.style.cursor = 'pointer';
  btn.style.boxShadow = '0 6px 18px rgba(0,0,0,0.2)';
  btn.style.fontSize = '16px';
  btn.style.background = '#111';
  btn.style.color = '#fff';

  var panel = document.createElement('div');
  panel.style.width = '380px';
  panel.style.height = '560px';
  panel.style.borderRadius = '16px';
  panel.style.boxShadow = '0 12px 30px rgba(0,0,0,0.25)';
  panel.style.overflow = 'hidden';
  panel.style.marginBottom = '12px';
  panel.style.display = 'none';
  panel.style.background = '#fff';

  var iframe = document.createElement('iframe');
  iframe.src = baseUrl + '/widget/' + encodeURIComponent(gymName);
  iframe.style.width = '100%';
  iframe.style.height = '100%';
  iframe.style.border = 'none';

  panel.appendChild(iframe);

  btn.onclick = function() {{
    panel.style.display = 'block';
    btn.style.display = 'none';
  }};

  wrap.appendChild(panel);
  wrap.appendChild(btn);
  document.body.appendChild(wrap);
}})();
"""
    return Response(content=js, media_type="application/javascript")
