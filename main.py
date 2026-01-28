from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
import re
import random
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

def is_timetable_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["timetable", "time table", "schedule", "class schedule", "class timetable", "class times", "session times"])

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

# -----------------------
# Follow-up variation (1 in 3)
# -----------------------
FOLLOW_UP_LINES = [
    "If you want, I can point you to the right page for this.",
    "If you’d like, I can share the relevant link as well.",
    "Happy to help — tell me what you’re aiming for and I’ll point you in the right direction.",
    "If you want more detail, just tell me what you’re looking for.",
    "If it helps, tell me your goal and I’ll suggest the best option.",
]

def should_add_follow_up(user_q: str) -> bool:
    ql = (user_q or "").lower()
    if is_timetable_intent(ql) or is_link_request_intent(ql):
        return False
    return random.random() < 0.34

def pick_follow_up_line() -> str:
    return random.choice(FOLLOW_UP_LINES)

# -----------------------
# Brand
# -----------------------
def load_brand(sheets: dict, gym_name: str) -> dict:
    brand = {
        "theme": "dark",
        "primary_color": "#ff2ea6",
        "secondary_color": "#ff2ea6",  # optional if you want it
        "gym_display_name": gym_name,
        "logo_url": "",
        "welcome_message": "Hi — welcome! How can I help?",
        "booking_url": "",

        # Powered by defaults
        "powered_by_text": "Gym Chat Bot",
        "powered_by_url": "https://gym-chat-bots.myshopify.com/?_ab=0&_fd=0&_sc=1&pb=0",
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
    brand["secondary_color"] = mapping.get("SecondaryColor", brand["secondary_color"])
    brand["gym_display_name"] = mapping.get("GymDisplayName", brand["gym_display_name"])
    brand["logo_url"] = mapping.get("LogoUrl", brand["logo_url"])
    brand["welcome_message"] = mapping.get("WelcomeMessage", brand["welcome_message"])
    brand["booking_url"] = mapping.get("BookingUrl", brand["booking_url"])

    # Powered by (optional overrides in Brand sheet)
    brand["powered_by_text"] = mapping.get("PoweredByText", brand["powered_by_text"])
    brand["powered_by_url"] = mapping.get("PoweredByUrl", brand["powered_by_url"])

    # Cleanups
    brand["welcome_message"] = titlecase_first_letter(brand["welcome_message"])
    if not str(brand["powered_by_text"]).strip():
        brand["powered_by_text"] = "Gym Chat Bot"
    if not str(brand["powered_by_url"]).strip():
        brand["powered_by_url"] = "https://gym-chat-bots.myshopify.com/?_ab=0&_fd=0&_sc=1&pb=0"

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

    # ✅ NEW (optional, controlled via Brand sheet if you want)
    brand["powered_by_text"] = mapping.get("PoweredByText", brand["powered_by_text"])
    brand["powered_by_url"] = mapping.get("PoweredByUrl", brand["powered_by_url"])

    # safety: never show an empty powered-by
    if not str(brand["powered_by_text"]).strip():
        brand["powered_by_text"] = "Gym Chat Bot"
    if not str(brand["powered_by_url"]).strip():
        brand["powered_by_url"] = "https://gym-chat-bots.myshopify.com/?_ab=0&_fd=0&_sc=1&pb=0"

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

    # ✅ Force welcome message first letter capitalized ("hi" -> "Hi")
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
        out.append({"name": name.strip(), "name_lower": name.strip().lower(), "keywords": list(dict.fromkeys(kw_list)), "description": desc, "url": url.strip()})
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
# Timetable
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
    if bucket == "morning": return (5*60, 12*60)
    if bucket == "afternoon": return (12*60, 17*60)
    if bucket == "evening": return (17*60, 24*60)
    return (0, 24*60)

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

    day_norm = normalize_day(day)
    df = df.copy()
    df["Day_norm"] = df["Day"].astype(str).apply(normalize_day)
    rows = df[df["Day_norm"] == day_norm].copy()

    if rows.empty:
        return f"No classes are listed for {day_norm.title()}."

    if bucket:
        a, b = bucket_range(bucket)
        rows["StartMin"] = rows["Time"].apply(parse_time_to_minutes)
        rows = rows[rows["StartMin"].notna()]
        rows = rows[(rows["StartMin"] >= a) & (rows["StartMin"] < b)]
        if rows.empty:
            return f"No {bucket} classes are listed for {day_norm.title()}."

    lines = [f"{day_norm.title()} classes" + (f" ({bucket})" if bucket else "") + ":"]
    for _, r in rows.iterrows():
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

    ans = f"{mentioned} is one of the sessions on our timetable.\nIt’s a coach-led session designed to help you build fitness and stay consistent."
    return ans, mentioned

# -----------------------
# AI rewrite
# -----------------------
def ai_rewrite(gym_name: str, user_question: str, raw_answer: str) -> str:
    follow_up_flag = "YES" if should_add_follow_up(user_question) else "NO"
    follow_up_line = pick_follow_up_line() if follow_up_flag == "YES" else ""

    system = (
        "You are the gym front desk receptionist (UK English). "
        "Sound natural, warm, and confident. "
        "Keep replies short (2–7 lines). "
        "Do NOT invent information. Only use the provided facts. "
        "Never say 'I'm not sure' or 'I don't know'. "
        "If you don't have something, confidently direct them to send an enquiry. "
        "IMPORTANT: If you include a URL, put it on its own line and do NOT add punctuation immediately after it.\n\n"
        "RULE: Only add a friendly follow-up line if FOLLOW_UP=YES. If FOLLOW_UP=NO, end cleanly."
    )

    user = (
        f"Gym: {gym_name}\n"
        f"FOLLOW_UP={follow_up_flag}\n"
        f"FOLLOW_UP_LINE={follow_up_line}\n"
        f"User question: {user_question}\n\n"
        f"Facts:\n{raw_answer}\n\n"
        "Reply (if FOLLOW_UP=YES, append FOLLOW_UP_LINE as the final sentence on a new line):"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.6,
        max_tokens=340,
    )
    return resp.choices[0].message.content.strip()

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

    sheets = load_excel_sheets(gym_name)
    links = load_links(sheets)
    coaches = load_coaches(sheets)
    class_defs = load_class_defs(sheets)

    timetable_link = find_link_by_title_contains(links, "timetable")
    location_link = find_link_by_title_contains(links, "location")
    group_link = find_link_by_title_contains(links, "group training")
    pt_link = find_link_by_title_contains(links, "personal training")
    challenge_link = find_link_by_title_contains(links, "21 day") or find_link_by_title_contains(links, "transformation")
    team_link = (
        find_link_by_title_contains(links, "meet the team")
        or find_link_by_title_contains(links, "team")
        or find_link_by_title_contains(links, "coaches")
    )

    if is_random_or_nonsense(user_q):
        ans = ai_rewrite(gym_name, user_q, CONFIDENT_ENQUIRY_LINE)
        log_question(gym_name, user_q, "random_enquiry")
        return {
            "answer": ans,
            "source": "random_enquiry",
            "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}
        }

    # Coaches: specific coach
    coach_hit = match_coach(user_q, coaches)
    if coach_hit:
        raw = f"{coach_hit['name']}:\n{coach_hit['bio']}"
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "coach_bio", "matched": coach_hit["name"]}
        if team_link:
            resp["links"] = [{"label": team_link["title"], "url": team_link["url"]}]
        log_question(gym_name, user_q, "coach_bio")
        return resp

    # Coaches: list
    if is_coaches_intent(user_q):
        if coaches:
            names = [c["name"] for c in coaches]
            raw = "Our coaching team includes:\n- " + "\n- ".join(names)
            ans = ai_rewrite(gym_name, user_q, raw)
            resp = {"answer": ans, "source": "coaches_list"}
            if team_link:
                resp["links"] = [{"label": team_link["title"], "url": team_link["url"]}]
            log_question(gym_name, user_q, "coaches_list")
            return resp
        ans = ai_rewrite(gym_name, user_q, CONFIDENT_ENQUIRY_LINE)
        log_question(gym_name, user_q, "coaches_missing")
        return {"answer": ans, "source": "coaches_missing", "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}}

    # Class defs first
    class_hit = match_class_def(user_q, class_defs)
    if class_hit:
        raw = f"{class_hit['name']}: {class_hit['description']}"
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "class_def", "matched": class_hit["name"]}
        if is_link_request_intent(user_q) and class_hit.get("url"):
            resp["links"] = [{"label": class_hit["name"], "url": class_hit["url"]}]
        log_question(gym_name, user_q, "class_def")
        return resp

    # Context inference for class names from timetable
    class_names = all_class_names_from_timetable(sheets)
    inferred = infer_class_answer_from_context(user_q, class_names, group_link, pt_link)
    if inferred:
        raw, cname = inferred
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "class_inferred", "matched": cname}
        log_question(gym_name, user_q, "class_inferred")
        return resp

    # Location
    if is_location_intent(user_q) and location_link:
        raw = location_link["summary"] or "Here are our location details."
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "location"}
        if is_link_request_intent(user_q):
            resp["links"] = [{"label": location_link["title"], "url": location_link["url"]}]
        log_question(gym_name, user_q, "location")
        return resp

    # Timetable day-specific
    bucket = time_bucket_from_question(user_q)
    now = datetime.now()

    def day_timetable_response(day_str: str):
        raw = timetable_for_day_and_bucket(sheets, day_str, bucket) or f"No timetable available for {gym_name}."
        resp = {"answer": raw, "source": "timetable"}
        if is_link_request_intent(user_q) and timetable_link:
            resp["links"] = [{"label": timetable_link["title"], "url": timetable_link["url"]}]
        log_question(gym_name, user_q, "timetable")
        return resp

    if "tomorrow" in uq_lower:
        day = weekday_name_for_date(now + timedelta(days=1))
        return day_timetable_response(day)

    if "today" in uq_lower:
        day = weekday_name_for_date(now)
        return day_timetable_response(day)

    for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday",
              "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun"]:
        if d in uq_lower:
            return day_timetable_response(d)

    # 21-day: ALWAYS include button
    if is_challenge_intent(user_q) and challenge_link:
        raw = challenge_link["summary"] or "Here’s what the 21-day transformation programme is about."
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "challenge"}
        resp["links"] = [{"label": challenge_link["title"], "url": challenge_link["url"]}]
        log_question(gym_name, user_q, "challenge")
        return resp

    # Classes overview
    if is_classes_general_question(user_q):
        raw_parts = []
        if group_link and group_link.get("summary"):
            raw_parts.append(f"Group training: {group_link['summary']}")
        if pt_link and pt_link.get("summary"):
            raw_parts.append(f"Personal training: {pt_link['summary']}")
        if challenge_link and challenge_link.get("summary"):
            raw_parts.append(f"21-day programme: {challenge_link['summary']}")
        raw = "\n\n".join([p for p in raw_parts if p]).strip() or "We offer a mix of group sessions and more personalised coaching options."
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "classes_overview"}
        log_question(gym_name, user_q, "classes_overview")
        return resp

    # Group vs individual
    if is_group_vs_individual_question(user_q):
        facts = []
        if group_link and group_link.get("summary"):
            facts.append(f"Group training: {group_link['summary']}")
        if pt_link and pt_link.get("summary"):
            facts.append(f"Personal training: {pt_link['summary']}")
        raw = "\n\n".join(facts).strip() or "We offer both group sessions and more personalised coaching options."
        ans = ai_rewrite(gym_name, user_q, raw)
        resp = {"answer": ans, "source": "group_vs_individual"}
        log_question(gym_name, user_q, "group_vs_individual")
        return resp

    # Links knowledge fallback
    hit = best_link_match(user_q, links)
    if hit and hit.get("summary"):
        ans = ai_rewrite(gym_name, user_q, hit["summary"])
        resp = {"answer": ans, "source": "links_knowledge", "matched": hit["title"]}
        if is_link_request_intent(user_q):
            resp["links"] = [{"label": hit["title"], "url": hit["url"]}]
        log_question(gym_name, user_q, "links_knowledge")
        return resp

    # Final fallback -> enquiry
    ans = ai_rewrite(gym_name, user_q, CONFIDENT_ENQUIRY_LINE)
    log_question(gym_name, user_q, "fallback_enquiry")
    return {
        "answer": ans,
        "source": "fallback_enquiry",
        "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}
    }

@app.get("/embed.js", response_class=Response)
def embed_js(request: Request, gym: str):
    base_url = str(request.base_url).rstrip("/")

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


