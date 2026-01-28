from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
import re
import random
from datetime import datetime

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

# Optional: serve static assets (like /static/logo.png)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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

def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

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
# Brand / Theme
# -----------------------
def load_brand(sheets: dict, gym_name: str) -> dict:
    brand = {
        "theme": "dark",
        "primary_color": "#ff2ea6",
        "secondary_color": "#2563eb",
        "gym_display_name": gym_name,
        "logo_url": "",
        "welcome_message": "Hi — welcome! How can I help?",
        "bot_mode": "gym",  # gym or demo
        "powered_by_text": "Powered by Gym Chat Bot",
        "powered_by_url": "https://gym-chat-bots.myshopify.com/?_ab=0&_fd=0&_sc=1&pb=0",

    }

    if "Brand" not in sheets:
        return brand

    df = sheets["Brand"]
    if "Key" not in df.columns or "Value" not in df.columns:
        return brand

    mapping = {}
    for _, row in df.iterrows():
        k = str(row.get("Key", "")).strip()
        v = str(row.get("Value", "")).strip()
        if k and v and k.lower() != "nan" and v.lower() != "nan":
            mapping[k] = v

    brand["theme"] = mapping.get("Theme", brand["theme"]).strip().lower()
    if brand["theme"] not in ("dark", "light"):
        brand["theme"] = "dark"

    brand["primary_color"] = mapping.get("PrimaryColor", brand["primary_color"]).strip()
    brand["secondary_color"] = mapping.get("SecondaryColor", brand["secondary_color"]).strip()
    brand["gym_display_name"] = mapping.get("GymDisplayName", brand["gym_display_name"]).strip()
    brand["logo_url"] = mapping.get("LogoUrl", brand["logo_url"]).strip()
    brand["welcome_message"] = mapping.get("WelcomeMessage", brand["welcome_message"]).strip()
    brand["bot_mode"] = mapping.get("BotMode", brand["bot_mode"]).strip().lower() or "gym"
    if brand["bot_mode"] not in ("gym", "demo"):
        brand["bot_mode"] = "gym"

    brand["powered_by_text"] = mapping.get("PoweredByText", "").strip()
    brand["powered_by_url"] = mapping.get("PoweredByUrl", "").strip()

    return brand


# -----------------------
# Links
# -----------------------
def load_links(sheets: dict) -> list[dict]:
    if "Links" not in sheets:
        return []
    df = sheets["Links"].copy()
    for c in ["Title", "URL", "Keywords", "Summary"]:
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

def best_link_match(user_q: str, links: list[dict]) -> tuple[dict | None, float]:
    ql = (user_q or "").lower()
    if not ql:
        return None, 0.0

    best = None
    best_score = 0.0
    q_tokens = set(tokenize(ql))

    for item in links:
        score = 0.0
        for kw in item["keywords"]:
            if kw and kw in ql:
                score += 5.0

        title_tokens = set(tokenize(item["title"]))
        score += len(title_tokens & q_tokens) * 1.5

        if score > best_score:
            best_score = score
            best = item

    conf = min(1.0, best_score / 10.0) if best else 0.0
    return best, conf


# -----------------------
# FAQ
# -----------------------
def load_faq(sheets: dict) -> list[dict]:
    if "FAQ" not in sheets:
        return []
    df = sheets["FAQ"].copy()
    if "Question" not in df.columns or "Answer" not in df.columns:
        return []
    items = []
    for _, r in df.iterrows():
        q = normalize_spaces(str(r.get("Question", "")).strip())
        a = clean_summary_text(r.get("Answer", ""))
        if not q or q.lower() == "nan":
            continue
        if not a:
            continue
        items.append({"q": q, "a": a, "q_tokens": set(tokenize(q))})
    return items

def best_faq_match(user_q: str, faq: list[dict]) -> tuple[dict | None, float]:
    uq = normalize_spaces(user_q).strip()
    if not uq:
        return None, 0.0

    uq_low = uq.lower()
    uq_tokens = set(tokenize(uq_low))

    best = None
    best_conf = 0.0

    for item in faq:
        q_text = item["q"]
        q_low = q_text.lower()

        if uq_low == q_low:
            return item, 1.0

        if uq_low in q_low or q_low in uq_low:
            conf = 0.85
        else:
            conf = jaccard(uq_tokens, item["q_tokens"])

        if conf > best_conf:
            best_conf = conf
            best = item

    return best, best_conf


# -----------------------
# Intents
# -----------------------
def is_link_request_intent(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["link", "website", "page", "url", "send me", "where can i find", "where do i find", "more info"])

def wants_signup(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["sign me up", "signup", "sign up", "get started", "start now", "buy", "purchase", "i want this", "im interested", "i'm interested"])

def wants_pricing(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ["price", "pricing", "cost", "how much", "monthly", "per month", "plans"])


# -----------------------
# Fallback (varied)
# -----------------------
FALLBACK_ENQUIRY_LINES = [
    "I want to make sure you get the right answer. The best next step is to send us an enquiry and the team will get back to you shortly.",
    "To make sure this is handled properly, please send an enquiry and the team will get back to you shortly.",
    "The quickest way to get the right help is to send an enquiry — the team will get back to you shortly.",
    "To get you an accurate answer, please send an enquiry and the team will get back to you shortly.",
]

def pick_fallback_line() -> str:
    return random.choice(FALLBACK_ENQUIRY_LINES)


# -----------------------
# AI rewrite (STRICT: only use provided facts)
# -----------------------
def ai_rewrite(gym_name: str, user_question: str, facts: str) -> str:
    system = (
        "You are a helpful assistant. Use UK English. Keep replies short.\n"
        "CRITICAL:\n"
        "- ONLY use the FACTS provided. Do not add extra examples, features, prices, industries, or claims.\n"
        "- Do NOT invent details.\n"
        "- Never say 'I'm not sure' or 'I don't know'.\n"
        "- If FACTS contain the answer, answer directly.\n"
        "- If FACTS do NOT contain the answer, reply with exactly this sentence:\n"
        "  [[FALLBACK_SENTENCE]]\n"
        "- If you include a URL, put it on its own line with NO punctuation immediately after it.\n"
    )

    fallback_sentence = pick_fallback_line()
    system = system.replace("[[FALLBACK_SENTENCE]]", fallback_sentence)

    user = (
        f"Gym: {gym_name}\n"
        f"User question: {user_question}\n\n"
        f"FACTS:\n{facts}\n\n"
        "Write the reply now:"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=260,
    )
    return resp.choices[0].message.content.strip()


# -----------------------
# Routes
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("<h3>Server running</h3><p>Open /widget/evolution or /widget/demo</p>")

@app.get("/widget/{gym_name}", response_class=HTMLResponse)
def widget(request: Request, gym_name: str):
    sheets = load_excel_sheets(gym_name)
    brand = load_brand(sheets, gym_name)
    return templates.TemplateResponse("widget.html", {"request": request, "gym_name": gym_name, "brand": brand})

@app.post("/chat/{gym_name}")
def chat_gym(gym_name: str, req: QuestionRequest):
    user_q = (req.question or "").strip()
    sheets = load_excel_sheets(gym_name)
    brand = load_brand(sheets, gym_name)

    links = load_links(sheets)
    faq = load_faq(sheets)

    pricing_link = (
        find_link_by_title_contains(links, "pricing")
        or find_link_by_title_contains(links, "plans")
        or find_link_by_title_contains(links, "packages")
    )

    # Nonsense -> enquiry CTA
    if is_random_or_nonsense(user_q):
        ans = ai_rewrite(gym_name, user_q, "")
        log_question(gym_name, user_q, "random_enquiry")
        return {
            "answer": ans,
            "source": "random_enquiry",
            "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}
        }

    # "Sign me up" -> pricing button if available
    if wants_signup(user_q) and pricing_link:
        log_question(gym_name, user_q, "signup_pricing")
        return {
            "answer": "Great — the easiest next step is to view pricing and choose a plan.",
            "source": "signup_pricing",
            "links": [{"label": pricing_link["title"], "url": pricing_link["url"]}]
        }

    # FAQ match
    faq_hit, faq_conf = best_faq_match(user_q, faq)
    threshold = 0.30 if brand.get("bot_mode") == "demo" else 0.55
    if faq_hit and faq_conf >= threshold:
        ans = ai_rewrite(gym_name, user_q, faq_hit["a"])
        resp = {"answer": ans, "source": "faq", "matched": faq_hit["q"]}
        log_question(gym_name, user_q, f"faq_{faq_conf:.2f}")

        # only show pricing button when pricing intent
        if wants_pricing(user_q) and pricing_link:
            resp["links"] = [{"label": pricing_link["title"], "url": pricing_link["url"]}]
        return resp

    # Links knowledge match
    link_hit, link_conf = best_link_match(user_q, links)
    link_threshold = 0.25 if brand.get("bot_mode") == "demo" else 0.40
    if link_hit and link_hit.get("summary") and link_conf >= link_threshold:
        ans = ai_rewrite(gym_name, user_q, link_hit["summary"])
        resp = {"answer": ans, "source": "links_knowledge", "matched": link_hit["title"]}
        log_question(gym_name, user_q, f"links_{link_conf:.2f}")

        # show link button only if user asked for a link / more info / pricing
        if is_link_request_intent(user_q) or wants_pricing(user_q):
            resp["links"] = [{"label": link_hit["title"], "url": link_hit["url"]}]
        return resp

    # Final fallback -> enquiry CTA
    ans = ai_rewrite(gym_name, user_q, "")
    log_question(gym_name, user_q, "fallback_enquiry")
    return {
        "answer": ans,
        "source": "fallback_enquiry",
        "cta": {"type": "enquiry", "label": "Send enquiry", "form_title": "Send an enquiry", "default_message": ""}
    }


# -----------------------
# ✅ EMBED SCRIPT (FIXED)
# Works on localhost AND when deployed (Render, etc.)
# -----------------------
@app.get("/embed.js", response_class=Response)
def embed_js(request: Request, gym: str = ""):
    base_url = str(request.base_url).rstrip("/")

    js = f"""
(function() {{
  var params = new URLSearchParams(window.location.search);
  var gymName = params.get('gym') || {gym!r};
  var baseUrl = {base_url!r};

  if (!gymName) {{
    console.error('[Gym Chat Bot] Missing gym name. Use ?gym=YOUR_GYM');
    return;
  }}

  var wrap = document.createElement('div');
  wrap.style.position = 'fixed';
  wrap.style.bottom = '20px';
  wrap.style.right = '20px';
  wrap.style.zIndex = '999999';
  wrap.style.fontFamily = 'ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial';

  var panel = document.createElement('div');
  panel.style.width = '380px';
  panel.style.height = '560px';
  panel.style.borderRadius = '18px';
  panel.style.boxShadow = '0 18px 50px rgba(0,0,0,0.28)';
  panel.style.overflow = 'hidden';
  panel.style.marginBottom = '12px';
  panel.style.display = 'none';
  panel.style.background = '#ffffff';
  panel.style.position = 'relative';

  var closeBtn = document.createElement('button');
  closeBtn.setAttribute('aria-label', 'Close chat');
  closeBtn.innerHTML = '&times;';
  closeBtn.style.position = 'absolute';
  closeBtn.style.top = '10px';
  closeBtn.style.right = '10px';
  closeBtn.style.width = '34px';
  closeBtn.style.height = '34px';
  closeBtn.style.borderRadius = '999px';
  closeBtn.style.border = 'none';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.background = 'rgba(0,0,0,0.10)';
  closeBtn.style.backdropFilter = 'blur(6px)';
  closeBtn.style.fontSize = '22px';
  closeBtn.style.lineHeight = '34px';
  closeBtn.style.textAlign = 'center';
  closeBtn.style.color = '#111';
  closeBtn.style.zIndex = '2';

  var iframe = document.createElement('iframe');
  iframe.src = baseUrl + '/widget/' + encodeURIComponent(gymName);
  iframe.style.width = '100%';
  iframe.style.height = '100%';
  iframe.style.border = 'none';

  panel.appendChild(iframe);
  panel.appendChild(closeBtn);

  var btn = document.createElement('button');
  btn.setAttribute('aria-label', 'Open chat');
  btn.style.width = '58px';
  btn.style.height = '58px';
  btn.style.borderRadius = '999px';
  btn.style.border = 'none';
  btn.style.cursor = 'pointer';
  btn.style.boxShadow = '0 14px 30px rgba(0,0,0,0.25)';
  btn.style.display = 'flex';
  btn.style.alignItems = 'center';
  btn.style.justifyContent = 'center';
  btn.style.background = 'linear-gradient(135deg, #111827 0%, #0b1220 55%, #111827 100%)';
  btn.style.color = '#fff';

  btn.innerHTML = `
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M21 12a8 8 0 0 1-8 8H8l-5 3 1.5-4.5A8 8 0 1 1 21 12Z"
        stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
      <path d="M8 12h.01M12 12h.01M16 12h.01"
        stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>
  `;

  function openPanel() {{
    panel.style.display = 'block';
    btn.style.display = 'none';
  }}

  function closePanel() {{
    panel.style.display = 'none';
    btn.style.display = 'flex';
  }}

  btn.onclick = openPanel;
  closeBtn.onclick = closePanel;

  document.addEventListener('keydown', function(e) {{
    if (e.key === 'Escape' && panel.style.display === 'block') {{
      closePanel();
    }}
  }});

  wrap.appendChild(panel);
  wrap.appendChild(btn);
  document.body.appendChild(wrap);
}})();
"""
    return Response(content=js, media_type="application/javascript")



