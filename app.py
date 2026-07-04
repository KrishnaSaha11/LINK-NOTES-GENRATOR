"""
NoteGenius AI — Flask backend
=============================

One endpoint:  POST /summarize   { "url": "<youtube-or-article-link>" }

Pipeline:
  1. TRANSCRIPT LAYER — figure out what the link is and extract raw text:
       - YouTube  -> youtube-transcript-api
       - Article  -> requests + BeautifulSoup
  2. AI LAYER — send the text to Groq (llama-3.3-70b-versatile),
     chunking first if the text is very long.
  3. Normalize the model's JSON and return it to the frontend.

Every failure path returns JSON like  { "error": "human readable message" }
with an appropriate HTTP status code, so the frontend can show it directly.
"""

import json
import os
import re

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env from the project root — never hardcode the key

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

app = Flask(__name__)
CORS(app)  # allow the frontend (any local origin / file://) to call us

# The exact instruction the model receives (per spec — do not reword).
SUMMARY_INSTRUCTION = (
    "Extract from this content: (1) a 3-line QUICK SUMMARY, "
    "(2) 5-6 KEY POINTS each with one line of explanation, "
    "(3) 3-5 ACTION ITEMS. Return strictly as JSON with keys: "
    "summary, key_points, action_items."
)

# Chunking limits (measured in characters; ~4 chars ≈ 1 token).
# A single request under this size fits comfortably in one Groq call
# even on rate-limited free tiers.
CHUNK_SIZE = 20_000          # max characters per model call
MAX_TOTAL_CHARS = 400_000    # hard cap so a huge page can't hang the server

# A browser-like User-Agent — many sites block Python's default one.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


class ExtractionError(Exception):
    """Raised by the transcript layer with a user-facing message."""

    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# 1. TRANSCRIPT LAYER
# ---------------------------------------------------------------------------

# Matches every common YouTube URL shape and captures the 11-char video id:
#   youtube.com/watch?v=ID · youtu.be/ID · youtube.com/shorts/ID
#   youtube.com/embed/ID   · youtube.com/live/ID
YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)


def extract_youtube_id(url):
    """Return the video id if `url` is a YouTube link, else None."""
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def fetch_youtube_transcript(video_id):
    """Fetch a video's transcript text, preferring English.

    Raises ExtractionError with a clear message for every known failure
    (no captions, private/removed video, etc.).
    """
    api = YouTubeTranscriptApi()
    try:
        try:
            transcript = api.fetch(video_id, languages=["en"])
        except NoTranscriptFound:
            # No English track — fall back to the first available language.
            available = api.list(video_id)
            transcript = next(iter(available)).fetch()
    except TranscriptsDisabled:
        raise ExtractionError(
            "This video has captions disabled, so there is no transcript to read."
        )
    except (NoTranscriptFound, StopIteration):
        raise ExtractionError("No transcript is available for this video.")
    except VideoUnavailable:
        raise ExtractionError(
            "This video is unavailable — it may be private, removed, or region-locked."
        )
    except CouldNotRetrieveTranscript:
        raise ExtractionError(
            "Couldn't retrieve a transcript for this video. It may be private "
            "or have no captions."
        )

    # A fetched transcript iterates as snippets; join them into one string.
    text = " ".join(snippet.text.strip() for snippet in transcript)
    return re.sub(r"\s+", " ", text).strip()


def fetch_youtube_title(url):
    """Best-effort video title via YouTube's public oEmbed endpoint."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            headers=HTTP_HEADERS,
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("title")
    except requests.RequestException:
        pass
    return None


def fetch_article_text(url):
    """Download an article page and return (clean_text, title).

    Strategy: strip obvious non-content tags, prefer <article>/<main>,
    then keep paragraph-like elements with real sentences in them.
    """
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.MissingSchema:
        raise ExtractionError("That doesn't look like a valid URL.", status=400)
    except requests.exceptions.ConnectionError:
        raise ExtractionError("Couldn't reach that URL — check the address and try again.")
    except requests.exceptions.Timeout:
        raise ExtractionError("The page took too long to respond. Try again later.")
    except requests.exceptions.HTTPError:
        raise ExtractionError(
            f"The page returned an error (HTTP {resp.status_code}) — "
            "it may be paywalled or blocked."
        )

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and "xml" not in content_type:
        raise ExtractionError("That URL doesn't point to a readable web page.")

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None

    # Remove everything that is never article content.
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Prefer semantic containers; fall back to the whole body.
    main = soup.find("article") or soup.find("main") or soup.body or soup

    # Keep paragraph-ish blocks that contain actual sentences.
    blocks = [
        el.get_text(" ", strip=True)
        for el in main.find_all(["p", "h1", "h2", "h3", "li", "blockquote"])
    ]
    text = "\n".join(b for b in blocks if len(b.split()) > 3)

    # Some pages don't use <p> tags at all — fall back to raw text.
    if len(text) < 400:
        text = re.sub(r"\s+", " ", main.get_text(" ", strip=True))

    if len(text.split()) < 60:
        raise ExtractionError(
            "Couldn't extract readable text from that page — it may be "
            "paywalled, JavaScript-rendered, or not an article."
        )
    return text.strip(), title


# ---------------------------------------------------------------------------
# 2. AI LAYER (Groq)
# ---------------------------------------------------------------------------

def groq_client():
    if not GROQ_API_KEY:
        raise ExtractionError(
            "Server is missing its GROQ_API_KEY — add it to the .env file.",
            status=500,
        )
    return Groq(api_key=GROQ_API_KEY)


def groq_chat(client, prompt, force_json=False):
    """One chat completion; optionally force valid-JSON output."""
    kwargs = {"response_format": {"type": "json_object"}} if force_json else {}
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        **kwargs,
    )
    return completion.choices[0].message.content


def split_into_chunks(text, size=CHUNK_SIZE):
    """Split text into ~`size`-char chunks, breaking on sentence ends
    so no chunk starts mid-sentence."""
    if len(text) <= size:
        return [text]
    chunks, current = [], []
    current_len = 0
    # Split on sentence boundaries ("...end. Next") to get natural pieces.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current_len + len(sentence) > size and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_text(text):
    """Run the full AI pipeline and return the parsed dict from the model.

    Short text  -> one call with SUMMARY_INSTRUCTION.
    Long text   -> summarize each chunk, then run SUMMARY_INSTRUCTION
                   over the combined chunk summaries (map-reduce).
    """
    client = groq_client()
    text = text[:MAX_TOTAL_CHARS]
    chunks = split_into_chunks(text)

    if len(chunks) > 1:
        # MAP: condense each chunk while keeping the important facts.
        partial_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            partial = groq_chat(
                client,
                "The following is section "
                f"{i} of {len(chunks)} of a longer document. Summarize it in "
                "detail (10-15 sentences), preserving all key facts, names, "
                "numbers and recommendations:\n\n" + chunk,
            )
            partial_summaries.append(partial)
        # REDUCE: the exact instruction now runs over the combined summaries.
        text = "\n\n".join(partial_summaries)

    raw = groq_chat(client, f"{SUMMARY_INSTRUCTION}\n\nCONTENT:\n{text}",
                    force_json=True)
    return parse_model_json(raw)


def parse_model_json(raw):
    """Parse the model's reply into a dict, tolerating ```json fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ExtractionError(
            "The AI returned a response that couldn't be parsed. Please try again.",
            status=502,
        )


def normalize_notes(data):
    """Coerce the model's JSON into the stable shape the frontend expects:

        summary      -> string
        key_points   -> [ { "title": str, "explanation": str }, ... ]
        action_items -> [ str, ... ]

    LLMs sometimes vary key names or return strings instead of objects,
    so we normalize defensively instead of trusting the shape.
    """
    summary = data.get("summary", "")
    if isinstance(summary, list):  # occasionally returned as 3 lines
        summary = " ".join(str(line) for line in summary)

    key_points = []
    for item in data.get("key_points", []):
        if isinstance(item, dict):
            title = item.get("point") or item.get("title") or item.get("key_point") or ""
            explanation = (item.get("explanation") or item.get("description")
                           or item.get("detail") or "")
        else:
            # A plain string — split "Title: explanation" if possible.
            title, _, explanation = str(item).partition(":")
            if not explanation:
                title, _, explanation = str(item).partition("—")
        title, explanation = title.strip(), explanation.strip()
        if title:
            key_points.append({"title": title, "explanation": explanation})

    action_items = [
        (item.get("action") or item.get("item") or json.dumps(item))
        if isinstance(item, dict) else str(item)
        for item in data.get("action_items", [])
    ]

    if not summary or not key_points:
        raise ExtractionError(
            "The AI response was missing expected fields. Please try again.",
            status=502,
        )
    return {"summary": str(summary).strip(),
            "key_points": key_points,
            "action_items": action_items}


# ---------------------------------------------------------------------------
# 3. ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/summarize")
def summarize():
    # -- validate input ------------------------------------------------
    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify(error="Missing 'url' in request body."), 400
    if not re.match(r"^https?://", url):
        url = "https://" + url  # be forgiving about a missing scheme

    # -- transcript layer ----------------------------------------------
    try:
        video_id = extract_youtube_id(url)
        if video_id:
            text = fetch_youtube_transcript(video_id)
            source = {
                "type": "youtube",
                "title": fetch_youtube_title(url) or "YouTube video",
            }
        else:
            text, title = fetch_article_text(url)
            source = {"type": "article", "title": title or "Article"}
        source["word_count"] = len(text.split())

        # -- AI layer ----------------------------------------------------
        notes = summarize_text(text)
        notes = normalize_notes(notes)

    except ExtractionError as err:
        return jsonify(error=str(err)), err.status
    except Exception:  # anything unexpected — never leak a stack trace
        app.logger.exception("summarize failed")
        return jsonify(error="Something went wrong on the server. Please try again."), 500

    # -- respond ---------------------------------------------------------
    return jsonify({"source": source, **notes})


@app.get("/health")
def health():
    """Quick check that the server is up and the API key is loaded."""
    return jsonify(status="ok", groq_key_loaded=bool(GROQ_API_KEY))


if __name__ == "__main__":
    # Port 5000 matches the frontend's fetch() target.
    app.run(host="127.0.0.1", port=5000, debug=True)
