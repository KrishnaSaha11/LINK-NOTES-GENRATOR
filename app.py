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
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq, RateLimitError
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

# The exact instructions the model receives (per spec — do not reword).
SUMMARY_INSTRUCTION = (
    "Extract from this content: (1) a 3-line QUICK SUMMARY, "
    "(2) 5-6 KEY POINTS each with one line of explanation, "
    "(3) 3-5 ACTION ITEMS. Return strictly as JSON with keys: "
    "summary, key_points, action_items."
)
ASK_INSTRUCTION = (
    "You are answering follow-up questions about this content. Use ONLY the "
    "provided context. Answer clearly and concisely (max 5-6 lines). If the "
    "answer isn't in the content, say so honestly."
)

# Chunking / rate limits (sizes in characters).
# Groq's free tier allows 12,000 tokens per minute (TPM), and its token
# estimator is conservative on caption-style text — so each request must
# stay WELL under that. 6,000 chars keeps even worst-case estimates in
# the few-thousand-token range, and MAX_COMPLETION caps the output
# allowance Groq adds to the estimate.
CHUNK_SIZE = 6_000           # max characters per model call
MAX_COMPLETION = 1_024       # max output tokens per call
CHUNK_DELAY = 20             # seconds between chunked Groq calls (TPM pacing)
RATE_LIMIT_COOLDOWN = 30     # seconds to wait before retrying a 429
MAX_TOTAL_CHARS = 400_000    # hard cap so a huge page can't hang the server
MAX_ASK_CONTEXT = 8_000      # max context characters accepted by /ask
MAX_HISTORY_PAIRS = 3        # Q&A pairs of chat history kept per /ask call

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


def groq_chat(client, prompt_or_messages, force_json=False):
    """One chat completion; optionally force valid-JSON output.

    Accepts either a plain prompt string or a full messages list
    (used by /ask to include the system prompt and chat history).
    If Groq reports the per-minute token limit is exhausted (429),
    waits once and retries before giving up.
    """
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages
    kwargs = {"response_format": {"type": "json_object"}} if force_json else {}
    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=MAX_COMPLETION,
                **kwargs,
            )
            return completion.choices[0].message.content
        except RateLimitError:
            if attempt:  # already retried once — let the endpoint handle it
                raise
            time.sleep(RATE_LIMIT_COOLDOWN)


def split_into_chunks(text, size=CHUNK_SIZE):
    """Split text into ~`size`-char chunks, breaking on sentence ends
    so no chunk starts mid-sentence.

    Important: YouTube auto-captions often have NO punctuation at all,
    which makes the whole transcript one giant "sentence" — so any
    oversized piece is additionally hard-split on word boundaries.
    Without this, a long unpunctuated transcript went to Groq as a
    single enormous request (the source of 413 rate_limit_exceeded).
    """
    if len(text) <= size:
        return [text]

    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > size:  # unpunctuated run — split on a space
            cut = sentence.rfind(" ", size // 2, size)
            if cut == -1:
                cut = size
            pieces.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if sentence:
            pieces.append(sentence)

    chunks, current, current_len = [], [], 0
    for piece in pieces:
        if current_len + len(piece) > size and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(piece)
        current_len += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def summarize_text(text):
    """Run the full AI pipeline. Returns (parsed_notes, context_used).

    Short text  -> one call with SUMMARY_INSTRUCTION.
    Long text   -> summarize each chunk, then run SUMMARY_INSTRUCTION
                   over the combined chunk summaries (map-reduce).

    `context_used` is what the final call actually saw — the full text
    for short content, or the combined chunk summaries for long content.
    The frontend stores it and sends it back with /ask questions, so
    follow-ups on long content automatically use the compact summaries
    instead of the oversized full transcript.
    """
    client = groq_client()
    text = text[:MAX_TOTAL_CHARS]
    made_calls = False

    # MAP (possibly repeated): while the text is too big for one call,
    # summarize it chunk by chunk and continue with the combined
    # summaries. Very long videos may need a second pass — 34 chunk
    # summaries can themselves exceed one request. CHUNK_DELAY pacing
    # between calls keeps us inside Groq's tokens-per-minute budget.
    passes = 0
    while len(text) > CHUNK_SIZE and passes < 3:
        chunks = split_into_chunks(text)
        partial_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            if made_calls:
                time.sleep(CHUNK_DELAY)
            partial_summaries.append(groq_chat(
                client,
                "The following is section "
                f"{i} of {len(chunks)} of a longer document. Summarize it in "
                "detail (8-12 sentences), preserving all key facts, names, "
                "numbers and recommendations:\n\n" + chunk,
            ))
            made_calls = True
        text = "\n\n".join(partial_summaries)
        passes += 1

    # REDUCE: the exact instruction runs over what's left (the original
    # text for short content, or the combined chunk summaries).
    if made_calls:
        time.sleep(CHUNK_DELAY)
    raw = groq_chat(client, f"{SUMMARY_INSTRUCTION}\n\nCONTENT:\n{text}",
                    force_json=True)
    return parse_model_json(raw), text


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
        notes, context_used = summarize_text(text)
        notes = normalize_notes(notes)

    except ExtractionError as err:
        return jsonify(error=str(err)), err.status
    except RateLimitError:
        return jsonify(
            error="The AI service hit its per-minute rate limit even after "
                  "waiting. Give it a minute and try again."
        ), 429
    except Exception:  # anything unexpected — never leak a stack trace
        app.logger.exception("summarize failed")
        return jsonify(error="Something went wrong on the server. Please try again."), 500

    # -- respond ---------------------------------------------------------
    # `context` is echoed back by the frontend on /ask calls (see
    # summarize_text docstring).
    return jsonify({"source": source, "context": context_used, **notes})


@app.post("/ask")
def ask():
    """Answer a follow-up question about previously summarized content.

    Body: {
      "question": "...",
      "context":  "<text returned by /summarize as 'context'>",
      "notes":    <the generated notes JSON (string or object)>,
      "history":  [ { "question": "...", "answer": "..." }, ... ]  # optional
    }
    """
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    context = (body.get("context") or "").strip()
    if not question:
        return jsonify(error="Missing 'question' in request body."), 400
    if not context:
        return jsonify(error="Missing 'context' — generate notes for a link first."), 400

    # Cap the context; /summarize already keeps it compact for long
    # content (chunk summaries), so this only trims pathological input.
    context = context[:MAX_ASK_CONTEXT]

    notes = body.get("notes") or ""
    if not isinstance(notes, str):
        notes = json.dumps(notes)

    # System message: the exact instruction + everything the model may use.
    system = f"{ASK_INSTRUCTION}\n\nCONTEXT:\n{context}"
    if notes:
        system += f"\n\nGENERATED NOTES:\n{notes}"
    messages = [{"role": "system", "content": system}]

    # Replay the last few Q&A pairs so follow-ups can reference earlier
    # answers ("what did you mean by that?").
    history = body.get("history") or []
    for pair in history[-MAX_HISTORY_PAIRS:]:
        if isinstance(pair, dict) and pair.get("question") and pair.get("answer"):
            messages.append({"role": "user", "content": str(pair["question"])[:2000]})
            messages.append({"role": "assistant", "content": str(pair["answer"])[:2000]})

    messages.append({"role": "user", "content": question[:2000]})

    try:
        answer = groq_chat(groq_client(), messages).strip()
    except ExtractionError as err:
        return jsonify(error=str(err)), err.status
    except RateLimitError:
        return jsonify(
            error="The AI service hit its per-minute rate limit. "
                  "Give it a minute and ask again."
        ), 429
    except Exception:
        app.logger.exception("ask failed")
        return jsonify(
            error="Couldn't get an answer right now. Please try again in a moment."
        ), 502

    return jsonify(answer=answer)


@app.get("/health")
def health():
    """Quick check that the server is up and the API key is loaded."""
    return jsonify(status="ok", groq_key_loaded=bool(GROQ_API_KEY))


if __name__ == "__main__":
    # Port 5000 matches the frontend's fetch() target.
    app.run(host="127.0.0.1", port=5000, debug=True)
