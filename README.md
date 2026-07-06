# 🧠 NoteGenius AI

Turn any YouTube video or article link into structured notes — quick summary,
key points, action items, and a mind map.

- **Frontend** — `index.html`: a single-file, zero-dependency dark-themed SPA.
- **Backend** — `app.py`: a Flask API that extracts the transcript/article text
  and summarizes it with **Groq** (`llama-3.3-70b-versatile`).

## Setup

### 1. Backend

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your Groq API key (free at https://console.groq.com/keys)
cp .env.example .env
# ...then edit .env and paste your real key

# Run the server (http://localhost:5000)
python app.py
```

Sanity check: open <http://localhost:5000/health> — you should see
`{"status": "ok", "groq_key_loaded": true}`.

### 2. Frontend

Serve `index.html` from any static server (opening the file directly also
works, since the backend allows all origins):

```bash
python -m http.server 8000
# then open http://localhost:8000
```

Paste a YouTube or article link and hit **Generate Notes ✨**.

## API

### `POST /summarize`

```json
{ "url": "https://www.youtube.com/watch?v=..." }
```

**Success (200):**

```json
{
  "source": { "type": "youtube", "title": "Video title", "word_count": 5230 },
  "context": "<the text the AI actually summarized — store this for /ask>",
  "summary": "Three-line overview of the content...",
  "key_points": [
    { "title": "Point title", "explanation": "One line of explanation" }
  ],
  "action_items": ["Do this", "Then this"]
}
```

**Error (4xx / 5xx):**

```json
{ "error": "This video has captions disabled, so there is no transcript to read." }
```

### `POST /ask`

Follow-up questions about previously summarized content. Send back the
`context` from `/summarize`, the notes, and (optionally) recent history:

```json
{
  "question": "What did the speaker say about deadlines?",
  "context": "<context string from /summarize>",
  "notes": { "summary": "...", "key_points": [], "action_items": [] },
  "history": [ { "question": "...", "answer": "..." } ]
}
```

**Success (200):** `{ "answer": "..." }` — answered strictly from the
provided context (the model says so honestly when the content doesn't
contain the answer). Only the last 3 history pairs are used. For long
content, `context` already holds the compact chunk summaries rather than
the full transcript, so follow-ups stay fast and within model limits.

## How it works

1. **Transcript layer** — YouTube links (`watch`, `youtu.be`, `shorts`,
   `embed`, `live`) go through `youtube-transcript-api`; anything else is
   fetched with `requests` and cleaned with BeautifulSoup (scripts, navs and
   footers stripped; `<article>`/`<main>` preferred).
2. **AI layer** — text under ~20k characters is summarized in one Groq call.
   Longer text is split on sentence boundaries, each chunk is summarized
   (map), and the final structured notes are produced from the combined
   summaries (reduce). The model is asked to return strict JSON, which is
   validated and normalized before it reaches the frontend.
3. **Frontend** — `generateNotes()` in `index.html` POSTs the link to
   `http://localhost:5000/summarize` and renders the response; the mind map
   is built from the returned key points.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| “Could not reach the backend” in the UI | Make sure `python app.py` is running on port 5000 |
| `groq_key_loaded: false` at `/health` | Your `.env` is missing or the key name isn't `GROQ_API_KEY` |
| “No transcript is available” | The video has no captions — try another video |
| “Couldn't extract readable text” | The article is paywalled or rendered with JavaScript |

---

Built by [@inflecta.ai](https://inflecta.ai) • Powered by AI
