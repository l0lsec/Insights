<p align="center">
  <img src="static/logo.png" alt="Insights Logo" width="200">
</p>

<h1 align="center">Insights</h1>

<p align="center">
  <strong>AI-Powered Content Platform for Ingestion, Analysis & Social Media Publishing</strong>
</p>

---

Insights is an AI-powered content platform that turns RSS feeds, podcasts, YouTube videos, articles, URLs, GitHub repositories, and images into summaries, generated articles, and platform-ready social media posts. It combines state-of-the-art speech recognition with OpenAI (and optional local Ollama models) to process audio, video, text, and visual content, then helps you publish and schedule posts across LinkedIn, Threads, Facebook, and X/Twitter — all from a single web interface or CLI.

What started as a podcast transcription tool has grown into a complete content pipeline: ingest any source — RSS feeds, YouTube channels, individual videos, GitHub repos, web pages, images, or raw text — let AI do the heavy lifting, and push polished posts out on your schedule.

It also works on content you already have. The **Content Library** catalogues a media archive on disk and sorts it by year and subject, including archives too large to download and stored as cloud placeholders, so a backlog measured in years becomes something you can search and reuse.

### Features

#### Content Ingestion
- **RSS Feeds** - Subscribe to audio (podcast) and text (blog/news) feeds with automatic metadata parsing
- **YouTube Videos** - Paste a YouTube video, channel, or playlist URL to transcribe and analyze video content via yt-dlp audio extraction
- **URL Extraction** - Paste any URL and extract article content, metadata, and Open Graph images via trafilatura
- **GitHub Repositories** - Paste a GitHub repo URL to ingest README content, metadata, and project details as a source
- **Image Inputs** - Drop in images to generate posts with vision-capable models (OpenAI or local Ollama)
- **Direct Text Input** - Provide raw text for processing without a source URL
- **Audio Transcription** - Transcribe podcast episodes and YouTube videos using mlx-whisper (Apple Silicon), faster-whisper, or the OpenAI Whisper API

#### AI Processing
- **Summarization** - Generate concise summaries of transcripts and articles using OpenAI
- **Action Item Extraction** - Pull actionable tasks and follow-ups from any processed content
- **Article Generation** - Transform source content into polished blog posts, news articles, opinion pieces, or technical deep-dives
- **Article Refinement** - Iteratively improve generated articles with AI-assisted feedback
- **Social Media Copy** - Auto-generate platform-optimized posts for LinkedIn, Threads, X/Twitter, Facebook, and Instagram
- **Provider Choice** - Run text and vision generation through OpenAI (default) or Anthropic Claude by setting `LLM_PROVIDER`
- **Local Model Support** - Optionally route generation through a local Ollama instance (text and vision models) instead of OpenAI

#### Content Library
Catalogue a large media archive and sort it by **year** and **category** — built for backlogs measured in years and hundreds of gigabytes, including archives that live in a cloud folder and are not fully downloaded.

- **Metadata-only scanning** - Walks the archive reading nothing but file metadata, so it downloads nothing and completes in seconds even at tens of thousands of files
- **Cloud-aware** - Detects OneDrive/iCloud "Files On-Demand" placeholders and treats downloading as a budgeted resource, so a pass can never exceed your free disk space
- **Learned categories** - Reads your category names from a folder you have already sorted by hand, so results land in your own vocabulary
- **Category discovery** - Names subjects your folders never covered, adopting a name once it recurs across several events; one-off proposals are kept as suggestions for you to promote
- **Event-based classification** - Groups files into shooting events and labels the event, cutting AI work by an order of magnitude and producing better labels than any single frame would
- **Local AI** - Images are captioned by a local vision model and video by local speech-to-text, so your photos never leave your machine. Only an optional short text-mapping step can use a cloud model
- **Cost estimation** - Predicts download size, runtime, and coverage before a run, calibrated against your archive's measured throughput
- **Duplicate detection** - Finds repeated files and reports the space reclaimable by keeping one copy of each
- **Reviewable copy plan** - Proposes a `Category/Year` (or `Year/Category`) tree that you approve per category. Files are **copied, never moved**, and an undo manifest is written alongside them
- **Resumable** - A long pass picks up where it left off after an interruption instead of re-downloading work already done

#### Social Media Management
- **Command Center** - Central hub for generating posts from prompts, URLs, saved sources, or free text
- **Multi-Platform Generation** - Create multiple posts per platform in a single batch (1-21 posts)
- **Tone Selection** - Choose from professional, casual, witty, educational, or promotional tones
- **Image Management** - Upload images, search stock photos (Unsplash, Pexels, Pixabay), and attach to posts
- **Bulk Operations** - Bulk edit, delete, find-and-replace, and image assignment across posts

#### Accounts & Cross-Posting
- **Multiple Accounts per Platform** - Connect as many logins as you like on each platform (two LinkedIn profiles, a personal and a brand Threads, several Facebook Pages). Each keeps its own token, and disconnecting one leaves the others posting
- **Named Accounts** - Give each login a name ("Work", "Studio") so two accounts on the same platform are told apart everywhere they appear
- **Default Account** - Each platform has a default that anything not naming an account goes to, so a single-account setup behaves exactly as it always did
- **Cross-Posting** - Write once and aim the same copy at any mix of platforms and accounts. One action publishes it to all of them and reports which landed and which did not, rather than stopping at the first failure
- **Cross-Posting to the Queue** - Queue the same copy for every target at once; each takes its own next free slot on its own platform, so nothing stacks up at the same minute
- **Per-Account Scheduling** - Every queued post records the account that will publish it, so two accounts on one platform move through the queue independently
- **Accounts Page** - `/accounts` lists every connected login, flags the ones that still need configuring or reconnecting, and is where accounts are added, named, defaulted and disconnected

#### Publishing & Scheduling
- **LinkedIn Integration** - OAuth-based posting with rich link previews and image support
- **Threads Integration** - OAuth-based posting with text and image support
- **Facebook Integration** - OAuth-based posting to Facebook Pages with text and image support
- **X/Twitter Integration** - OAuth 2.0 with PKCE for posting text and images (pay-per-use media uploads)
- **Instagram Integration** - OAuth-based publishing of feed posts, **carousels** (2–10 images/videos), **Reels**, and **Stories** (image or video). Requires a professional (Business/Creator) account
- **Time Slot Management** - Configure recurring posting times by day of week and platform
- **Auto-Queue** - Posts automatically slot into the next available time
- **Daily Limits** - Set per-platform daily posting caps
- **Background Workers** - Automated publishing, feed refresh, and episode processing run in the background

#### Integrations
- **JIRA** - Create tickets from extracted action items with full source context
- **GitHub** - Fetch repo metadata and README content as a source for posts and articles
- **Stock Images** - Search Unsplash, Pexels, and Pixabay for post images
- **Cloudinary** - Optional cloud image hosting for platform compatibility
- **OpenAI** - Powers transcription, summarization, article generation, and post creation
- **Ollama (optional)** - Run text and vision generation locally against your own models

#### API & Documentation
- **Swagger/OpenAPI** - Interactive API docs at `/apidocs/` via Flasgger
- **REST Endpoints** - Full API for programmatic access to all features

### Use Cases

- Content creators repurposing podcast episodes and YouTube videos into articles and social posts
- Marketing teams scheduling a consistent social media presence from any content source
- Researchers extracting structured summaries and action items from interviews or video lectures
- Social media managers generating and queuing posts from URLs, topics, or existing text
- Business professionals converting recorded meetings into JIRA tickets
- Thought leaders building content queues across LinkedIn, Threads, Facebook, X/Twitter, and other platforms
- Podcast fans who want quick summaries before committing to a full episode
- YouTube viewers who prefer reading transcripts and summaries over watching long-form videos
- Creators with years of unsorted photo and video backlog who need it catalogued by year and subject before it can be reused

*Insights - Transforming content into actionable intelligence and engaging social media posts.*


### Feeds Page
Manage your podcast, text, and YouTube feeds from a central dashboard. Add new feeds by pasting an RSS URL or YouTube link, open existing ones, or delete feeds you no longer need.

### Podcast Feed View
Browse episodes from audio podcast feeds with release dates, descriptions, and built-in audio players.

### YouTube Feed View
Browse videos from YouTube channels and playlists with embedded video players, thumbnails, and one-click transcription.

### Text Feed View
Browse articles from text-based RSS feeds (like news sites and blogs) with thumbnail images and article previews.


### Episode Results
View AI-generated summaries and extracted action items from processed episodes. The summary renders markdown formatting for easy reading.


### Generate Article
Transform content into polished blog posts and articles. Choose your topic, style, and add optional context.


### Processing Status
Track all processed episodes across feeds. Reprocess or delete episodes as needed.


### Articles Page
Access all generated articles in one place.


### JIRA Tickets
View and manage JIRA tickets created from action items.


### Command Center (Compose)
Generate social media posts from any source - prompts, URLs, or text. Save URL sources for future use and manage your content pipeline.


### Schedule Queue
View and manage your posting queue with drag-and-drop reordering, status/platform filters, and automated time slot management.


### Content Library
Catalogue a media archive by year and category, browse it by either, review discovered categories and duplicates, and approve a copy plan that sorts the files.


---

## Project Structure

| File | Description |
|------|-------------|
| `insights.py` | CLI entry point and core AI generation library (transcription, summaries, articles, social copy, vision, thumbnails) |
| `insights_web.py` | Flask web application with all routes, background workers, and UI logic |
| `social_publisher.py` | One publish path for every platform and account - resolves the target account, refreshes its token, calls the right client, and reports each target's outcome in the same shape |
| `database.py` | SQLite database operations for feeds, episodes, articles, posts, schedules, sources, library, and more |
| `content_agent.py` | Content brief orchestrator - researches sources and prepares draft posts and articles for review |
| `content_library.py` | Content Library engine - archive scanning, event grouping, taxonomy learning, classification, and copy planning |
| `media_probe.py` | Filesystem and media inspection - cloud-placeholder detection, hydration budgeting, HEIC/video thumbnails and metadata |
| `document_extractor.py` | Text extraction from uploaded PDF, Word, PowerPoint, Excel, CSV, HTML, and plain-text documents |
| `research_engine.py` | Source research and retrieval used by the content agent |
| `web_search.py` | External web-search providers used for research |
| `usage_meter.py` | AI usage and cost accounting across providers |
| `starter_prompts.py` | Built-in prompt library seed data |
| `linkedin_client.py` | LinkedIn API client - OAuth flow, token management, and post publishing |
| `threads_client.py` | Threads (Meta) API client - OAuth flow, token management, and post publishing |
| `facebook_client.py` | Facebook Pages API client - OAuth flow, page token management, and post publishing |
| `twitter_client.py` | X/Twitter API v2 client - OAuth 2.0 PKCE flow, token management, text and image posting |
| `instagram_client.py` | Instagram Graph API client - OAuth flow, feed posts, carousels, Reels, and Stories |
| `github_client.py` | GitHub repo URL parsing and metadata/README fetching for source ingestion |
| `stock_images.py` | Stock image search across Unsplash, Pexels, and Pixabay with keyword extraction |
| `templates/` | Flask HTML templates for all pages (feeds, articles, compose, schedule, library, etc.) |
| `static/` | Static assets (logo, favicon) |
| `insights.db` | Local SQLite database (created on first run) |

## Requirements

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) for AI features
- [FFmpeg](https://ffmpeg.org/) installed on your system (required by yt-dlp for YouTube audio extraction)
- For audio transcription, one of:
  - [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (recommended for Apple Silicon Macs)
  - [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) (Linux, Windows, Intel Macs)
  - OpenAI Whisper API (no extra package needed - uses your API key)

For the **Content Library** specifically:

- FFmpeg (already required above) provides `ffprobe`/`ffmpeg` for video metadata and frame extraction
- `sips` (built into macOS) converts HEIC images, which most modern phone archives are full of and which Pillow cannot read without the optional `pillow-heif` package
- An [Ollama](https://ollama.com/) server with a vision model for local captioning, so images are never sent off your machine:
  ```bash
  ollama pull llama3.2-vision
  ollama pull llama3.2
  ```
  The Library page reports which of these are present and warns you before a run if any are missing.

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. Install dependencies

Use the venv’s pip so packages install into the venv (avoids “Defaulting to user installation” and scripts not on PATH):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you see **“Defaulting to user installation because normal site-packages is not writeable”** or scripts installed to `~/Library/Python/3.x/bin`, you’re not using the venv’s pip. Use `python -m pip` (and run the app with `python insights_web.py`), not `pip3`/`python3`, so the venv’s interpreter is used. With the venv activated, `which python` should show a path inside the project’s `venv/`.

The default `requirements.txt` includes `mlx-whisper` for Apple Silicon Macs. If you are on a different platform:

- **Linux / Windows / Intel Mac** - Replace `mlx-whisper` with `faster-whisper`:
  ```bash
  pip install faster-whisper
  ```
- **Any platform (API-based)** - Skip both packages entirely. Set your `OPENAI_API_KEY` and Insights will use the OpenAI Whisper API as a fallback.

### 3. Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

At minimum, set your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

See the [Environment Variables](#environment-variables) section below for all options.

### 4. Run the application

**Web UI** (recommended):
```bash
python insights_web.py
```
Open `http://localhost:5001` in your browser.

**CLI** (quick one-off processing):
```bash
python insights.py path/to/podcast.mp3
```

## Environment Variables

All variables can be set in a `.env` file in the project root. See `.env.example` for a template.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key. Required for all AI features. Still required for audio transcription and thumbnail generation even when `LLM_PROVIDER=anthropic`, since those have no Anthropic equivalent. |

### Optional (General)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Cloud provider for text/vision generation: `openai` or `anthropic` |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model used for summarization, articles, and post generation |
| `ANTHROPIC_API_KEY` | — | Anthropic (Claude) API key. Required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model used when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for an optional local Ollama server |
| `OLLAMA_TEXT_MODEL` | `llama3.2` | Local text model used when generation is routed through Ollama |
| `OLLAMA_VISION_MODEL` | `llama3.2-vision` | Local vision model used when image inputs are routed through Ollama |
| `PORT` | `5001` | Port for the Flask web server |
| `FLASK_SECRET_KEY` | auto-generated | Secret key for Flask sessions |

### Content Library

The Library reuses `OLLAMA_BASE_URL`, `OLLAMA_TEXT_MODEL`, and `OLLAMA_VISION_MODEL` above for local captioning and mapping. These tune how it reads a cloud-backed archive.

| Variable | Default | Description |
|----------|---------|-------------|
| `LIBRARY_PREFETCH_WORKERS` | `6` | Parallel downloads warmed ahead of the classifier. Cloud providers are limited by per-request latency rather than bandwidth, so this is the most effective performance knob — measured on a real archive, 1 thread sustained 0.30 MB/s and 4 reached 3.11 MB/s |
| `LIBRARY_PROBE_TIMEOUT` | `60` | Seconds to wait on a metadata probe of a single file. On a cloud path this includes the provider downloading the file first |
| `LIBRARY_CONVERT_TIMEOUT` | `90` | Seconds to wait on a thumbnail or frame extraction |

### JIRA Integration

| Variable | Description |
|----------|-------------|
| `JIRA_BASE_URL` | Your JIRA Cloud instance URL (e.g., `https://example.atlassian.net`) |
| `JIRA_EMAIL` | Email associated with your JIRA API token |
| `JIRA_API_TOKEN` | Your JIRA API token ([generate here](https://id.atlassian.com/manage-profile/security/api-tokens)) |
| `JIRA_PROJECT_KEY` | Project key where issues are created |

### LinkedIn Integration

| Variable | Description |
|----------|-------------|
| `LINKEDIN_CLIENT_ID` | Client ID from the [LinkedIn Developer Portal](https://www.linkedin.com/developers/) |
| `LINKEDIN_CLIENT_SECRET` | Client Secret from your LinkedIn app |
| `LINKEDIN_REDIRECT_URI` | OAuth callback URL (default: `http://localhost:5001/linkedin/callback`) |
| `LINKEDIN_SCOPES` | OAuth scopes (default: `openid profile w_member_social`) |

### Threads Integration

| Variable | Description |
|----------|-------------|
| `THREADS_APP_ID` | App ID from the [Meta Developer Portal](https://developers.facebook.com/) |
| `THREADS_APP_SECRET` | App Secret from your Meta app |
| `THREADS_REDIRECT_URI` | OAuth callback URL (must be HTTPS, e.g., `https://your-domain.com/threads/callback`) |
| `THREADS_SCOPES` | OAuth scopes (default: `threads_basic,threads_content_publish`) |

### Facebook Integration

| Variable | Description |
|----------|-------------|
| `FACEBOOK_APP_ID` | App ID from the [Meta Developer Portal](https://developers.facebook.com/) |
| `FACEBOOK_APP_SECRET` | App Secret from your Meta app |
| `FACEBOOK_REDIRECT_URI` | OAuth callback URL (default: `http://localhost:5001/facebook/callback`) |

### X/Twitter Integration

| Variable | Description |
|----------|-------------|
| `TWITTER_CLIENT_ID` | Client ID from the [X Developer Portal](https://developer.x.com/) |
| `TWITTER_CLIENT_SECRET` | Client Secret from your X app |
| `TWITTER_REDIRECT_URI` | OAuth callback URL (default: `http://localhost:5001/twitter/callback`) |

### Instagram Integration

| Variable | Description |
|----------|-------------|
| `INSTAGRAM_APP_ID` | Instagram App ID from your Meta app's **Instagram** product (not the Meta app's own ID) |
| `INSTAGRAM_APP_SECRET` | Instagram App Secret from the same product |
| `INSTAGRAM_REDIRECT_URI` | OAuth callback URL — must be **HTTPS** (default: `https://localhost:5001/instagram/callback`) |
| `INSTAGRAM_SCOPES` | Optional; defaults to `instagram_business_basic,instagram_business_content_publish` |

> Uses the "Instagram API with Instagram Login" flavor (no Facebook Page required). Requires a professional (Business/Creator) Instagram account. Feed/carousel/reel/story media must be publicly hosted — configure Cloudinary or paste public URLs.

### Stock Images (all free)

| Variable | Rate Limit | Sign Up |
|----------|-----------|---------|
| `UNSPLASH_ACCESS_KEY` | 50 req/hour | [unsplash.com/developers](https://unsplash.com/developers) |
| `PEXELS_API_KEY` | 200 req/hour | [pexels.com/api](https://www.pexels.com/api/) |
| `PIXABAY_API_KEY` | 5,000 req/hour | [pixabay.com/api/docs](https://pixabay.com/api/docs/) |

### Cloudinary (optional)

| Variable | Description |
|----------|-------------|
| `CLOUDINARY_CLOUD_NAME` | Your Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | Cloudinary API secret |

> Cloudinary provides publicly accessible image and video URLs needed by Threads, Facebook, X/Twitter, and Instagram. It is effectively required for Instagram (and for Instagram Reels/video, which upload as `resource_type=video`). Sign up free at [cloudinary.com](https://cloudinary.com).

## Usage (CLI)

The CLI is designed for quick one-off processing of audio files:

```bash
python insights.py path/to/podcast.mp3
```

The script transcribes the audio, generates a summary, and extracts action items. Results are printed to the terminal and saved to a JSON file alongside the audio. Use `--json` to specify a custom output path and `--verbose` for debug logging.

The JSON output contains:
- `transcript` - the full transcript
- `summary` - the generated summary
- `action_items` - a list of extracted action items

> **Note**: Summarization and action item extraction require `OPENAI_API_KEY` to be set.

## Usage (Web UI)

The web UI is the primary interface and provides access to all features. Start it with:

```bash
python insights_web.py
```

Navigate to `http://localhost:5001` to get started.

### Feeds & Content Processing

1. **Add feeds** - Enter an RSS feed URL or YouTube link on the home page (supports audio podcasts, text/news feeds, YouTube channels, playlists, and individual videos)
2. **Browse content** - Select a feed to see its episodes, videos, or articles with descriptions, images, and embedded players
3. **Process content** - Click an episode or video to transcribe and analyze it, or process text articles to extract summaries and action items
4. **View results** - See AI-generated summaries, action items, and the full transcript on the results page

Processed content is stored in a local SQLite database (`insights.db`) for quick access.

### YouTube Videos

YouTube videos are supported as a first-class content source. Paste any of these URL formats into the feed input:

- **Single video** - `https://www.youtube.com/watch?v=VIDEO_ID` or `https://youtu.be/VIDEO_ID` — the video is added to a catch-all "YouTube Videos" feed and immediately queued for transcription
- **Channel** - `https://www.youtube.com/@ChannelName` or `https://www.youtube.com/channel/UC...` — creates a feed from the channel's latest uploads
- **Playlist** - `https://www.youtube.com/playlist?list=PL...` — creates a feed from all videos in the playlist

YouTube feeds display embedded video players alongside Queue and Process buttons. Audio is extracted from each video using yt-dlp, then transcribed through the same Whisper pipeline used for podcasts. From there, all downstream features work identically: summaries, action items, article generation, and social media post creation.

### Generating Articles

Transform any processed content into polished articles:

1. Process an episode or article to get the transcript and summary
2. Scroll to the **Generate Article** section
3. Enter a topic or angle (e.g., "Privacy implications of AI voice assistants")
4. Select an article style:
   - **Blog Post** - Conversational and engaging
   - **News Article** - Factual and objective reporting
   - **Opinion/Editorial** - Analysis with perspective
   - **Technical Deep-Dive** - Detailed for practitioners
5. Click **Generate Article**

Articles can be refined with AI-assisted feedback and are saved on the **Articles** page.

### Command Center

The Command Center (`/compose`) is your hub for social media content creation:

**Generating Posts:**
1. **From Prompt** - Enter any topic or idea and let AI generate platform-optimized posts
2. **From URL** - Paste a URL (including GitHub repos) and the system extracts content to generate relevant posts
3. **From Text** - Paste existing content and transform it into social media posts
4. **From Images** - Drop in images and use a vision model to generate posts grounded in what they show
5. **From Saved Source** - Reuse previously saved URL, GitHub, or text content with different instructions

For each generation, select target platforms, choose how many posts to create (1-21 per platform), set a tone, and add optional context.

**Managing Posts:**
- Copy, edit, mark as used, post immediately, add to queue, schedule for a specific time, or delete
- Attach images from uploads or stock photo search
- Bulk edit, delete, or find-and-replace across posts

**URL & GitHub Sources:**
When generating from URLs or GitHub repos, extracted content (article body, repo metadata, README) is saved automatically. Access the **Sources** page to reuse content for future generations.

### Content Library

The **Library** page catalogues a media archive on disk and sorts it by year and category. It is designed for archives that are far larger than your free disk space — a cloud folder whose files are mostly placeholders — so it separates work that is free from work that costs a download, and never spends the latter without telling you first.

**Setup:**
1. Add the archive you want sorted as a folder
2. If you have a folder you already sorted by hand, add it too and press **Learn categories** — its subfolder names become your starting taxonomy. This is optional; with discovery enabled the Library can build a taxonomy from nothing
3. Press **Scan**. This reads only file metadata, so it downloads nothing and finishes in seconds even for tens of thousands of files

The scan alone gives you a browsable catalogue: files by year, file kinds, total size, how much is stored locally versus in the cloud, and duplicate groups with the space they waste.

**Classifying:**

Press **Estimate cost** before running anything. It measures your archive's real download throughput and reports exactly how much will be downloaded, roughly how long it will take, and what percentage of the archive the run will cover — then refuses to start a run that would not fit on your disk.

Classification runs cheapest-first:

| Tier | Signal | Downloads? | Cost |
|------|--------|-----------|------|
| Path and filename rules | folder and file names | no | free |
| Vision captions | a sampled frame per event | yes | free (local model) |
| Speech transcripts | audio from video | yes | free (local model) |

Options worth knowing:

- **Years** — scope a run to a year range and work through a large archive in stages
- **Discover new categories** (on by default) — lets the classifier name subjects your folders never covered. A name is adopted once it recurs across 3 events; rarer proposals are saved switched off in the Categories panel for you to promote with one click
- **Cloud mapping** — sends only the short text step to a cloud model, which is meaningfully more accurate at matching a caption to a category. Images are never sent
- **Max file size** — the per-file download ceiling. Events whose smallest file exceeds it are deferred rather than downloaded, and a later run with a higher ceiling will pick them up
- **Resume** (on by default) — skips events an earlier pass already finished, so an interrupted run continues instead of starting over

**Reviewing and copying:**

Build a copy plan, choose `Category/Year` or `Year/Category`, set a minimum confidence, and approve per category. Nothing is copied until you approve it, files are **copied rather than moved**, and an undo manifest is written into the destination so the whole operation can be reversed. Anything the classifier could not place lands in `Unsorted` for review rather than being guessed at.

### Schedule Management

The **Schedule** page provides full queue management for automated publishing:

**Time Slots:**
- Configure recurring posting times (daily or specific days of the week)
- Assign slots to specific platforms
- Set daily posting limits per platform
- Enable/disable slots as needed

**Queue Features:**
- **Drag-and-drop reordering** - Rearrange posts by dragging the grip handle
- **Filter by status** - View pending, posted, failed, or cancelled posts
- **Filter by platform** - Show only LinkedIn, Threads, Facebook, or X/Twitter posts
- **Post Now** - Immediately publish any pending post (remaining posts auto-redistribute)
- **Edit time** - Change the scheduled time for any pending post
- **Bulk actions** - Select and delete multiple posts at once

The background worker checks every 60 seconds and publishes posts when their scheduled time arrives.

### Creating JIRA Tickets

Create JIRA issues directly from extracted action items. Set the following environment variables:

- `JIRA_BASE_URL` - e.g., `https://example.atlassian.net`
- `JIRA_EMAIL` - email associated with an API token
- `JIRA_API_TOKEN` - your JIRA API token
- `JIRA_PROJECT_KEY` - project key for new issues

Select action items on any results page and click **Create JIRA Tickets**. Each ticket includes the source context (episode title and summary) so your team has immediate background. Ticket status syncs live from JIRA whenever you view the tickets page.

### Managing Accounts and Cross-Posting

Open **Accounts** in the navigation to see every connected login, grouped by
platform.

#### Connecting more than one account on a platform

1. Click **Connect** for a platform to add your first login, then **Connect
   another** to add a second (or fifth) on the same platform
2. Each login keeps its own credentials. Re-authorising an account you already
   connected refreshes it in place rather than adding a duplicate
3. Click **Rename** to name an account ("Work", "Studio"). The name is what
   tells two accounts on the same platform apart everywhere they appear
4. Click **Make default** to choose which account a bare platform name means.
   Anything that does not name an account, including posts saved before you
   connected a second one, goes to the default
5. Click **Disconnect** to remove one login. The platform's other accounts keep
   working; any queued posts aimed at the removed account are taken out of the
   queue rather than quietly published from a different account

#### Sending one post to several platforms and accounts

In the Command Center, each saved post is a card with a chip per destination:
one chip per platform, and one chip per account where a platform has more than
one connected.

- **Tick a chip** to send this card's copy there. Ticking creates that
  destination's post; unticking removes it
- **🚀 Post to all** publishes the same copy to every ticked destination in one
  action. Targets are published independently, so one failing does not stop the
  rest, and the result names exactly which landed and which did not
- **📅 Queue all** puts the same copy in the queue for every ticked destination.
  Each takes its own next free slot on its own platform, so two accounts on one
  platform do not land at the same minute
- When generating posts, the **Accounts** row under the platform picker chooses
  which accounts each platform writes to. Copy is generated once per platform
  whatever you pick, so aiming at two accounts costs nothing extra

### Posting to LinkedIn

#### Setup

1. Create a LinkedIn App at the [LinkedIn Developer Portal](https://www.linkedin.com/developers/apps)
2. Add the **Share on LinkedIn** and **Sign In with LinkedIn using OpenID Connect** products
3. Add `http://localhost:5001/linkedin/callback` as an OAuth redirect URL
4. Set `LINKEDIN_CLIENT_ID`, `LINKEDIN_CLIENT_SECRET`, and `LINKEDIN_REDIRECT_URI` in your `.env`
5. Click **Connect LinkedIn** in the web UI

Posts containing URLs automatically include rich link previews with title, description, and thumbnail.

### Posting to Threads

#### Setup

1. Create a Meta App at the [Meta Developer Portal](https://developers.facebook.com/apps/)
2. Add the **Threads API** use case and request `threads_basic` and `threads_content_publish` permissions
3. Add your HTTPS redirect URI in Threads API settings
4. Set `THREADS_APP_ID`, `THREADS_APP_SECRET`, and `THREADS_REDIRECT_URI` in your `.env`
5. Click **Connect Threads** in the web UI

> **Local Development:** Meta requires HTTPS for OAuth redirects. Use [ngrok](https://ngrok.com/) to create a tunnel:
> ```bash
> ngrok http 5001
> ```
> Then set `THREADS_REDIRECT_URI` to the ngrok HTTPS URL and add it to your Meta app settings.

### Posting to Facebook

#### Setup

1. Create a Meta App at the [Meta Developer Portal](https://developers.facebook.com/apps/) (or reuse your Threads app)
2. Add the **Facebook Login** product and request `pages_manage_posts` and `pages_read_engagement` permissions
3. Add `http://localhost:5001/facebook/callback` as an OAuth redirect URI
4. Set `FACEBOOK_APP_ID`, `FACEBOOK_APP_SECRET`, and `FACEBOOK_REDIRECT_URI` in your `.env`
5. Click **Connect Facebook** in the web UI and authorize the Page you want to post to

Posts are published to the selected Facebook Page with text and optional image attachments.

### Posting to X/Twitter

#### Setup

1. Create a project and app at the [X Developer Portal](https://developer.x.com/)
2. Enable **OAuth 2.0** with the **PKCE** type and set the callback URL to `http://localhost:5001/twitter/callback`
3. Request at minimum the `tweet.read`, `tweet.write`, `users.read`, and `offline.access` scopes
4. Set `TWITTER_CLIENT_ID`, `TWITTER_CLIENT_SECRET`, and `TWITTER_REDIRECT_URI` in your `.env`
5. Click **Connect X** in the web UI

X/Twitter uses the v2 API with pay-per-use pricing. Text posts cost $0.01 each. Image uploads use the chunked media upload endpoint (max 5 MB per image).

### Posting to Instagram

#### Setup

1. Create a Meta App at the [Meta Developer Portal](https://developers.facebook.com/apps/) and add the **Instagram** product ("API setup with Instagram login")
2. Under that product, copy the **Instagram App ID** and **Instagram App Secret** (these differ from the Meta app's own ID/secret)
3. In **Set up business login**, add an **HTTPS** redirect URI (e.g. `https://localhost:5001/instagram/callback` or your public tunnel URL)
4. Under **App roles → Roles**, add your Instagram account as an **Instagram Tester**, then accept the invite in the Instagram app (*Settings → Apps and websites → Tester Invites*)
5. Set `INSTAGRAM_APP_ID`, `INSTAGRAM_APP_SECRET`, and `INSTAGRAM_REDIRECT_URI` in your `.env`
6. Click **Connect Instagram** in the web UI and authorize

Requires a professional (**Business** or **Creator**) Instagram account. In Meta dev mode, tester access is sufficient for single-user posting — no App Review needed.

#### Post formats

In the Command Center, each Instagram post has a **format selector**:

- **Feed** — a single image (auto-attaches a stock image if none is set)
- **Carousel** — 2–10 images and/or videos
- **Reel** — a single video
- **Story** — a single image or video (no caption; Stories expire after 24h)

Media is publicly hosted via Cloudinary (image/video upload) or a pasted public URL. Instagram has no text-only feed posts, so every post needs media. Videos must be MP4/MOV (H.264/AAC, ~90s) — Instagram rejects unsupported codecs/aspect ratios/durations, which the app surfaces as a readable error.

## Credits

Developed by Sedric "ShowUpShowOut" Louissaint. 

Learn more about Show Up Show Out Security at [susos.co](https://susos.co).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
