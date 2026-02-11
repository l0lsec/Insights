<p align="center">
  <img src="static/logo.png" alt="Insights Logo" width="200">
</p>

<h1 align="center">Insights</h1>

<p align="center">
  <strong>AI-Powered Content Platform for Ingestion, Analysis & Social Media Publishing</strong>
</p>

---

Insights is an AI-powered content platform that turns RSS feeds, podcasts, articles, and URLs into summaries, generated articles, and platform-ready social media posts. It combines state-of-the-art speech recognition with OpenAI to process audio and text content, then helps you publish and schedule posts across LinkedIn, Threads, and more — all from a single web interface or CLI.

What started as a podcast transcription tool has grown into a complete content pipeline: ingest any source, let AI do the heavy lifting, and push polished posts out on your schedule.

### Features

#### Content Ingestion
- **RSS Feeds** - Subscribe to audio (podcast) and text (blog/news) feeds with automatic metadata parsing
- **URL Extraction** - Paste any URL and extract article content, metadata, and Open Graph images via trafilatura
- **Direct Text Input** - Provide raw text for processing without a source URL
- **Audio Transcription** - Transcribe podcast episodes using mlx-whisper (Apple Silicon), faster-whisper, or the OpenAI Whisper API

#### AI Processing
- **Summarization** - Generate concise summaries of transcripts and articles using OpenAI
- **Action Item Extraction** - Pull actionable tasks and follow-ups from any processed content
- **Article Generation** - Transform source content into polished blog posts, news articles, opinion pieces, or technical deep-dives
- **Article Refinement** - Iteratively improve generated articles with AI-assisted feedback
- **Social Media Copy** - Auto-generate platform-optimized posts for LinkedIn, Threads, Twitter/X, Facebook, Bluesky, Instagram, and Mastodon

#### Social Media Management
- **Command Center** - Central hub for generating posts from prompts, URLs, saved sources, or free text
- **Multi-Platform Generation** - Create multiple posts per platform in a single batch (1-21 posts)
- **Tone Selection** - Choose from professional, casual, witty, educational, or promotional tones
- **Image Management** - Upload images, search stock photos (Unsplash, Pexels, Pixabay), and attach to posts
- **Bulk Operations** - Bulk edit, delete, find-and-replace, and image assignment across posts

#### Publishing & Scheduling
- **LinkedIn Integration** - OAuth-based posting with rich link previews and image support
- **Threads Integration** - OAuth-based posting with text and image support
- **Time Slot Management** - Configure recurring posting times by day of week and platform
- **Auto-Queue** - Posts automatically slot into the next available time
- **Daily Limits** - Set per-platform daily posting caps
- **Background Workers** - Automated publishing, feed refresh, and episode processing run in the background

#### Integrations
- **JIRA** - Create tickets from extracted action items with full source context
- **Stock Images** - Search Unsplash, Pexels, and Pixabay for post images
- **Cloudinary** - Optional cloud image hosting for platform compatibility
- **OpenAI** - Powers transcription, summarization, article generation, and post creation

#### API & Documentation
- **Swagger/OpenAPI** - Interactive API docs at `/apidocs/` via Flasgger
- **REST Endpoints** - Full API for programmatic access to all features

### Use Cases

- Content creators repurposing podcast episodes into articles and social posts
- Marketing teams scheduling a consistent social media presence from any content source
- Researchers extracting structured summaries and action items from interviews
- Social media managers generating and queuing posts from URLs, topics, or existing text
- Business professionals converting recorded meetings into JIRA tickets
- Thought leaders building content queues across LinkedIn, Threads, and other platforms
- Podcast fans who want quick summaries before committing to a full episode

*Insights - Transforming content into actionable intelligence and engaging social media posts.*


### Feeds Page
Manage your podcast and text RSS feeds from a central dashboard. Add new feeds, open existing ones, or delete feeds you no longer need.


### Podcast Feed View
Browse episodes from audio podcast feeds with release dates, descriptions, and built-in audio players.

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


---

## Project Structure

| File | Description |
|------|-------------|
| `podinsights.py` | CLI entry point - transcribe, summarize, and extract action items from audio files |
| `podinsights_web.py` | Flask web application with all routes, background workers, and UI logic |
| `database.py` | SQLite database operations for feeds, episodes, articles, posts, schedules, and more |
| `linkedin_client.py` | LinkedIn API client - OAuth flow, token management, and post publishing |
| `threads_client.py` | Threads (Meta) API client - OAuth flow, token management, and post publishing |
| `stock_images.py` | Stock image search across Unsplash, Pexels, and Pixabay with keyword extraction |
| `templates/` | Flask HTML templates for all pages (feeds, articles, compose, schedule, etc.) |
| `static/` | Static assets (logo, favicon) |

## Requirements

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) for AI features
- For audio transcription, one of:
  - [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (recommended for Apple Silicon Macs)
  - [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) (Linux, Windows, Intel Macs)
  - OpenAI Whisper API (no extra package needed - uses your API key)

## Installation

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

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
python podinsights_web.py
```
Open `http://localhost:5001` in your browser.

**CLI** (quick one-off processing):
```bash
python podinsights.py path/to/podcast.mp3
```

## Environment Variables

All variables can be set in a `.env` file in the project root. See `.env.example` for a template.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key. Required for all AI features. |

### Optional (General)

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model used for summarization, articles, and post generation |
| `PORT` | `5001` | Port for the Flask web server |
| `FLASK_SECRET_KEY` | auto-generated | Secret key for Flask sessions |

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

> Cloudinary provides publicly accessible image URLs needed by Threads and other platforms. Sign up free at [cloudinary.com](https://cloudinary.com).

## Usage (CLI)

The CLI is designed for quick one-off processing of audio files:

```bash
python podinsights.py path/to/podcast.mp3
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
python podinsights_web.py
```

Navigate to `http://localhost:5001` to get started.

### Feeds & Content Processing

1. **Add feeds** - Enter an RSS feed URL on the home page (supports audio podcasts and text/news feeds)
2. **Browse content** - Select a feed to see its episodes or articles with descriptions, images, and audio players
3. **Process content** - Click an episode to transcribe and analyze it, or process text articles to extract summaries and action items
4. **View results** - See AI-generated summaries, action items, and the full transcript on the results page

Processed content is stored in a local SQLite database (`podinsights.db`) for quick access.

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
2. **From URL** - Paste a URL and the system extracts content to generate relevant posts
3. **From Text** - Paste existing content and transform it into social media posts
4. **From Saved Source** - Reuse previously saved URL content with different instructions

For each generation, select target platforms, choose how many posts to create (1-10), set a tone, and add optional context.

**Managing Posts:**
- Copy, edit, mark as used, post immediately, add to queue, schedule for a specific time, or delete
- Attach images from uploads or stock photo search
- Bulk edit, delete, or find-and-replace across posts

**URL Sources:**
When generating from URLs, extracted content is saved automatically. Access the **Sources** page to reuse content for future generations.

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
- **Filter by platform** - Show only LinkedIn or Threads posts
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

## Credits

Developed by Sedric "ShowUpShowOut" Louissaint from Show Up Show Out Security. 

Learn more about Show Up Show Out Security at [susos.co](https://susos.co).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
