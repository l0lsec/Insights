"""Simple SQLite helpers for PodInsights.

This module abstracts the small SQLite database used by both the CLI and the
web interface. Each function wraps a query so callers don't need to know SQL.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, Optional, List
from datetime import datetime

DB_PATH = "episodes.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if the database file is empty."""
    with sqlite3.connect(db_path) as conn:
        # Table storing RSS feeds that users have added
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                feed_type TEXT,
                last_post TEXT,
                item_count INTEGER,
                last_checked TEXT
            )
            """
        )
        # Upgrade existing feeds table with new columns
        cur = conn.execute("PRAGMA table_info(feeds)")
        columns = [row[1] for row in cur.fetchall()]
        if "feed_type" not in columns:
            conn.execute("ALTER TABLE feeds ADD COLUMN feed_type TEXT")
        if "last_post" not in columns:
            conn.execute("ALTER TABLE feeds ADD COLUMN last_post TEXT")
        if "item_count" not in columns:
            conn.execute("ALTER TABLE feeds ADD COLUMN item_count INTEGER")
        if "last_checked" not in columns:
            conn.execute("ALTER TABLE feeds ADD COLUMN last_checked TEXT")
        # Each processed episode is stored here along with its state
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_id INTEGER,
                url TEXT UNIQUE,
                title TEXT,
                transcript TEXT,
                summary TEXT,
                action_items TEXT,
                status TEXT,
                published TEXT,
                processed_at TEXT,
                FOREIGN KEY(feed_id) REFERENCES feeds(id)
            )
            """
        )
        # Created JIRA tickets are tracked in a separate table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jira_tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                action_item TEXT,
                ticket_key TEXT,
                ticket_url TEXT,
                FOREIGN KEY(episode_id) REFERENCES episodes(id)
            )
            """
        )
        # Generated articles are stored here
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id INTEGER,
                topic TEXT,
                style TEXT,
                content TEXT,
                created_at TEXT,
                FOREIGN KEY(episode_id) REFERENCES episodes(id)
            )
            """
        )
        # Social media posts generated for articles
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS social_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                platform TEXT,
                content TEXT,
                image_url TEXT,
                created_at TEXT,
                used INTEGER DEFAULT 0,
                FOREIGN KEY(article_id) REFERENCES articles(id)
            )
            """
        )
        # Upgrade social_posts table to include image_url if missing
        cur = conn.execute("PRAGMA table_info(social_posts)")
        social_columns = [row[1] for row in cur.fetchall()]
        if "image_url" not in social_columns:
            conn.execute("ALTER TABLE social_posts ADD COLUMN image_url TEXT")
        # LinkedIn OAuth tokens storage
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS linkedin_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_token TEXT,
                refresh_token TEXT,
                expires_at TEXT,
                member_id TEXT,
                user_urn TEXT,
                display_name TEXT,
                email TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        # Threads OAuth tokens storage
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_token TEXT,
                expires_at TEXT,
                user_id TEXT,
                username TEXT,
                display_name TEXT,
                profile_picture_url TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        # Facebook OAuth tokens storage (Pages + Groups)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facebook_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_token TEXT,
                expires_at TEXT,
                user_id TEXT,
                user_name TEXT,
                page_id TEXT,
                page_name TEXT,
                page_access_token TEXT,
                group_ids TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        # Scheduled posts queue for LinkedIn and other platforms
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scheduled_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                social_post_id INTEGER,
                article_id INTEGER,
                standalone_post_id INTEGER,
                post_type TEXT,
                platform TEXT DEFAULT 'linkedin',
                scheduled_for TEXT,
                status TEXT DEFAULT 'pending',
                linkedin_post_urn TEXT,
                error_message TEXT,
                created_at TEXT,
                posted_at TEXT,
                retry_count INTEGER DEFAULT 0,
                FOREIGN KEY(social_post_id) REFERENCES social_posts(id),
                FOREIGN KEY(article_id) REFERENCES articles(id),
                FOREIGN KEY(standalone_post_id) REFERENCES standalone_posts(id)
            )
            """
        )
        # Upgrade scheduled_posts table to include missing columns
        cur = conn.execute("PRAGMA table_info(scheduled_posts)")
        sched_columns = [row[1] for row in cur.fetchall()]
        if "standalone_post_id" not in sched_columns:
            conn.execute("ALTER TABLE scheduled_posts ADD COLUMN standalone_post_id INTEGER")
        if "retry_count" not in sched_columns:
            conn.execute("ALTER TABLE scheduled_posts ADD COLUMN retry_count INTEGER DEFAULT 0")
        # Schedule settings for configurable time slots
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schedule_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key TEXT UNIQUE,
                setting_value TEXT,
                updated_at TEXT
            )
            """
        )
        # Time slots for queue-based scheduling
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schedule_time_slots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week INTEGER,
                time_slot TEXT,
                enabled INTEGER DEFAULT 1,
                created_at TEXT
            )
            """
        )
        # Platform assignments for time slots (many-to-many)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS time_slot_platforms (
                slot_id INTEGER NOT NULL,
                platform TEXT NOT NULL,
                PRIMARY KEY (slot_id, platform),
                FOREIGN KEY(slot_id) REFERENCES schedule_time_slots(id) ON DELETE CASCADE
            )
            """
        )
        # Platform daily posting limits
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS platform_daily_limits (
                platform TEXT PRIMARY KEY,
                max_posts_per_day INTEGER DEFAULT 0
            )
            """
        )
        # Standalone posts for the Command Center (not tied to articles)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS standalone_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT,
                source_content TEXT,
                platform TEXT,
                content TEXT,
                image_url TEXT,
                created_at TEXT,
                used INTEGER DEFAULT 0
            )
            """
        )
        # Upgrade standalone_posts table to include image_url if missing
        cur = conn.execute("PRAGMA table_info(standalone_posts)")
        standalone_columns = [row[1] for row in cur.fetchall()]
        if "image_url" not in standalone_columns:
            conn.execute("ALTER TABLE standalone_posts ADD COLUMN image_url TEXT")
        # URL sources - stores extracted content from URLs for reuse
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS url_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT,
                description TEXT,
                content TEXT,
                og_image TEXT,
                created_at TEXT,
                last_used_at TEXT
            )
            """
        )
        # Uploaded images library - tracks images uploaded to Cloudinary or locally
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                url TEXT UNIQUE,
                storage TEXT,
                size INTEGER,
                created_at TEXT
            )
            """
        )
        # Upgrade any existing DB with newer columns
        cur = conn.execute("PRAGMA table_info(episodes)")
        columns = [row[1] for row in cur.fetchall()]
        if "status" not in columns:
            conn.execute("ALTER TABLE episodes ADD COLUMN status TEXT")
            conn.execute("UPDATE episodes SET status = 'complete'")
        if "published" not in columns:
            conn.execute("ALTER TABLE episodes ADD COLUMN published TEXT")
        if "processed_at" not in columns:
            conn.execute("ALTER TABLE episodes ADD COLUMN processed_at TEXT")
        conn.commit()


def get_feed(url: str, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Retrieve a feed record by its RSS URL."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM feeds WHERE url = ?", (url,))
        return cur.fetchone()


def get_feed_by_id(feed_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Return a feed row given its integer ``id``."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,))
        return cur.fetchone()


def list_feeds(db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """Return all stored feeds ordered by title."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM feeds ORDER BY title")
        return cur.fetchall()


def delete_feed(feed_id: int, db_path: str = DB_PATH) -> None:
    """Delete a feed and all associated episodes, articles, and tickets."""
    with sqlite3.connect(db_path) as conn:
        # Delete tickets for episodes in this feed
        conn.execute(
            """
            DELETE FROM jira_tickets WHERE episode_id IN (
                SELECT id FROM episodes WHERE feed_id = ?
            )
            """,
            (feed_id,),
        )
        # Delete articles for episodes in this feed
        conn.execute(
            """
            DELETE FROM articles WHERE episode_id IN (
                SELECT id FROM episodes WHERE feed_id = ?
            )
            """,
            (feed_id,),
        )
        # Delete episodes for this feed
        conn.execute("DELETE FROM episodes WHERE feed_id = ?", (feed_id,))
        # Delete the feed itself
        conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
        conn.commit()


def delete_feeds_bulk(feed_ids: List[int], db_path: str = DB_PATH) -> int:
    """Delete multiple feeds and all their associated data. Returns count deleted."""
    if not feed_ids:
        return 0
    placeholders = ",".join("?" * len(feed_ids))
    with sqlite3.connect(db_path) as conn:
        # Delete tickets for episodes in these feeds
        conn.execute(
            f"""
            DELETE FROM jira_tickets WHERE episode_id IN (
                SELECT id FROM episodes WHERE feed_id IN ({placeholders})
            )
            """,
            feed_ids,
        )
        # Delete articles for episodes in these feeds
        conn.execute(
            f"""
            DELETE FROM articles WHERE episode_id IN (
                SELECT id FROM episodes WHERE feed_id IN ({placeholders})
            )
            """,
            feed_ids,
        )
        # Delete episodes for these feeds
        conn.execute(
            f"DELETE FROM episodes WHERE feed_id IN ({placeholders})",
            feed_ids,
        )
        # Delete the feeds themselves
        cur = conn.execute(
            f"DELETE FROM feeds WHERE id IN ({placeholders})",
            feed_ids,
        )
        conn.commit()
        return cur.rowcount


def add_feed(url: str, title: str, db_path: str = DB_PATH) -> int:
    """Insert a new feed if needed and return its ``id``."""
    with sqlite3.connect(db_path) as conn:
        # ``INSERT OR IGNORE`` lets us call this repeatedly with the same URL
        cur = conn.execute(
            "INSERT OR IGNORE INTO feeds (url, title) VALUES (?, ?)",
            (url, title),
        )
        # ``rowcount`` will be >0 when a new row was inserted
        if cur.rowcount:
            feed_id = cur.lastrowid
        else:
            cur = conn.execute("SELECT id FROM feeds WHERE url = ?", (url,))
            feed_id = cur.fetchone()[0]
        conn.commit()
        return feed_id


def update_feed_metadata(
    feed_id: int,
    feed_type: str,
    last_post: str | None,
    item_count: int,
    db_path: str = DB_PATH,
) -> None:
    """Update cached metadata for a feed."""
    from datetime import datetime
    last_checked = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE feeds 
            SET feed_type = ?, last_post = ?, item_count = ?, last_checked = ?
            WHERE id = ?
            """,
            (feed_type, last_post, item_count, last_checked, feed_id),
        )
        conn.commit()


def get_episode(url: str, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Retrieve a processed episode by its audio URL."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM episodes WHERE url = ?", (url,))
        return cur.fetchone()


def save_episode(
    url: str,
    title: str,
    transcript: str,
    summary: str,
    action_items: Iterable[str],
    feed_id: int,
    published: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Persist a fully processed episode."""
    # Store the list of action items as newline separated text
    actions = "\n".join(action_items)
    processed_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO episodes
                (url, title, transcript, summary, action_items, feed_id, status, published, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, 'complete', ?, ?)
            """,
            (url, title, transcript, summary, actions, feed_id, published, processed_at),
        )  # episode is complete once saved
        conn.commit()


def queue_episode(
    url: str,
    title: str,
    feed_id: int,
    published: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Mark an episode as awaiting background processing."""
    with sqlite3.connect(db_path) as conn:
        # Insert only if we haven't seen this URL before
        conn.execute(
            """
            INSERT OR IGNORE INTO episodes (url, title, feed_id, status, published)
            VALUES (?, ?, ?, 'queued', ?)
            """,
            (url, title, feed_id, published),
        )
        conn.commit()


def update_episode_status(url: str, status: str, db_path: str = DB_PATH) -> None:
    """Update the processing status for an episode."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE episodes SET status = ? WHERE url = ?",
            (status, url),
        )  # simple status update used by the worker thread
        conn.commit()


def get_episode_by_id(episode_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Retrieve an episode by its database ID."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return cur.fetchone()


def delete_episode_by_id(episode_id: int, db_path: str = DB_PATH) -> None:
    """Delete an episode and all its associated articles and tickets."""
    with sqlite3.connect(db_path) as conn:
        # Delete associated tickets
        conn.execute("DELETE FROM jira_tickets WHERE episode_id = ?", (episode_id,))
        # Delete associated articles
        conn.execute("DELETE FROM articles WHERE episode_id = ?", (episode_id,))
        # Delete the episode
        conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        conn.commit()


def delete_episodes_bulk(episode_ids: List[int], db_path: str = DB_PATH) -> int:
    """Delete multiple episodes and all their associated data. Returns count deleted."""
    if not episode_ids:
        return 0
    placeholders = ",".join("?" * len(episode_ids))
    with sqlite3.connect(db_path) as conn:
        # Delete associated tickets
        conn.execute(
            f"DELETE FROM jira_tickets WHERE episode_id IN ({placeholders})",
            episode_ids,
        )
        # Delete associated articles
        conn.execute(
            f"DELETE FROM articles WHERE episode_id IN ({placeholders})",
            episode_ids,
        )
        # Delete the episodes
        cur = conn.execute(
            f"DELETE FROM episodes WHERE id IN ({placeholders})",
            episode_ids,
        )
        conn.commit()
        return cur.rowcount


def reset_episode_for_reprocess(episode_id: int, db_path: str = DB_PATH) -> None:
    """Clear episode data to prepare for reprocessing."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE episodes 
            SET transcript = NULL, summary = NULL, action_items = NULL, 
                status = 'queued', processed_at = NULL
            WHERE id = ?
            """,
            (episode_id,),
        )
        conn.commit()


def list_episodes(feed_id: int, db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """Return all episodes belonging to a particular feed."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM episodes WHERE feed_id = ? ORDER BY id",
            (feed_id,),
        )
        return cur.fetchall()


def list_all_episodes(order_by: str = "id", db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """List episodes from all feeds ordered by the provided column."""
    valid = {"id", "published", "processed_at"}
    column = order_by if order_by in valid else "id"
    direction = "DESC" if column in {"published", "processed_at"} else "ASC"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(f"SELECT * FROM episodes ORDER BY {column} {direction}")
        return cur.fetchall()


def add_ticket(
    episode_id: int,
    action_item: str,
    ticket_key: str,
    ticket_url: str,
    db_path: str = DB_PATH,
) -> None:
    """Save a JIRA ticket associated with an episode."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jira_tickets (episode_id, action_item, ticket_key, ticket_url)
            VALUES (?, ?, ?, ?)
            """,
            (episode_id, action_item, ticket_key, ticket_url),
        )
        conn.commit()


def list_tickets(
    episode_id: Optional[int] | None = None, db_path: str = DB_PATH
) -> List[sqlite3.Row]:
    """List all JIRA tickets or those for a specific episode."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        # Build a query that joins ticket info with episode context
        columns = (
            "jt.*, e.title AS episode_title, e.summary AS episode_summary,"
            " e.url AS episode_url, e.feed_id AS feed_id, e.published AS published"
        )
        if episode_id is None:
            # All tickets across every episode
            cur = conn.execute(
                f"""
                SELECT {columns}
                FROM jira_tickets jt
                JOIN episodes e ON jt.episode_id = e.id
                ORDER BY jt.id
                """
            )
        else:
            # Only tickets for a specific episode
            cur = conn.execute(
                f"""
                SELECT {columns}
                FROM jira_tickets jt
                JOIN episodes e ON jt.episode_id = e.id
                WHERE jt.episode_id = ?
                ORDER BY jt.id
                """,
                (episode_id,),
            )
        return cur.fetchall()


def delete_ticket(ticket_id: int, db_path: str = DB_PATH) -> bool:
    """Delete a JIRA ticket by its ID. Returns True if deleted."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM jira_tickets WHERE id = ?", (ticket_id,))
        conn.commit()
        return cur.rowcount > 0


def delete_tickets_bulk(ticket_ids: List[int], db_path: str = DB_PATH) -> int:
    """Delete multiple JIRA tickets by their IDs. Returns count deleted."""
    if not ticket_ids:
        return 0
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" for _ in ticket_ids)
        cur = conn.execute(
            f"DELETE FROM jira_tickets WHERE id IN ({placeholders})",
            ticket_ids,
        )
        conn.commit()
        return cur.rowcount


def add_article(
    episode_id: int,
    topic: str,
    style: str,
    content: str,
    db_path: str = DB_PATH,
) -> int:
    """Save a generated article and return its id."""
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO articles (episode_id, topic, style, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (episode_id, topic, style, content, created_at),
        )
        conn.commit()
        return cur.lastrowid


def get_article(article_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Retrieve a single article by its id."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT a.*, e.title AS episode_title, e.url AS episode_url, e.feed_id,
                   f.title AS podcast_title, f.url AS podcast_url
            FROM articles a
            JOIN episodes e ON a.episode_id = e.id
            LEFT JOIN feeds f ON e.feed_id = f.id
            WHERE a.id = ?
            """,
            (article_id,),
        )
        return cur.fetchone()


def update_article(
    article_id: int,
    topic: str | None = None,
    style: str | None = None,
    content: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Update an existing article's fields."""
    with sqlite3.connect(db_path) as conn:
        # Build update query dynamically based on provided fields
        updates = []
        params = []
        if topic is not None:
            updates.append("topic = ?")
            params.append(topic)
        if style is not None:
            updates.append("style = ?")
            params.append(style)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if updates:
            params.append(article_id)
            conn.execute(
                f"UPDATE articles SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()


def delete_article(article_id: int, db_path: str = DB_PATH) -> None:
    """Delete an article and its social posts by its id."""
    with sqlite3.connect(db_path) as conn:
        # Delete associated social posts first
        conn.execute("DELETE FROM social_posts WHERE article_id = ?", (article_id,))
        conn.execute("DELETE FROM articles WHERE id = ?", (article_id,))
        conn.commit()


def list_articles(
    episode_id: Optional[int] = None, db_path: str = DB_PATH
) -> List[sqlite3.Row]:
    """List articles, optionally filtered by episode."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if episode_id is None:
            cur = conn.execute(
                """
                SELECT a.*, e.title AS episode_title, e.url AS episode_url, e.feed_id,
                       f.title AS podcast_title
                FROM articles a
                JOIN episodes e ON a.episode_id = e.id
                LEFT JOIN feeds f ON e.feed_id = f.id
                ORDER BY a.created_at DESC
                """
            )
        else:
            cur = conn.execute(
                """
                SELECT a.*, e.title AS episode_title, e.url AS episode_url, e.feed_id,
                       f.title AS podcast_title
                FROM articles a
                JOIN episodes e ON a.episode_id = e.id
                LEFT JOIN feeds f ON e.feed_id = f.id
                WHERE a.episode_id = ?
                ORDER BY a.created_at DESC
                """,
                (episode_id,),
            )
        return cur.fetchall()


# --- Social Posts Functions ---


def add_social_post(
    article_id: int,
    platform: str,
    content: str,
    image_url: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Save a generated social media post and return its id."""
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO social_posts (article_id, platform, content, image_url, created_at, used)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (article_id, platform, content, image_url, created_at),
        )
        conn.commit()
        return cur.lastrowid


def list_social_posts(
    article_id: Optional[int] = None, db_path: str = DB_PATH
) -> List[sqlite3.Row]:
    """List social posts, optionally filtered by article."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if article_id is None:
            cur = conn.execute(
                """
                SELECT sp.*, a.topic AS article_topic
                FROM social_posts sp
                JOIN articles a ON sp.article_id = a.id
                ORDER BY sp.created_at DESC
                """
            )
        else:
            cur = conn.execute(
                """
                SELECT sp.*, a.topic AS article_topic
                FROM social_posts sp
                JOIN articles a ON sp.article_id = a.id
                WHERE sp.article_id = ?
                ORDER BY sp.platform, sp.id
                """,
                (article_id,),
            )
        return cur.fetchall()


def get_social_post(post_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Retrieve a single social post by its id, including the article topic."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT sp.*, a.topic AS article_topic
            FROM social_posts sp
            LEFT JOIN articles a ON sp.article_id = a.id
            WHERE sp.id = ?
            """,
            (post_id,),
        )
        return cur.fetchone()


def delete_social_post(post_id: int, db_path: str = DB_PATH) -> None:
    """Delete a social post by its id."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM social_posts WHERE id = ?", (post_id,))
        conn.commit()


def delete_social_posts_bulk(post_ids: List[int], db_path: str = DB_PATH) -> int:
    """Delete multiple social posts. Returns count deleted."""
    if not post_ids:
        return 0
    placeholders = ",".join("?" * len(post_ids))
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"DELETE FROM social_posts WHERE id IN ({placeholders})",
            post_ids,
        )
        conn.commit()
        return cur.rowcount


def delete_social_posts_for_article(article_id: int, db_path: str = DB_PATH) -> int:
    """Delete all social posts for an article. Returns count deleted."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM social_posts WHERE article_id = ?",
            (article_id,),
        )
        conn.commit()
        return cur.rowcount


def mark_social_post_used(post_id: int, used: bool = True, db_path: str = DB_PATH) -> None:
    """Mark a social post as used or unused."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE social_posts SET used = ? WHERE id = ?",
            (1 if used else 0, post_id),
        )
        conn.commit()


def update_social_post(post_id: int, content: str, db_path: str = DB_PATH) -> None:
    """Update the content of a social post."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE social_posts SET content = ? WHERE id = ?",
            (content, post_id),
        )
        conn.commit()


def bulk_replace_post_content(
    find_text: str,
    replace_text: str,
    post_type: str,
    case_sensitive: bool = False,
    whole_word: bool = False,
    post_ids: Optional[List[int]] = None,
    excluded_matches: Optional[Dict[str, bool]] = None,
    db_path: str = DB_PATH,
) -> int:
    """Replace text in posts of a given type, optionally filtered by post IDs.
    
    Args:
        find_text: The text to search for
        replace_text: The text to replace with
        post_type: 'social' for social_posts or 'standalone' for standalone_posts
        case_sensitive: Whether to do case-sensitive matching
        whole_word: Whether to match whole words only
        post_ids: Optional list of post IDs to limit replacement to
        excluded_matches: Optional dict of excluded matches {"postId-matchIndex": True}
        
    Returns:
        Number of posts that were modified
    """
    import re
    
    table_name = "social_posts" if post_type == "social" else "standalone_posts"
    excluded_matches = excluded_matches or {}
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Get posts with their content, optionally filtered by post_ids
        if post_ids:
            placeholders = ",".join("?" for _ in post_ids)
            cur = conn.execute(
                f"SELECT id, content FROM {table_name} WHERE id IN ({placeholders})",
                post_ids
            )
        else:
            cur = conn.execute(f"SELECT id, content FROM {table_name}")
        posts = cur.fetchall()
        
        affected_count = 0
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # Build pattern with optional word boundaries
        if whole_word:
            pattern = re.compile(r'\b' + re.escape(find_text) + r'\b', flags)
        else:
            pattern = re.compile(re.escape(find_text), flags)
        
        for post in posts:
            post_id = post['id']
            content = post['content'] or ''
            
            # Check if any matches in this post are NOT excluded
            matches = list(pattern.finditer(content))
            if not matches:
                continue
            
            # If there are excluded matches for this post, do selective replacement
            has_exclusions = any(f"{post_id}-{i+1}" in excluded_matches for i in range(len(matches)))
            
            if has_exclusions:
                # Do selective replacement - replace only non-excluded matches
                new_content = []
                last_end = 0
                for i, match in enumerate(matches):
                    match_key = f"{post_id}-{i+1}"
                    # Add content before this match
                    new_content.append(content[last_end:match.start()])
                    # Add either replacement or original based on exclusion
                    if match_key in excluded_matches:
                        new_content.append(match.group())  # Keep original
                    else:
                        new_content.append(replace_text)  # Replace
                    last_end = match.end()
                # Add remaining content after last match
                new_content.append(content[last_end:])
                new_content = ''.join(new_content)
            else:
                # No exclusions, replace all matches
                new_content = pattern.sub(replace_text, content)
            
            if new_content != content:
                conn.execute(
                    f"UPDATE {table_name} SET content = ? WHERE id = ?",
                    (new_content, post_id),
                )
                affected_count += 1
        
        conn.commit()
    
    return affected_count


# --- LinkedIn Token Functions ---


def save_linkedin_token(
    access_token: str,
    expires_at: str,
    member_id: str,
    user_urn: str,
    display_name: str | None = None,
    email: str | None = None,
    refresh_token: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Save or update LinkedIn OAuth tokens. Returns the token record id."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        # Check if we already have a token (single user mode)
        cur = conn.execute("SELECT id FROM linkedin_tokens LIMIT 1")
        existing = cur.fetchone()

        if existing:
            # Update existing token
            conn.execute(
                """
                UPDATE linkedin_tokens SET
                    access_token = ?,
                    refresh_token = ?,
                    expires_at = ?,
                    member_id = ?,
                    user_urn = ?,
                    display_name = ?,
                    email = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    access_token,
                    refresh_token,
                    expires_at,
                    member_id,
                    user_urn,
                    display_name,
                    email,
                    now,
                    existing[0],
                ),
            )
            conn.commit()
            return existing[0]
        else:
            # Insert new token
            cur = conn.execute(
                """
                INSERT INTO linkedin_tokens
                    (access_token, refresh_token, expires_at, member_id, user_urn,
                     display_name, email, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    access_token,
                    refresh_token,
                    expires_at,
                    member_id,
                    user_urn,
                    display_name,
                    email,
                    now,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid


def get_linkedin_token(db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get the stored LinkedIn token (single user mode)."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM linkedin_tokens LIMIT 1")
        return cur.fetchone()


def delete_linkedin_token(db_path: str = DB_PATH) -> None:
    """Delete all LinkedIn tokens (disconnect)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM linkedin_tokens")
        conn.commit()


def update_linkedin_token(
    access_token: str,
    expires_at: str,
    refresh_token: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Update the access token after a refresh."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        if refresh_token:
            conn.execute(
                """
                UPDATE linkedin_tokens SET
                    access_token = ?,
                    refresh_token = ?,
                    expires_at = ?,
                    updated_at = ?
                """,
                (access_token, refresh_token, expires_at, now),
            )
        else:
            conn.execute(
                """
                UPDATE linkedin_tokens SET
                    access_token = ?,
                    expires_at = ?,
                    updated_at = ?
                """,
                (access_token, expires_at, now),
            )
        conn.commit()


def update_linkedin_member_urn(
    member_id: str,
    user_urn: str | None = None,
    display_name: str | None = None,
    db_path: str = DB_PATH,
) -> bool:
    """Manually update the member ID and URN for LinkedIn posting.
    
    This is useful when the user only has w_member_social scope
    and profile endpoints don't work.
    
    Returns True if updated successfully.
    """
    now = datetime.utcnow().isoformat(timespec="seconds")
    if user_urn is None:
        user_urn = f"urn:li:person:{member_id}"
    
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT id FROM linkedin_tokens LIMIT 1")
        existing = cur.fetchone()
        if not existing:
            return False
        
        conn.execute(
            """
            UPDATE linkedin_tokens SET
                member_id = ?,
                user_urn = ?,
                display_name = COALESCE(?, display_name),
                updated_at = ?
            WHERE id = ?
            """,
            (member_id, user_urn, display_name, now, existing[0]),
        )
        conn.commit()
        return True


# --- Threads Token Functions ---


def save_threads_token(
    access_token: str,
    expires_at: str,
    user_id: str,
    username: str,
    display_name: str | None = None,
    profile_picture_url: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Save or update Threads OAuth tokens. Returns the token record id."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        # Check if we already have a token (single user mode)
        cur = conn.execute("SELECT id FROM threads_tokens LIMIT 1")
        existing = cur.fetchone()

        if existing:
            # Update existing token
            conn.execute(
                """
                UPDATE threads_tokens SET
                    access_token = ?,
                    expires_at = ?,
                    user_id = ?,
                    username = ?,
                    display_name = ?,
                    profile_picture_url = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    access_token,
                    expires_at,
                    user_id,
                    username,
                    display_name,
                    profile_picture_url,
                    now,
                    existing[0],
                ),
            )
            conn.commit()
            return existing[0]
        else:
            # Insert new token
            cur = conn.execute(
                """
                INSERT INTO threads_tokens
                    (access_token, expires_at, user_id, username,
                     display_name, profile_picture_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    access_token,
                    expires_at,
                    user_id,
                    username,
                    display_name,
                    profile_picture_url,
                    now,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid


def get_threads_token(db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get the stored Threads token (single user mode)."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM threads_tokens LIMIT 1")
        return cur.fetchone()


def delete_threads_token(db_path: str = DB_PATH) -> None:
    """Delete all Threads tokens (disconnect)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM threads_tokens")
        conn.commit()


def update_threads_token(
    access_token: str,
    expires_at: str,
    db_path: str = DB_PATH,
) -> None:
    """Update the access token after a refresh."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE threads_tokens SET
                access_token = ?,
                expires_at = ?,
                updated_at = ?
            """,
            (access_token, expires_at, now),
        )
        conn.commit()


def update_threads_user_info(
    user_id: str,
    username: str | None = None,
    display_name: str | None = None,
    db_path: str = DB_PATH,
) -> bool:
    """Update Threads user info fields manually (user_id, username, display_name).

    Returns True if a record was updated, False if no token exists.
    """
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT id FROM threads_tokens LIMIT 1")
        existing = cur.fetchone()
        if not existing:
            return False

        conn.execute(
            """
            UPDATE threads_tokens SET
                user_id = ?,
                username = COALESCE(?, username),
                display_name = COALESCE(?, display_name),
                updated_at = ?
            WHERE id = ?
            """,
            (user_id, username, display_name, now, existing[0]),
        )
        conn.commit()
        return True


# --- Facebook Token Functions ---


def save_facebook_token(
    access_token: str,
    expires_at: str,
    user_id: str,
    user_name: str | None = None,
    page_id: str | None = None,
    page_name: str | None = None,
    page_access_token: str | None = None,
    group_ids: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Save or update Facebook OAuth tokens. Returns the token record id."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT id FROM facebook_tokens LIMIT 1")
        existing = cur.fetchone()

        if existing:
            conn.execute(
                """
                UPDATE facebook_tokens SET
                    access_token = ?,
                    expires_at = ?,
                    user_id = ?,
                    user_name = ?,
                    page_id = ?,
                    page_name = ?,
                    page_access_token = ?,
                    group_ids = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    access_token,
                    expires_at,
                    user_id,
                    user_name,
                    page_id,
                    page_name,
                    page_access_token,
                    group_ids,
                    now,
                    existing[0],
                ),
            )
            conn.commit()
            return existing[0]
        else:
            cur = conn.execute(
                """
                INSERT INTO facebook_tokens
                    (access_token, expires_at, user_id, user_name,
                     page_id, page_name, page_access_token, group_ids,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    access_token,
                    expires_at,
                    user_id,
                    user_name,
                    page_id,
                    page_name,
                    page_access_token,
                    group_ids,
                    now,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid


def get_facebook_token(db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get the stored Facebook token (single user mode)."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM facebook_tokens LIMIT 1")
        return cur.fetchone()


def delete_facebook_token(db_path: str = DB_PATH) -> None:
    """Delete all Facebook tokens (disconnect)."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM facebook_tokens")
        conn.commit()


def update_facebook_token(
    access_token: str,
    expires_at: str,
    db_path: str = DB_PATH,
) -> None:
    """Update the access token after a refresh."""
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE facebook_tokens SET
                access_token = ?,
                expires_at = ?,
                updated_at = ?
            """,
            (access_token, expires_at, now),
        )
        conn.commit()


def update_facebook_page_selection(
    page_id: str,
    page_name: str,
    page_access_token: str,
    db_path: str = DB_PATH,
) -> bool:
    """Update the selected Facebook Page for posting.

    Returns True if a record was updated, False if no token exists.
    """
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT id FROM facebook_tokens LIMIT 1")
        existing = cur.fetchone()
        if not existing:
            return False

        conn.execute(
            """
            UPDATE facebook_tokens SET
                page_id = ?,
                page_name = ?,
                page_access_token = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (page_id, page_name, page_access_token, now, existing[0]),
        )
        conn.commit()
        return True


def update_facebook_group_ids(
    group_ids: str,
    db_path: str = DB_PATH,
) -> bool:
    """Update the selected Facebook Group IDs (comma-separated).

    Returns True if a record was updated, False if no token exists.
    """
    now = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT id FROM facebook_tokens LIMIT 1")
        existing = cur.fetchone()
        if not existing:
            return False

        conn.execute(
            """
            UPDATE facebook_tokens SET
                group_ids = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (group_ids, now, existing[0]),
        )
        conn.commit()
        return True


# --- Scheduled Posts Functions ---


def add_scheduled_post(
    scheduled_for: str,
    post_type: str,
    social_post_id: int | None = None,
    article_id: int | None = None,
    standalone_post_id: int | None = None,
    platform: str = "linkedin",
    status: str = "pending",
    linkedin_post_urn: str | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Add a post to the schedule queue. Returns the scheduled post id."""
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    posted_at = created_at if status == "posted" else None
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scheduled_posts
                (social_post_id, article_id, standalone_post_id, post_type, platform, scheduled_for,
                 status, linkedin_post_urn, created_at, posted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (social_post_id, article_id, standalone_post_id, post_type, platform, scheduled_for, 
             status, linkedin_post_urn, created_at, posted_at),
        )
        conn.commit()
        return cur.lastrowid


def get_scheduled_post(scheduled_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get a single scheduled post by id with joined content data."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT sp.*, 
                   soc.content AS social_content, soc.platform AS social_platform,
                   soc.image_url AS social_image_url,
                   a.topic AS article_topic, a.content AS article_content,
                   a.episode_id,
                   st.content AS standalone_content, st.platform AS standalone_platform,
                   st.image_url AS standalone_image_url
            FROM scheduled_posts sp
            LEFT JOIN social_posts soc ON sp.social_post_id = soc.id
            LEFT JOIN articles a ON sp.article_id = a.id
            LEFT JOIN standalone_posts st ON sp.standalone_post_id = st.id
            WHERE sp.id = ?
            """,
            (scheduled_id,),
        )
        return cur.fetchone()


def get_pending_schedules_for_social_posts(social_post_ids: List[int], db_path: str = DB_PATH) -> dict:
    """Get pending scheduled posts for a list of social post IDs.
    
    Returns a dict mapping social_post_id -> list of {platform, scheduled_for} dicts
    """
    if not social_post_ids:
        return {}
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in social_post_ids)
        cur = conn.execute(
            f"""
            SELECT social_post_id, platform, scheduled_for
            FROM scheduled_posts
            WHERE social_post_id IN ({placeholders})
            AND status = 'pending'
            ORDER BY scheduled_for ASC
            """,
            social_post_ids,
        )
        
        result = {}
        for row in cur.fetchall():
            post_id = row['social_post_id']
            if post_id not in result:
                result[post_id] = []
            result[post_id].append({
                'platform': row['platform'],
                'scheduled_for': row['scheduled_for'],
            })
        return result


def get_pending_schedules_for_standalone_posts(standalone_post_ids: List[int], db_path: str = DB_PATH) -> dict:
    """Get pending scheduled posts for a list of standalone post IDs.
    
    Returns a dict mapping standalone_post_id -> {platform: scheduled_for}
    """
    if not standalone_post_ids:
        return {}
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in standalone_post_ids)
        cur = conn.execute(
            f"""
            SELECT standalone_post_id, platform, scheduled_for
            FROM scheduled_posts
            WHERE standalone_post_id IN ({placeholders})
            AND status = 'pending'
            ORDER BY scheduled_for ASC
            """,
            standalone_post_ids,
        )
        
        result = {}
        for row in cur.fetchall():
            post_id = row['standalone_post_id']
            if post_id not in result:
                result[post_id] = {}
            # Store as platform -> scheduled_for dict for easy lookup
            result[post_id][row['platform']] = row['scheduled_for']
        return result


def get_posted_info_for_standalone_posts(standalone_post_ids: List[int], db_path: str = DB_PATH) -> dict:
    """Get posted info for a list of standalone post IDs.
    
    Returns a dict mapping standalone_post_id -> {platform: {url, posted_at}}
    Only includes posts that have been successfully posted.
    """
    if not standalone_post_ids:
        return {}
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" for _ in standalone_post_ids)
        cur = conn.execute(
            f"""
            SELECT standalone_post_id, platform, linkedin_post_urn, posted_at
            FROM scheduled_posts
            WHERE standalone_post_id IN ({placeholders})
            AND status = 'posted'
            ORDER BY posted_at DESC
            """,
            standalone_post_ids,
        )
        
        result = {}
        for row in cur.fetchall():
            post_id = row['standalone_post_id']
            if post_id not in result:
                result[post_id] = {}
            # Store the most recent posted info per platform
            if row['platform'] not in result[post_id]:
                result[post_id][row['platform']] = {
                    'url': row['linkedin_post_urn'],  # This stores URL/URN for all platforms
                    'posted_at': row['posted_at'],
                }
        return result


def list_scheduled_posts(
    status: str | None = None,
    platform: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_order: str = 'asc',
    db_path: str = DB_PATH,
) -> List[sqlite3.Row]:
    """List scheduled posts with optional filtering.
    
    Args:
        status: Filter by post status (pending, posted, failed, cancelled)
        platform: Filter by platform (linkedin, threads)
        date_from: Filter posts scheduled on or after this date (YYYY-MM-DD)
        date_to: Filter posts scheduled on or before this date (YYYY-MM-DD)
        sort_order: Sort order for scheduled_for column ('asc' or 'desc')
        db_path: Database path
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT sp.*, 
                   soc.content AS social_content, soc.platform AS social_platform,
                   soc.image_url AS social_image_url,
                   a.topic AS article_topic, a.content AS article_content,
                   st.content AS standalone_content, st.platform AS standalone_platform,
                   st.image_url AS standalone_image_url
            FROM scheduled_posts sp
            LEFT JOIN social_posts soc ON sp.social_post_id = soc.id
            LEFT JOIN articles a ON sp.article_id = a.id
            LEFT JOIN standalone_posts st ON sp.standalone_post_id = st.id
            WHERE 1=1
        """
        params = []

        if status:
            query += " AND sp.status = ?"
            params.append(status)
        if platform:
            query += " AND sp.platform = ?"
            params.append(platform)
        if date_from:
            query += " AND sp.scheduled_for >= ?"
            params.append(date_from)
        if date_to:
            # Include the entire end date by appending end-of-day time
            query += " AND sp.scheduled_for <= ?"
            params.append(date_to + "T23:59:59")

        # Dynamic sort order (validate to prevent SQL injection)
        order = "DESC" if sort_order.lower() == 'desc' else "ASC"
        query += f" ORDER BY sp.scheduled_for {order}"
        cur = conn.execute(query, params)
        return cur.fetchall()


def get_pending_scheduled_posts(db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """Get all pending scheduled posts that are due (scheduled_for <= now).
    
    Uses local time since time slots are configured in local time by users.
    """
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT sp.*, 
                   soc.content AS social_content, soc.platform AS social_platform,
                   soc.image_url AS social_image_url,
                   a.topic AS article_topic, a.content AS article_content,
                   a.episode_id,
                   st.content AS standalone_content, st.platform AS standalone_platform,
                   st.image_url AS standalone_image_url
            FROM scheduled_posts sp
            LEFT JOIN social_posts soc ON sp.social_post_id = soc.id
            LEFT JOIN articles a ON sp.article_id = a.id
            LEFT JOIN standalone_posts st ON sp.standalone_post_id = st.id
            WHERE sp.status = 'pending' AND sp.scheduled_for <= ?
            ORDER BY sp.scheduled_for ASC
            """,
            (now,),
        )
        return cur.fetchall()


def update_scheduled_post_time(
    scheduled_id: int,
    scheduled_for: str,
    db_path: str = DB_PATH,
) -> bool:
    """Update the scheduled time for a pending post.
    
    Returns True if updated successfully, False if post not found or not pending.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE scheduled_posts
            SET scheduled_for = ?
            WHERE id = ? AND status = 'pending'
            """,
            (scheduled_for, scheduled_id),
        )
        conn.commit()
        return cur.rowcount > 0


def redistribute_scheduled_posts(platform: str, db_path: str = DB_PATH) -> int:
    """Redistribute all pending posts for a platform to use the earliest available slots.
    
    This should be called when:
    - A time slot is added, deleted, or toggled
    - Daily posting limits are changed
    
    The function clears all existing scheduled times for pending posts of the platform,
    then reassigns them in order using get_next_available_slot().
    
    Args:
        platform: The platform to redistribute ('linkedin' or 'threads')
        db_path: Database path
        
    Returns:
        Number of posts redistributed
    """
    from datetime import datetime
    
    # Get all pending posts for this platform, ordered by their creation time
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id FROM scheduled_posts
            WHERE platform = ? AND status = 'pending'
            ORDER BY created_at ASC
            """,
            (platform,),
        )
        pending_posts = [row['id'] for row in cur.fetchall()]
    
    if not pending_posts:
        return 0
    
    # Clear all scheduled times first (set to far future temporarily)
    # This ensures get_next_available_slot doesn't see conflicts
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE scheduled_posts
            SET scheduled_for = '9999-12-31T23:59:59'
            WHERE platform = ? AND status = 'pending'
            """,
            (platform,),
        )
        conn.commit()
    
    # Reassign each post to the next available slot
    redistributed = 0
    for post_id in pending_posts:
        next_slot = get_next_available_slot(platform, db_path)
        if next_slot:
            update_scheduled_post_time(post_id, next_slot, db_path)
            redistributed += 1
        else:
            # No more slots available, leave at far future (will need manual intervention)
            pass
    
    return redistributed


def reorder_scheduled_posts(post_ids: List[int], db_path: str = DB_PATH) -> bool:
    """Reorder pending scheduled posts by swapping their scheduled times.
    
    Takes a list of post IDs in the desired new order. The scheduled_for times
    are preserved but reassigned based on the new order.
    
    Args:
        post_ids: List of scheduled post IDs in the desired order
        db_path: Database path
        
    Returns:
        True if successful, False otherwise
    """
    if not post_ids or len(post_ids) < 2:
        return True  # Nothing to reorder
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Get current scheduled times for all provided post IDs
        placeholders = ",".join("?" for _ in post_ids)
        cur = conn.execute(
            f"""
            SELECT id, scheduled_for FROM scheduled_posts
            WHERE id IN ({placeholders}) AND status = 'pending'
            ORDER BY scheduled_for ASC
            """,
            post_ids,
        )
        rows = cur.fetchall()
        
        if len(rows) < 2:
            return True  # Not enough posts to reorder
        
        # Get the times in chronological order (these are the slots we'll keep)
        times_in_order = sorted([row['scheduled_for'] for row in rows])
        
        # Now assign each post_id (in the new order) to a time slot (in chronological order)
        # This way, the first post in the user's new order gets the earliest time, etc.
        for i, post_id in enumerate(post_ids):
            if i < len(times_in_order):
                conn.execute(
                    "UPDATE scheduled_posts SET scheduled_for = ? WHERE id = ? AND status = 'pending'",
                    (times_in_order[i], post_id),
                )
        
        conn.commit()
    
    return True


def move_posts_to_position(
    post_ids: List[int],
    position: str,
    db_path: str = DB_PATH,
) -> bool:
    """Move selected pending posts to the top or bottom of the queue.
    
    Args:
        post_ids: List of scheduled post IDs to move
        position: 'top' to move to earliest times, 'bottom' to move to latest times
        db_path: Database path
        
    Returns:
        True if successful, False otherwise
    """
    if not post_ids:
        return True  # Nothing to move
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Get ALL pending posts ordered by scheduled_for
        cur = conn.execute(
            """
            SELECT id, scheduled_for FROM scheduled_posts
            WHERE status = 'pending'
            ORDER BY scheduled_for ASC
            """
        )
        all_posts = cur.fetchall()
        
        if len(all_posts) < 2:
            return True  # Not enough posts to reorder
        
        # Separate selected posts from non-selected posts
        selected_ids_set = set(post_ids)
        selected_posts = [p for p in all_posts if p['id'] in selected_ids_set]
        other_posts = [p for p in all_posts if p['id'] not in selected_ids_set]
        
        if not selected_posts:
            return True  # No selected posts found
        
        # Get all times in order
        all_times = sorted([p['scheduled_for'] for p in all_posts])
        
        # Create new ordering based on position
        if position == 'top':
            # Selected posts first, then others
            new_order = selected_posts + other_posts
        else:  # bottom
            # Others first, then selected posts
            new_order = other_posts + selected_posts
        
        # Assign times to new order
        for i, post in enumerate(new_order):
            if i < len(all_times):
                conn.execute(
                    "UPDATE scheduled_posts SET scheduled_for = ? WHERE id = ? AND status = 'pending'",
                    (all_times[i], post['id']),
                )
        
        conn.commit()
    
    return True


def update_scheduled_post_status(
    scheduled_id: int,
    status: str,
    linkedin_post_urn: str | None = None,
    error_message: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Update the status of a scheduled post."""
    with sqlite3.connect(db_path) as conn:
        posted_at = None
        if status == "posted":
            posted_at = datetime.utcnow().isoformat(timespec="seconds")

        conn.execute(
            """
            UPDATE scheduled_posts SET
                status = ?,
                linkedin_post_urn = ?,
                error_message = ?,
                posted_at = ?
            WHERE id = ?
            """,
            (status, linkedin_post_urn, error_message, posted_at, scheduled_id),
        )
        conn.commit()


def increment_retry_count(scheduled_id: int, db_path: str = DB_PATH) -> int:
    """Increment the retry count for a scheduled post and return the new value."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE scheduled_posts SET retry_count = COALESCE(retry_count, 0) + 1 WHERE id = ?",
            (scheduled_id,),
        )
        conn.commit()
        cur = conn.execute(
            "SELECT retry_count FROM scheduled_posts WHERE id = ?",
            (scheduled_id,),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def cancel_scheduled_post(scheduled_id: int, db_path: str = DB_PATH) -> bool:
    """Cancel a pending scheduled post. Returns True if cancelled."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE scheduled_posts SET status = 'cancelled'
            WHERE id = ? AND status = 'pending'
            """,
            (scheduled_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def cancel_scheduled_post_by_source(
    post_type: str,
    post_id: int,
    platform: str,
    db_path: str = DB_PATH,
) -> bool:
    """Cancel a pending scheduled post by its source post ID and platform.
    
    Args:
        post_type: 'social' or 'standalone'
        post_id: The social_post_id or standalone_post_id
        platform: 'linkedin' or 'threads'
        
    Returns True if a post was cancelled.
    """
    with sqlite3.connect(db_path) as conn:
        if post_type == 'social':
            cur = conn.execute(
                """
                UPDATE scheduled_posts SET status = 'cancelled'
                WHERE social_post_id = ? AND platform = ? AND status = 'pending'
                """,
                (post_id, platform),
            )
        elif post_type == 'standalone':
            cur = conn.execute(
                """
                UPDATE scheduled_posts SET status = 'cancelled'
                WHERE standalone_post_id = ? AND platform = ? AND status = 'pending'
                """,
                (post_id, platform),
            )
        else:
            return False
        conn.commit()
        return cur.rowcount > 0


def delete_scheduled_post(scheduled_id: int, db_path: str = DB_PATH) -> None:
    """Delete a scheduled post."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM scheduled_posts WHERE id = ?", (scheduled_id,))
        conn.commit()


def clear_pending_scheduled_posts(db_path: str = DB_PATH) -> int:
    """Clear all pending scheduled posts. Returns the count of deleted posts."""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM scheduled_posts WHERE status = 'pending'"
        )
        conn.commit()
        return cur.rowcount


def delete_scheduled_posts_bulk(post_ids: List[int], db_path: str = DB_PATH) -> int:
    """Delete multiple scheduled posts by their IDs. Returns count deleted."""
    if not post_ids:
        return 0
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" for _ in post_ids)
        cur = conn.execute(
            f"DELETE FROM scheduled_posts WHERE id IN ({placeholders})",
            post_ids,
        )
        conn.commit()
        return cur.rowcount


def get_scheduled_posts_for_article(
    article_id: int,
    db_path: str = DB_PATH,
) -> List[sqlite3.Row]:
    """Get all scheduled posts for a specific article."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT sp.*, soc.content AS social_content
            FROM scheduled_posts sp
            LEFT JOIN social_posts soc ON sp.social_post_id = soc.id
            WHERE sp.article_id = ? OR soc.article_id = ?
            ORDER BY sp.scheduled_for ASC
            """,
            (article_id, article_id),
        )
        return cur.fetchall()


# --- Schedule Time Slots Functions ---


def get_slot_platforms(slot_id: int, db_path: str = DB_PATH) -> List[str]:
    """Get the list of platforms assigned to a time slot.
    
    Returns an empty list if no platforms are explicitly assigned,
    which means the slot applies to ALL platforms.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT platform FROM time_slot_platforms WHERE slot_id = ? ORDER BY platform",
            (slot_id,),
        )
        return [row[0] for row in cur.fetchall()]


def set_slot_platforms(slot_id: int, platforms: List[str], db_path: str = DB_PATH) -> None:
    """Set the platforms for a time slot (replaces any existing assignments).
    
    Pass an empty list to make the slot apply to all platforms.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM time_slot_platforms WHERE slot_id = ?", (slot_id,))
        for platform in platforms:
            conn.execute(
                "INSERT INTO time_slot_platforms (slot_id, platform) VALUES (?, ?)",
                (slot_id, platform),
            )
        conn.commit()


def add_time_slot(
    day_of_week: int,
    time_slot: str,
    enabled: bool = True,
    platforms: List[str] | None = None,
    db_path: str = DB_PATH,
) -> int:
    """Add a new time slot for queue-based scheduling.
    
    Args:
        day_of_week: 0-6 (Monday-Sunday), or -1 for every day
        time_slot: Time in HH:MM format (24-hour)
        enabled: Whether the slot is active
        platforms: List of platform names this slot applies to.
                   None or empty list means all platforms.
    
    Returns:
        The ID of the created time slot
    """
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO schedule_time_slots (day_of_week, time_slot, enabled, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (day_of_week, time_slot, 1 if enabled else 0, created_at),
        )
        slot_id = cur.lastrowid
        # Store platform assignments if provided
        if platforms:
            for platform in platforms:
                conn.execute(
                    "INSERT INTO time_slot_platforms (slot_id, platform) VALUES (?, ?)",
                    (slot_id, platform),
                )
        conn.commit()
        return slot_id


def list_time_slots(db_path: str = DB_PATH) -> List[dict]:
    """Get all configured time slots ordered by day and time.
    
    Each returned dict includes a 'platforms' key with the list of
    platform names assigned to the slot (empty list = all platforms).
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT * FROM schedule_time_slots
            ORDER BY day_of_week ASC, time_slot ASC
            """
        )
        slots = []
        for row in cur.fetchall():
            slot = dict(row)
            slot['platforms'] = get_slot_platforms(slot['id'], db_path)
            slots.append(slot)
        return slots


def get_enabled_time_slots(platform: str | None = None, db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """Get only enabled time slots, optionally filtered by platform.
    
    Args:
        platform: If provided, only return slots that are assigned to this
                  platform (or slots with no platform restriction, i.e. all-platform slots).
        db_path: Database path
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if platform:
            # Return slots that either:
            # 1. Have no platform assignments (applies to all), OR
            # 2. Are explicitly assigned to this platform
            cur = conn.execute(
                """
                SELECT s.* FROM schedule_time_slots s
                WHERE s.enabled = 1
                AND (
                    NOT EXISTS (
                        SELECT 1 FROM time_slot_platforms tp WHERE tp.slot_id = s.id
                    )
                    OR EXISTS (
                        SELECT 1 FROM time_slot_platforms tp
                        WHERE tp.slot_id = s.id AND tp.platform = ?
                    )
                )
                ORDER BY s.day_of_week ASC, s.time_slot ASC
                """,
                (platform,),
            )
        else:
            cur = conn.execute(
                """
                SELECT * FROM schedule_time_slots
                WHERE enabled = 1
                ORDER BY day_of_week ASC, time_slot ASC
                """
            )
        return cur.fetchall()


def update_time_slot(
    slot_id: int,
    day_of_week: int | None = None,
    time_slot: str | None = None,
    enabled: bool | None = None,
    platforms: List[str] | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Update a time slot's settings.
    
    Args:
        slot_id: ID of the slot to update
        day_of_week: New day of week (None to leave unchanged)
        time_slot: New time (None to leave unchanged)
        enabled: New enabled state (None to leave unchanged)
        platforms: New platform list (None to leave unchanged;
                   empty list to apply to all platforms)
    """
    with sqlite3.connect(db_path) as conn:
        if day_of_week is not None:
            conn.execute(
                "UPDATE schedule_time_slots SET day_of_week = ? WHERE id = ?",
                (day_of_week, slot_id),
            )
        if time_slot is not None:
            conn.execute(
                "UPDATE schedule_time_slots SET time_slot = ? WHERE id = ?",
                (time_slot, slot_id),
            )
        if enabled is not None:
            conn.execute(
                "UPDATE schedule_time_slots SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, slot_id),
            )
        if platforms is not None:
            conn.execute(
                "DELETE FROM time_slot_platforms WHERE slot_id = ?", (slot_id,)
            )
            for platform in platforms:
                conn.execute(
                    "INSERT INTO time_slot_platforms (slot_id, platform) VALUES (?, ?)",
                    (slot_id, platform),
                )
        conn.commit()


def delete_time_slot(slot_id: int, db_path: str = DB_PATH) -> None:
    """Delete a time slot and its platform assignments."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM time_slot_platforms WHERE slot_id = ?", (slot_id,))
        conn.execute("DELETE FROM schedule_time_slots WHERE id = ?", (slot_id,))
        conn.commit()


def get_next_available_slot(platform: str = "linkedin", db_path: str = DB_PATH) -> str | None:
    """Find the next available time slot for scheduling on a specific platform.
    
    Each platform has its own queue, so a LinkedIn post and a Threads post
    can be scheduled for the same time slot without conflict.
    
    Also respects daily posting limits - if a platform has a max posts per day
    limit set, days that have reached that limit will be skipped.
    
    Args:
        platform: The platform to check slots for (e.g., 'linkedin', 'threads')
        db_path: Database path
    
    Returns the next datetime (ISO format) based on configured time slots
    that doesn't conflict with existing pending posts FOR THE SAME PLATFORM.
    
    Note: Uses LOCAL time for comparison since time slots are configured
    in local time by the user.
    
    Returns:
        ISO format datetime string, or None if no slots configured
    """
    from datetime import datetime, timedelta
    
    slots = get_enabled_time_slots(platform=platform, db_path=db_path)
    if not slots:
        return None
    
    # Get daily limit for this platform (0 = unlimited)
    daily_limit = get_daily_limit(platform, db_path)
    
    # Get existing pending posts for THIS PLATFORM ONLY to check for conflicts
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT scheduled_for FROM scheduled_posts
            WHERE status = 'pending' AND platform = ?
            """,
            (platform,),
        )
        existing_times = {row['scheduled_for'] for row in cur.fetchall()}
    
    # Use local time since time slots are configured in local time
    now = datetime.now()
    
    # Cache for daily post counts to avoid repeated DB queries
    daily_counts_cache = {}
    
    # Look up to 30 days ahead
    for day_offset in range(30):
        check_date = now + timedelta(days=day_offset)
        current_day_of_week = check_date.weekday()  # 0=Monday, 6=Sunday
        date_str = check_date.strftime('%Y-%m-%d')
        
        # Check daily limit if set
        if daily_limit > 0:
            if date_str not in daily_counts_cache:
                daily_counts_cache[date_str] = count_scheduled_posts_for_day(
                    platform, date_str, db_path
                )
            
            # Skip this day if limit reached
            if daily_counts_cache[date_str] >= daily_limit:
                continue
        
        for slot in slots:
            slot_day = slot['day_of_week']
            # -1 means every day
            if slot_day != -1 and slot_day != current_day_of_week:
                continue
            
            # Parse the time slot
            try:
                hour, minute = map(int, slot['time_slot'].split(':'))
            except (ValueError, AttributeError):
                continue
            
            # Create the candidate datetime
            candidate = check_date.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            
            # Skip if in the past
            if candidate <= now:
                continue
            
            # Check if this slot is already taken for THIS PLATFORM
            candidate_str = candidate.isoformat(timespec="seconds")
            if candidate_str not in existing_times:
                # Update cache to account for this new post if we were to add it
                if daily_limit > 0:
                    daily_counts_cache[date_str] = daily_counts_cache.get(date_str, 0) + 1
                return candidate_str
    
    return None


def initialize_default_time_slots(db_path: str = DB_PATH) -> None:
    """Create default time slots if none exist.
    
    Default slots: 9:00 AM, 12:00 PM, 5:00 PM every day
    """
    existing = list_time_slots(db_path)
    if existing:
        return  # Already have slots configured
    
    default_times = ["09:00", "12:00", "17:00"]
    for time in default_times:
        add_time_slot(
            day_of_week=-1,  # Every day
            time_slot=time,
            enabled=True,
            db_path=db_path,
        )


# =============================================================================
# Platform Daily Limits Functions
# =============================================================================


def get_daily_limit(platform: str, db_path: str = DB_PATH) -> int:
    """Get the max posts per day limit for a platform.
    
    Returns 0 if no limit is set (unlimited).
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT max_posts_per_day FROM platform_daily_limits WHERE platform = ?",
            (platform,),
        )
        row = cur.fetchone()
        return row[0] if row else 0


def set_daily_limit(platform: str, max_posts_per_day: int, db_path: str = DB_PATH) -> None:
    """Set the max posts per day limit for a platform.
    
    Set to 0 for unlimited posts.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO platform_daily_limits (platform, max_posts_per_day)
            VALUES (?, ?)
            ON CONFLICT(platform) DO UPDATE SET max_posts_per_day = excluded.max_posts_per_day
            """,
            (platform, max_posts_per_day),
        )
        conn.commit()


def get_all_daily_limits(db_path: str = DB_PATH) -> dict:
    """Get all platform daily limits as a dictionary.
    
    Returns: {'linkedin': 3, 'threads': 10, ...}
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT platform, max_posts_per_day FROM platform_daily_limits")
        return {row['platform']: row['max_posts_per_day'] for row in cur.fetchall()}


def count_scheduled_posts_for_day(platform: str, date_str: str, db_path: str = DB_PATH) -> int:
    """Count pending scheduled posts for a platform on a specific date.
    
    Args:
        platform: The platform (e.g., 'linkedin', 'threads')
        date_str: Date in YYYY-MM-DD format
        
    Returns:
        Number of pending posts scheduled for that day
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM scheduled_posts
            WHERE platform = ?
            AND status = 'pending'
            AND date(scheduled_for) = ?
            """,
            (platform, date_str),
        )
        return cur.fetchone()[0]


# =============================================================================
# Standalone Posts Functions (Command Center)
# =============================================================================


def add_standalone_post(
    source_type: str,
    source_content: str,
    platform: str,
    content: str,
    image_url: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Save a standalone post (not tied to an article) and return its id.
    
    Args:
        source_type: 'freeform', 'url', or 'text'
        source_content: The original prompt, URL, or text used to generate
        platform: Target platform (e.g., 'linkedin', 'threads', 'twitter')
        content: The generated post content
        image_url: Optional URL of an image to attach to the post
        
    Returns:
        The ID of the newly created post
    """
    created_at = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO standalone_posts (source_type, source_content, platform, content, image_url, created_at, used)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (source_type, source_content, platform, content, image_url, created_at),
        )
        conn.commit()
        return cur.lastrowid


def list_standalone_posts(
    source_type: Optional[str] = None,
    platform: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[sqlite3.Row]:
    """List standalone posts, optionally filtered by source type and/or platform.
    
    Args:
        source_type: Optional filter by source type ('freeform', 'url', 'text')
        platform: Optional filter by platform
        
    Returns:
        List of standalone post rows
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        conditions = []
        params = []
        
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)
        if platform:
            conditions.append("platform = ?")
            params.append(platform)
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        cur = conn.execute(
            f"""
            SELECT * FROM standalone_posts
            {where_clause}
            ORDER BY created_at DESC
            """,
            params,
        )
        return cur.fetchall()


def get_standalone_post(post_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Retrieve a single standalone post by its id.
    
    Args:
        post_id: The post ID
        
    Returns:
        The post row or None if not found
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM standalone_posts WHERE id = ?",
            (post_id,),
        )
        return cur.fetchone()


def update_standalone_post(
    post_id: int,
    content: str,
    image_url: Optional[str] = None,
    clear_image: bool = False,
    db_path: str = DB_PATH,
) -> None:
    """Update the content and optionally the image of a standalone post.
    
    Args:
        post_id: The post ID
        content: New content for the post
        image_url: Optional new image URL (only updated if provided or clear_image is True)
        clear_image: If True, remove the image (set to NULL)
    """
    with sqlite3.connect(db_path) as conn:
        if clear_image:
            conn.execute(
                "UPDATE standalone_posts SET content = ?, image_url = NULL WHERE id = ?",
                (content, post_id),
            )
        elif image_url is not None:
            conn.execute(
                "UPDATE standalone_posts SET content = ?, image_url = ? WHERE id = ?",
                (content, image_url, post_id),
            )
        else:
            conn.execute(
                "UPDATE standalone_posts SET content = ? WHERE id = ?",
                (content, post_id),
            )
        conn.commit()


def update_standalone_post_image(
    post_id: int,
    image_url: Optional[str],
    db_path: str = DB_PATH,
) -> None:
    """Update only the image URL of a standalone post.
    
    Args:
        post_id: The post ID
        image_url: New image URL (or None to remove image)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE standalone_posts SET image_url = ? WHERE id = ?",
            (image_url, post_id),
        )
        conn.commit()


def update_social_post_image(
    post_id: int,
    image_url: Optional[str],
    db_path: str = DB_PATH,
) -> None:
    """Update only the image URL of a social post.
    
    Args:
        post_id: The post ID
        image_url: New image URL (or None to remove image)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE social_posts SET image_url = ? WHERE id = ?",
            (image_url, post_id),
        )
        conn.commit()


def delete_standalone_post(post_id: int, db_path: str = DB_PATH) -> None:
    """Delete a standalone post by its id.
    
    Args:
        post_id: The post ID to delete
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM standalone_posts WHERE id = ?", (post_id,))
        conn.commit()


def delete_standalone_posts_bulk(post_ids: List[int], db_path: str = DB_PATH) -> int:
    """Delete multiple standalone posts. Returns count deleted.
    
    Args:
        post_ids: List of post IDs to delete
        
    Returns:
        Number of posts deleted
    """
    if not post_ids:
        return 0
    placeholders = ",".join("?" * len(post_ids))
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"DELETE FROM standalone_posts WHERE id IN ({placeholders})",
            post_ids,
        )
        conn.commit()
        return cur.rowcount


def mark_standalone_post_used(post_id: int, used: bool = True, db_path: str = DB_PATH) -> None:
    """Mark a standalone post as used or unused.
    
    Args:
        post_id: The post ID
        used: True to mark as used, False to mark as unused
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE standalone_posts SET used = ? WHERE id = ?",
            (1 if used else 0, post_id),
        )
        conn.commit()


# =============================================================================
# URL Sources CRUD Functions
# =============================================================================

def add_url_source(
    url: str,
    title: str,
    description: str,
    content: str,
    og_image: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Save extracted URL content for future reuse.
    
    If the URL already exists, updates the existing record.
    
    Args:
        url: The source URL
        title: Page title
        description: Meta description or og:description
        content: Extracted body text content
        og_image: Open Graph image URL (optional)
        
    Returns:
        The id of the inserted or updated record
    """
    created_at = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        # Check if URL already exists
        cur = conn.execute("SELECT id FROM url_sources WHERE url = ?", (url,))
        existing = cur.fetchone()
        
        if existing:
            # Update existing record
            conn.execute(
                """
                UPDATE url_sources 
                SET title = ?, description = ?, content = ?, og_image = ?, last_used_at = ?
                WHERE id = ?
                """,
                (title, description, content, og_image, created_at, existing[0]),
            )
            conn.commit()
            return existing[0]
        else:
            # Insert new record
            cur = conn.execute(
                """
                INSERT INTO url_sources (url, title, description, content, og_image, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, title, description, content, og_image, created_at, created_at),
            )
            conn.commit()
            return cur.lastrowid


def list_url_sources(db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """List all saved URL sources, ordered by last used date.
    
    Returns:
        List of url_sources rows
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT * FROM url_sources
            ORDER BY last_used_at DESC
            """
        )
        return cur.fetchall()


def get_url_source(source_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get a single URL source by ID.
    
    Args:
        source_id: The source ID
        
    Returns:
        The url_sources row or None if not found
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM url_sources WHERE id = ?",
            (source_id,),
        )
        return cur.fetchone()


def get_url_source_by_url(url: str, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get a URL source by its URL.
    
    Args:
        url: The source URL
        
    Returns:
        The url_sources row or None if not found
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM url_sources WHERE url = ?",
            (url,),
        )
        return cur.fetchone()


def delete_url_source(source_id: int, db_path: str = DB_PATH) -> bool:
    """Delete a URL source by ID.
    
    Args:
        source_id: The source ID to delete
        
    Returns:
        True if a row was deleted, False otherwise
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM url_sources WHERE id = ?",
            (source_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def update_url_source_last_used(source_id: int, db_path: str = DB_PATH) -> None:
    """Update the last_used_at timestamp for a URL source.
    
    Args:
        source_id: The source ID
    """
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE url_sources SET last_used_at = ? WHERE id = ?",
            (now, source_id),
        )
        conn.commit()


def update_url_source_content(
    source_id: int,
    title: str,
    description: str,
    content: str,
    og_image: Optional[str] = None,
    db_path: str = DB_PATH
) -> bool:
    """Update the content of a URL source (for re-extraction).
    
    Args:
        source_id: The source ID to update
        title: New title
        description: New description
        content: New extracted content
        og_image: New OG image URL (optional)
        
    Returns:
        True if updated successfully
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE url_sources 
            SET title = ?, description = ?, content = ?, og_image = ?
            WHERE id = ?
            """,
            (title, description, content, og_image, source_id),
        )
        conn.commit()
        return cur.rowcount > 0


# =============================================================================
# Uploaded Images Library Functions
# =============================================================================


def add_uploaded_image(
    filename: str,
    url: str,
    storage: str,
    size: int = 0,
    db_path: str = DB_PATH,
) -> int:
    """Save an uploaded image to the library.
    
    Args:
        filename: Original or generated filename
        url: The URL to access the image (local path or Cloudinary URL)
        storage: 'local' or 'cloudinary'
        size: File size in bytes
        
    Returns:
        The id of the inserted record
    """
    created_at = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        # Check if URL already exists (avoid duplicates)
        cur = conn.execute("SELECT id FROM uploaded_images WHERE url = ?", (url,))
        existing = cur.fetchone()
        if existing:
            return existing[0]
        
        cur = conn.execute(
            """
            INSERT INTO uploaded_images (filename, url, storage, size, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, url, storage, size, created_at),
        )
        conn.commit()
        return cur.lastrowid


def list_uploaded_images(db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """List all uploaded images, ordered by most recent first.
    
    Returns:
        List of uploaded_images rows
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT * FROM uploaded_images
            ORDER BY created_at DESC
            """
        )
        return cur.fetchall()


def delete_uploaded_image(image_id: int, db_path: str = DB_PATH) -> bool:
    """Delete an uploaded image record by ID.
    
    Args:
        image_id: The image ID to delete
        
    Returns:
        True if a row was deleted, False otherwise
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM uploaded_images WHERE id = ?",
            (image_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def get_uploaded_image(image_id: int, db_path: str = DB_PATH) -> Optional[sqlite3.Row]:
    """Get an uploaded image by ID.
    
    Args:
        image_id: The image ID
        
    Returns:
        The uploaded_images row or None if not found
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM uploaded_images WHERE id = ?",
            (image_id,),
        )
        return cur.fetchone()


# =============================================================================
# Recent Prompts Functions
# =============================================================================


def list_recent_prompts(limit: int = 20, db_path: str = DB_PATH) -> List[sqlite3.Row]:
    """Get unique recent freeform prompts from standalone_posts.
    
    Returns distinct prompts ordered by most recently used, so users can
    easily reuse previous prompts in the Command Center.
    
    Args:
        limit: Maximum number of prompts to return (default 20)
        
    Returns:
        List of rows with 'source_content' and 'created_at' fields
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT source_content, MAX(created_at) as created_at
            FROM standalone_posts
            WHERE source_type = 'freeform'
            AND source_content IS NOT NULL
            AND source_content != ''
            GROUP BY source_content
            ORDER BY MAX(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def clear_recent_prompts(db_path: str = DB_PATH) -> int:
    """Clear the prompt history by setting source_content to empty for freeform posts.
    
    This preserves the generated posts but removes them from the recent prompts list.
    
    Returns:
        Number of posts affected
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE standalone_posts
            SET source_content = ''
            WHERE source_type = 'freeform'
            AND source_content IS NOT NULL
            AND source_content != ''
            """
        )
        conn.commit()
        return cur.rowcount


def delete_prompt_by_content(prompt_content: str, db_path: str = DB_PATH) -> int:
    """Delete a specific prompt from history by clearing its source_content.
    
    This preserves the generated posts but removes the prompt from the history.
    
    Args:
        prompt_content: The exact prompt text to remove
        
    Returns:
        Number of posts affected
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE standalone_posts
            SET source_content = ''
            WHERE source_type = 'freeform'
            AND source_content = ?
            """,
            (prompt_content,),
        )
        conn.commit()
        return cur.rowcount


def delete_prompts_bulk(prompt_contents: List[str], db_path: str = DB_PATH) -> int:
    """Delete multiple prompts from history by clearing their source_content.
    
    Args:
        prompt_contents: List of prompt texts to remove
        
    Returns:
        Number of posts affected
    """
    if not prompt_contents:
        return 0
    
    with sqlite3.connect(db_path) as conn:
        placeholders = ",".join("?" for _ in prompt_contents)
        cur = conn.execute(
            f"""
            UPDATE standalone_posts
            SET source_content = ''
            WHERE source_type = 'freeform'
            AND source_content IN ({placeholders})
            """,
            prompt_contents,
        )
        conn.commit()
        return cur.rowcount
