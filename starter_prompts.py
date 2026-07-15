"""Built-in starter prompts for the Prompt Library.

These are curated, ready-to-use prompt templates surfaced in the Command Center
Prompt Library modal ("Starter Templates"). They are defined in code (not the
database) so they always ship with the app, can't be accidentally deleted, and
stay versioned. Users can Insert one into a field to tweak, or "Save a copy" to
add an editable version to their own library.

Design intent: every template encodes anti-"AI slop" guidance and is
contextualized to a cybersecurity practitioner's content + business-development
needs (blogs, videos, and posts about specific security service areas). Fill the
[BRACKETED] placeholders before generating.

Grouped by ``category``; ``STARTER_CATEGORY_ORDER`` controls display order.
"""

from typing import Dict, List

# Display order + emoji for each category shown in the modal.
STARTER_CATEGORY_ORDER: List[str] = [
    "Foundation",
    "Promote Content",
    "Business Development",
    "Industry Angles",
    "Thought Leadership",
    "Community & Engagement",
]

STARTER_CATEGORY_EMOJI: Dict[str, str] = {
    "Foundation": "⭐",              # ⭐
    "Promote Content": "\U0001F4E3",     # 📣
    "Business Development": "\U0001F3AF", # 🎯
    "Industry Angles": "\U0001F3E2",     # 🏢
    "Thought Leadership": "\U0001F4A1",  # 💡
    "Community & Engagement": "\U0001F91D",  # 🤝
}


STARTER_PROMPTS: List[Dict[str, str]] = [
    # ---------------------------------------------------------------- Foundation
    {
        "category": "Foundation",
        "title": "My anti-slop house rules",
        "content": (
            "Write in my voice, not generic AI marketing copy. Non-negotiables:\n"
            "- Open with a specific hook: a real scenario, a concrete number, or a "
            "sharp question. Never \"In today's digital landscape\" or \"In an "
            "ever-evolving threat landscape.\"\n"
            "- One clear idea per post. Don't try to say everything.\n"
            "- Plain English. If a term needs jargon, explain it in the same breath.\n"
            "- Have a real point of view. Take a position.\n"
            "- No buzzword stacking (leverage, robust, seamless, cutting-edge, "
            "synergy, game-changer). No fear-mongering or FUD.\n"
            "- Vary sentence length so it reads like a person, not a template. Avoid "
            "the \"it's not just X, it's Y\" cadence and cliche transitions like "
            "\"let's dive in.\"\n"
            "- End with a genuine, low-pressure call to action (a question or an "
            "offer), never \"Contact us today to learn more!\"\n"
            "- Match the platform's vibe. Sound like a practitioner who has actually "
            "done the work."
        ),
    },

    # ---------------------------------------------------------- Promote Content
    {
        "category": "Promote Content",
        "title": "Promote a blog post",
        "content": (
            "I published a blog titled \"[BLOG TITLE]\" — link: [URL].\n"
            "The core takeaway is: [ONE-SENTENCE MAIN POINT].\n\n"
            "Write posts that make a busy [AUDIENCE, e.g. IT director / practice "
            "owner] stop scrolling and click through.\n"
            "- Open with a specific hook pulled from the piece — a scenario, a "
            "stat, or a pointed question. No generic intros.\n"
            "- Build the whole post around ONE concrete idea from the blog. Don't "
            "summarize the whole thing; leave a reason to click.\n"
            "- Sound like a practitioner talking to a peer, not a brochure. Clear "
            "point of view, plain English, no buzzwords, no FUD.\n"
            "- Close with a low-key invite to read the full post — a reason to "
            "click, not \"Read more today!\""
        ),
    },
    {
        "category": "Promote Content",
        "title": "Promote a video / YouTube",
        "content": (
            "I just posted a video: \"[VIDEO TITLE]\" — link: [URL].\n"
            "It covers [WHAT IT COVERS]. The one thing I want people to walk away "
            "with: [KEY TAKEAWAY].\n\n"
            "Write posts that tease the value without giving everything away.\n"
            "- Lead with the payoff: who this is for and what they'll be able to do "
            "after watching.\n"
            "- Use a real hook, not \"Check out my new video!\" Pull the most "
            "surprising or useful moment forward.\n"
            "- Human and specific. No hype words, no clickbait promises.\n"
            "- End by telling people exactly why it's worth their [X] minutes, then "
            "the link."
        ),
    },
    {
        "category": "Promote Content",
        "title": "Turn a blog into a thread / breakdown",
        "content": (
            "Break my blog \"[BLOG TITLE]\" ([URL]) into a punchy [X/Threads thread "
            "OR LinkedIn breakdown].\n"
            "- Each point = one concrete, usable idea a [AUDIENCE] can act on. No "
            "filler lines, no \"let's dive in,\" no restating the obvious.\n"
            "- Start with a hook strong enough to earn the second line.\n"
            "- Keep a consistent, human voice throughout — short sentences, real "
            "examples, a clear POV.\n"
            "- End with the link and one honest reason the full piece is worth "
            "reading."
        ),
    },

    # ------------------------------------------------------ Business Development
    {
        "category": "Business Development",
        "title": "Penetration testing (vs. a scan)",
        "content": (
            "Write posts for [AUDIENCE, e.g. a mid-market IT leader] on why real "
            "penetration testing beats an automated vulnerability scan.\n"
            "- Ground it in something concrete: a class of finding scanners miss, a "
            "realistic attack path (chaining low-severity issues into real access), "
            "or what a human tester does that a tool can't.\n"
            "- Position: we test end-to-end and hand you findings you can actually "
            "act on — offensive and defensive expertise, clear remediation.\n"
            "- Voice: confident practitioner, not salesy. No \"hackers are "
            "everywhere\" FUD.\n"
            "- Close with a soft invite to talk through a scoped test — a "
            "question or offer, not a hard pitch."
        ),
    },
    {
        "category": "Business Development",
        "title": "HIPAA risk assessments (healthcare)",
        "content": (
            "Audience: [healthcare practice leaders / compliance officers].\n"
            "Write posts that make HIPAA risk assessments feel practical, not scary "
            "or bureaucratic.\n"
            "- Use a concrete example of a gap that trips up real practices (e.g. "
            "[EXAMPLE: unencrypted backups, no documented risk analysis, ex-employee "
            "access]).\n"
            "- Explain what a proper risk assessment actually surfaces — and why "
            "\"we have a firewall\" is not a risk analysis.\n"
            "- Calm, credible, plain English. No regulatory scare tactics.\n"
            "- Close with an offer to walk through where a practice usually stands "
            "before an auditor does."
        ),
    },
    {
        "category": "Business Development",
        "title": "CMMC readiness (gov contractors)",
        "content": (
            "Audience: defense-industrial-base and government contractors facing "
            "CMMC.\n"
            "Write posts that cut through the confusion.\n"
            "- Explain what CMMC readiness really involves and the trap of waiting "
            "until an RFP requires it.\n"
            "- Make one specific, useful point per post (e.g. the difference between "
            "\"we think we're compliant\" and evidence an assessor will accept).\n"
            "- Steadying and specific, not alarmist. Position a readiness assessment "
            "as the thing that de-risks the timeline.\n"
            "- End with a low-pressure invite to assess where they stand today."
        ),
    },
    {
        "category": "Business Development",
        "title": "vCISO & security leadership",
        "content": (
            "Audience: growing organizations that need security leadership but not a "
            "full-time CISO yet.\n"
            "Write posts about the real cost of nobody owning security strategy — "
            "decisions that stall, risks nobody's tracking, board questions with no "
            "answer.\n"
            "- Make it concrete with a scenario, not abstractions.\n"
            "- Position fractional / interim CISO leadership as the bridge: strategy, "
            "roadmap, and someone accountable.\n"
            "- Voice: seasoned advisor who has sat in that seat.\n"
            "- Soft CTA: invite them to name the leadership gap they feel most."
        ),
    },
    {
        "category": "Business Development",
        "title": "Ransomware readiness & tabletops",
        "content": (
            "Write posts on ransomware readiness for [execs / IT leaders].\n"
            "Angle: the organizations that recover fast aren't lucky — they "
            "practiced.\n"
            "- Describe what a tabletop exercise actually reveals: who calls whom, "
            "whether the backups are truly tested, whether the plan survives contact "
            "with a real incident.\n"
            "- A little provocative (\"hope is not a recovery plan\") without "
            "fear-mongering.\n"
            "- Practical and calm. One clear idea per post.\n"
            "- Close by inviting them to run a tabletop before an attacker schedules "
            "one for them."
        ),
    },
    {
        "category": "Business Development",
        "title": "Board-level security reporting",
        "content": (
            "Audience: executives and board members.\n"
            "Write posts about how to talk cyber risk to a board without drowning "
            "them in red/yellow/green dashboards.\n"
            "- Tie security to business impact: dollars, decisions, and risk the "
            "board actually owns.\n"
            "- Give one concrete before/after (e.g. \"we blocked 4,000 threats\" vs. "
            "\"here's the one risk that could stop operations, and what it costs to "
            "fix\").\n"
            "- Voice: business-fluent and credible, not technical for its own sake.\n"
            "- Position our reporting/analytics as translating technical risk into "
            "language leadership can act on. Soft invite to help them report better."
        ),
    },

    # ------------------------------------------------------------ Industry Angles
    {
        "category": "Industry Angles",
        "title": "Tailor to any industry (fill-in)",
        "content": (
            "Take my topic — [YOUR TOPIC OR SERVICE, e.g. ransomware readiness, a "
            "pen test finding, a blog] — and tailor it for the [INDUSTRY] audience.\n"
            "- Ground it in that industry's real world: the regulations they answer "
            "to, the threats that actually hit them, and who signs off on security.\n"
            "- Use the words insiders use, not generic \"organizations face "
            "evolving threats\" filler.\n"
            "- One concrete, industry-specific example beats three abstract points.\n"
            "- Plain English, clear POV, no FUD. Close with a next step that fits how "
            "this industry actually buys and decides."
        ),
    },
    {
        "category": "Industry Angles",
        "title": "Healthcare",
        "content": (
            "Tailor my topic — [YOUR TOPIC OR SERVICE] — for a healthcare audience "
            "(practice administrators, compliance officers, health-system CISOs).\n"
            "Their reality:\n"
            "- HIPAA / HITECH and protected health information (PHI); business "
            "associates and third-party risk.\n"
            "- The real stakes aren't just fines — ransomware can halt care delivery "
            "and put patient safety on the line.\n"
            "- Medical devices and legacy systems that can't just be patched.\n"
            "Write it in their language. One concrete scenario (e.g. an EHR outage, a "
            "vendor breach) beats abstractions. Calm and credible, no scare tactics. "
            "Close with a next step framed around patient safety and continuity of "
            "care, not just compliance."
        ),
    },
    {
        "category": "Industry Angles",
        "title": "Financial Services",
        "content": (
            "Tailor my topic — [YOUR TOPIC OR SERVICE] — for financial services "
            "(banks, credit unions, wealth/advisory firms; audience: CISO, risk, and "
            "compliance leaders).\n"
            "Their reality:\n"
            "- GLBA Safeguards Rule, PCI DSS, FFIEC guidance, SOX, and living under "
            "examiner scrutiny.\n"
            "- Fraud, wire/business-email-compromise, and account takeover are "
            "daily-money problems, not hypotheticals.\n"
            "- Customer trust and third-party/vendor risk carry real weight.\n"
            "Speak to examiner-readiness and protecting customer funds and data. One "
            "concrete example (a BEC attempt, a failed control an examiner flags) "
            "lands better than generalities. Credible and precise. Soft CTA tied to "
            "reducing exam findings or fraud exposure."
        ),
    },
    {
        "category": "Industry Angles",
        "title": "Nonprofit",
        "content": (
            "Tailor my topic — [YOUR TOPIC OR SERVICE] — for a nonprofit audience "
            "(executive directors, boards, development staff).\n"
            "Their reality:\n"
            "- Lean budgets and lean (or outsourced) IT — security has to be "
            "practical and affordable.\n"
            "- Donor data and payment info to protect; grant and funder security "
            "requirements to meet.\n"
            "- A phishing or BEC hit that diverts funds is an existential, "
            "mission-and-trust problem.\n"
            "Frame security as protecting the mission and donor trust, not as "
            "enterprise overkill. Give one high-value, low-cost move. Warm, practical, "
            "no jargon. Close with a next step that respects a tight budget."
        ),
    },
    {
        "category": "Industry Angles",
        "title": "State & Local Government",
        "content": (
            "Tailor my topic — [YOUR TOPIC OR SERVICE] — for state and local "
            "government (IT directors, agency leaders, elected officials).\n"
            "Their reality:\n"
            "- CJIS, StateRAMP, public-records and (where relevant) election-security "
            "expectations.\n"
            "- Ransomware against municipalities can take down services residents "
            "depend on — 911, permits, payments.\n"
            "- Tight procurement, legacy systems, and public accountability for every "
            "dollar.\n"
            "Speak to continuity of public services and taxpayer/constituent trust. "
            "One concrete municipal scenario beats abstractions. Steady and "
            "non-alarmist. Close with a next step that fits public-sector procurement "
            "and transparency."
        ),
    },
    {
        "category": "Industry Angles",
        "title": "Higher Education",
        "content": (
            "Tailor my topic — [YOUR TOPIC OR SERVICE] — for higher education "
            "(CISOs, CIOs, IT leaders at colleges and universities).\n"
            "Their reality:\n"
            "- FERPA (student records), GLBA (student financial data), sometimes "
            "HIPAA (student health / research), plus valuable research data and IP.\n"
            "- A decentralized, open environment: many departments, BYOD everywhere, "
            "a huge attack surface, and a culture of academic freedom that resists "
            "lockdown.\n"
            "- Ransomware and research-targeted espionage are real.\n"
            "Acknowledge the tension between openness and security instead of "
            "pretending it away. One concrete campus scenario. Respectful of "
            "faculty/researcher realities. Close with a next step that works in a "
            "federated, budget-constrained environment."
        ),
    },
    {
        "category": "Industry Angles",
        "title": "Retail",
        "content": (
            "Tailor my topic — [YOUR TOPIC OR SERVICE] — for retail (IT, loss "
            "prevention, franchise or multi-location owners).\n"
            "Their reality:\n"
            "- PCI DSS and cardholder data; point-of-sale malware and e-commerce "
            "skimming (Magecart-style) attacks.\n"
            "- Uptime and fraud pressure spike during peak/holiday season.\n"
            "- Multi-location and franchise models multiply the attack surface and "
            "make consistency hard.\n"
            "Speak to protecting payment data and staying up during peak. One "
            "concrete scenario (a skimmer, a POS breach, a holiday-season outage) "
            "beats theory. Practical and clear. Close with a next step framed around "
            "peak-season readiness and cardholder-data protection."
        ),
    },

    # ---------------------------------------------------------- Thought Leadership
    {
        "category": "Thought Leadership",
        "title": "Myth-buster post",
        "content": (
            "Pick a common cybersecurity myth [MYTH — e.g. \"we're too small to "
            "be a target,\" \"compliance means we're secure,\" \"the firewall "
            "protects us\"].\n"
            "Write a post that:\n"
            "- Names the myth in the hook.\n"
            "- Explains why it's wrong with ONE concrete example.\n"
            "- Gives the better mental model to replace it.\n"
            "Strong point of view, respectful — no dunking on people. This is "
            "where I sound like someone who has done the work. End with a question "
            "that invites people to share their own take."
        ),
    },
    {
        "category": "Thought Leadership",
        "title": "Lesson from the field",
        "content": (
            "Write a post built around a real, anonymized lesson from doing this "
            "work: \"[SHORT DESCRIPTION OF THE STORY / FINDING]\".\n"
            "Structure: the situation → what we found → why it mattered "
            "→ the takeaway anyone can use.\n"
            "- No client-identifying details. Change specifics if needed.\n"
            "- Human and concrete — this is the opposite of AI slop. Real detail "
            "is what makes it land.\n"
            "- End with the one thing you want readers to go check in their own "
            "environment today."
        ),
    },
    {
        "category": "Thought Leadership",
        "title": "React to a breach or trend",
        "content": (
            "A story just broke: [HEADLINE / LINK].\n"
            "Write a post with MY take — not a rehash of the news.\n"
            "- What's the real lesson? What would I tell a client to do differently?\n"
            "- What's overblown vs. what actually matters?\n"
            "- Measured and credible. No hot-take cringe, no ambulance chasing.\n"
            "- Tie it to one practical thing a [AUDIENCE] can do this week."
        ),
    },

    # ------------------------------------------------------ Community & Engagement
    {
        "category": "Community & Engagement",
        "title": "Encourage people breaking into cyber",
        "content": (
            "Write a post for people trying to break into cybersecurity — Show "
            "Up Show Out energy: direct, encouraging, real.\n"
            "- Share ONE concrete, non-obvious piece of advice [ADVICE / TOPIC] "
            "instead of \"get certs and network.\"\n"
            "- Speak to someone who feels behind or locked out of the field.\n"
            "- Warm and motivating, zero corporate stiffness, zero toxic "
            "positivity.\n"
            "- End by inviting them to reply with where they're stuck."
        ),
    },
    {
        "category": "Community & Engagement",
        "title": "Discussion starter / poll",
        "content": (
            "Write a short post that asks [AUDIENCE] a genuine question about [TOPIC] "
            "to spark real replies.\n"
            "- Make it something people actually have opinions on — not "
            "engagement bait.\n"
            "- Give my own quick answer so it doesn't read like a survey.\n"
            "- Conversational, human, a little bold. Keep it tight."
        ),
    },
    {
        "category": "Community & Engagement",
        "title": "Explain a concept in plain English",
        "content": (
            "Explain \"[CONCEPT — e.g. MFA fatigue, phishing-resistant auth, "
            "attack surface]\" to a non-technical [AUDIENCE] in plain English.\n"
            "- Use ONE everyday analogy that actually fits (don't force it).\n"
            "- No jargon without an immediate translation.\n"
            "- The vibe: the friend who happens to do security and can make it make "
            "sense.\n"
            "- End with the single action that's actually worth taking."
        ),
    },
]


def grouped_starter_prompts() -> List[Dict[str, object]]:
    """Return starter prompts grouped by category in display order.

    Each group is ``{"category", "emoji", "prompts": [...]}`` and each prompt
    carries a stable ``idx`` (its position in ``STARTER_PROMPTS``) plus a short
    ``preview`` for compact rendering.
    """
    groups: List[Dict[str, object]] = []
    for category in STARTER_CATEGORY_ORDER:
        prompts = []
        for idx, p in enumerate(STARTER_PROMPTS):
            if p["category"] != category:
                continue
            content = p["content"]
            prompts.append({
                "idx": idx,
                "title": p["title"],
                "content": content,
                "preview": content[:110] + ("..." if len(content) > 110 else ""),
            })
        if prompts:
            groups.append({
                "category": category,
                "emoji": STARTER_CATEGORY_EMOJI.get(category, ""),
                "prompts": prompts,
            })
    return groups
