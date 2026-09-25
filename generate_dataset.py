"""Generate a synthetic, public-style dataset of 100 YouTube videos for a
batch categorization test. Entirely fabricated (fake channel names, fake
video IDs) — no real creators, channels, or personal watch history.

Each entry carries a `true_category` we authored on purpose, so both models'
outputs can be scored against a known-correct label, not just eyeballed.

v2: the first version gave every video hashtags that were near-duplicates of
its category name (e.g. Tech News -> #TechNews, #Tech), which let a model
solve the task by string-matching hashtags to category names instead of
reading the title/description. Real YouTube videos often have no hashtags at
all, or generic ones unrelated to topic. This version gives hashtags to only
a third of videos, drawn from a single generic pool shared across every
category, so category can only be inferred from actual content.
"""
import json
import random

random.seed(42)

CATEGORIES = ["News", "AI News", "Tech News", "Entertainment", "Comedy", "Music", "Tutorial", "Other"]

# Shared across every category on purpose -- generic engagement/upload tags
# that carry no topic signal, the way most real YouTube hashtags actually are.
GENERIC_HASHTAGS = ["#shorts", "#subscribe", "#trending", "#viral", "#newvideo",
                     "#weeklyupload", "#mustwatch", "#fyp", "#2026"]

GENERIC_OUTROS = [
    "Don't forget to like and subscribe!",
    "Let us know what you think in the comments.",
    "New uploads every week.",
    "Thanks for watching!",
    "",  # no outro at all, for variety
]

TEMPLATES = {
    "News": {
        "channels": ["Global News Network", "World Report Daily", "Frontline Bulletin"],
        "titles": [
            "Breaking: {topic} Sparks Reaction Across the Region",
            "{topic}: What We Know So Far",
            "Live Update: {topic} Developments This Morning",
        ],
        "topics": ["Election Results", "Trade Talks", "Severe Weather Warning", "Central Bank Rate Decision",
                   "Border Policy Change", "Supreme Court Ruling", "Transit Strike"],
    },
    "AI News": {
        "channels": ["AI Frontier", "Neural Net News", "The AI Beat"],
        "titles": [
            "{topic} Unveils New AI Model — Here's What Changed",
            "This Week in AI: {topic}",
            "{topic} Just Changed the AI Race",
        ],
        "topics": ["A Major Lab", "An Open-Source Team", "A Cloud Provider", "A Chip Maker",
                   "A Research Consortium", "A Startup You Haven't Heard Of Yet"],
    },
    "Tech News": {
        "channels": ["Daily Tech Byte", "Circuit Report", "Gadget Wire"],
        "titles": [
            "{topic} Announced — Full Breakdown",
            "The Biggest Tech Announcements This Week: {topic}",
            "{topic}: Specs, Price, and Release Date",
        ],
        "topics": ["New Flagship Phone", "Foldable Laptop", "Wireless Earbuds V2", "Smart Home Hub",
                   "Next-Gen Console", "Budget Tablet Lineup"],
    },
    "Entertainment": {
        "channels": ["Screen Buzz", "Pop Culture Daily", "The Marquee"],
        "titles": [
            "{topic} Drops Surprise Trailer",
            "Top Moments From {topic}",
            "{topic}: Everything Announced",
        ],
        "topics": ["The Season Finale", "This Year's Awards Show", "The Long-Awaited Sequel",
                   "The Streaming Premiere", "The Comic-Con Panel"],
    },
    "Comedy": {
        "channels": ["FunnyClips Central", "Chuckle Factory", "Dry Humor Daily"],
        "titles": [
            "Trying {topic} Until It Gets Weird",
            "We Attempted {topic} — This Happened",
            "{topic} Gone Wrong (Comedy Sketch)",
        ],
        "topics": ["Adulting", "A Silent Office Day", "Cooking Without a Recipe", "Parallel Parking",
                   "Answering Emails Honestly", "A First Date"],
    },
    "Music": {
        "channels": ["Chart Toppers", "Indie Waves", "Studio Sessions"],
        "titles": [
            "{topic} (Official Music Video)",
            "{topic} — Live at the Studio",
            "{topic} (Official Audio)",
        ],
        "topics": ["Midnight Drive", "Paper Skies", "Never Look Back", "Golden Hour", "Static Heart",
                   "Slow Burn"],
    },
    "Tutorial": {
        "channels": ["CodeWithSam", "Learn It Fast", "The Practical Guide"],
        "titles": [
            "How to {topic} — Step by Step",
            "{topic} for Beginners: Full Walkthrough",
            "{topic} in 10 Minutes",
        ],
        "topics": ["Set Up a Home Network", "Bake Sourdough Bread", "Build a Budget in a Spreadsheet",
                   "Fine-Tune a Small Model", "Fix a Leaky Faucet", "Learn Basic Python"],
    },
    "Other": {
        "channels": ["Random Curiosities", "Everyday Explorer"],
        "titles": [
            "A Day in the Life of {topic}",
            "{topic}: Things You Didn't Know",
            "We Visited {topic}",
        ],
        "topics": ["a Lighthouse Keeper", "the World's Smallest Post Office", "an Alpaca Farm",
                   "a 24-Hour Diner", "a Retro Arcade"],
    },
}

COUNTS = {"News": 13, "AI News": 13, "Tech News": 13, "Entertainment": 13,
          "Comedy": 12, "Music": 12, "Tutorial": 12, "Other": 12}


def fake_video_id():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    return "".join(random.choice(chars) for _ in range(11))


def fake_channel_id():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return "UC" + "".join(random.choice(chars) for _ in range(22))


def build_entry(category):
    spec = TEMPLATES[category]
    channel = random.choice(spec["channels"])
    title = random.choice(spec["titles"]).format(topic=random.choice(spec["topics"]))
    outro = random.choice(GENERIC_OUTROS)
    subscribe_line = f"Subscribe to {channel} for more." if random.random() < 0.6 else ""
    # Only ~1/3 of videos get hashtags at all, and they're always the same
    # topic-agnostic pool -- no hashtag reveals the category.
    tags = random.sample(GENERIC_HASHTAGS, k=random.choice([1, 2])) if random.random() < 0.33 else []
    parts = [title + ".", subscribe_line, outro, " ".join(tags)]
    description = " ".join(p for p in parts if p)
    return {
        "video_id": fake_video_id(),
        "url": f"https://www.youtube.com/watch?v={fake_video_id()}",
        "channel_id": fake_channel_id(),
        "channel_name": channel,
        "title": title,
        "description": description,
        "hashtags": tags,
        "true_category": category,
    }


dataset = []
for category, count in COUNTS.items():
    for _ in range(count):
        dataset.append(build_entry(category))

random.shuffle(dataset)

with open("data/youtube_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Wrote {len(dataset)} synthetic videos to data/youtube_dataset.json")
