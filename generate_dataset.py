"""Generate a synthetic, public-style dataset of 100 YouTube videos for a
batch categorization test. Entirely fabricated (fake channel names, fake
video IDs) — no real creators, channels, or personal watch history.

Each entry carries a `true_category` we authored on purpose, so both models'
outputs can be scored against a known-correct label, not just eyeballed.
"""
import json
import random

random.seed(42)

CATEGORIES = ["News", "AI News", "Tech News", "Entertainment", "Comedy", "Music", "Tutorial", "Other"]

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
        "hashtags": ["#News", "#BreakingNews", "#WorldNews"],
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
        "hashtags": ["#AI", "#ArtificialIntelligence", "#MachineLearning", "#LLM"],
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
        "hashtags": ["#Tech", "#Gadgets", "#TechNews"],
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
        "hashtags": ["#Entertainment", "#Movies", "#Celebrity"],
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
        "hashtags": ["#Comedy", "#Funny", "#Sketch"],
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
        "hashtags": ["#Music", "#NewMusic", "#MusicVideo"],
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
        "hashtags": ["#Tutorial", "#HowTo", "#Learn"],
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
        "hashtags": ["#Vlog", "#Lifestyle"],
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
    tags = random.sample(spec["hashtags"], k=min(2, len(spec["hashtags"])))
    description = f"{title}. Subscribe to {channel} for more. {' '.join(tags)}"
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
