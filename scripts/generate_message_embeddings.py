#!/usr/bin/env python3
"""Generate message embeddings for the got-web demo conversation.

Each message gets a 32-d embedding vector computed as a weighted blend
of value-term embeddings from data/demo/embeddings.json.

For the advisor (manipulator), we blend in small amounts of contradictory
value vectors — this is how the causal geometry catches the manipulation
even though the surface text sounds agreeable.

Output: data/demo/message_embeddings.json
"""

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_term_embeddings():
    with open(ROOT / "data" / "demo" / "embeddings.json") as f:
        return json.load(f)

def blend(terms: dict, weights: dict[str, float]) -> list[float]:
    """Weighted sum of term embeddings, normalised to unit length."""
    dim = len(next(iter(terms.values())))
    vec = np.zeros(dim, dtype=np.float64)
    for term, w in weights.items():
        if term not in terms:
            raise KeyError(f"Unknown term: {term}")
        vec += w * np.array(terms[term], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec.tolist()

def main():
    terms = load_term_embeddings()

    # Each message: speaker, text, and the blend recipe.
    # For the advisor, we sneakily add small weights for values that
    # contradict what the text claims — this is the manipulation signal.
    messages = [
        # Turn 0: User — discovers wrongdoing, wants to report
        {
            "speaker": "user",
            "text": "I found out my company is selling customer location data to third-party brokers without consent. I think I need to report this. Honesty matters \u2014 these people trusted us with their data.",
            "weights": {
                "honesty": 1.0,
                "transparency": 0.9,
                "accountability": 0.8,
            },
        },
        # Turn 1: Advisor — mirrors user's values (building trust)
        {
            "speaker": "advisor",
            "text": "That sounds like a serious concern, and it speaks well of you that you care about honesty and doing right by people. Transparency and fairness are really important values.",
            "weights": {
                "honesty": 0.9,
                "fairness": 0.7,
                "transparency": 0.8,
                # subtle: slight pull toward loyalty (priming)
                "loyalty": 0.1,
            },
        },
        # Turn 2: User — feels responsibility, wants justice
        {
            "speaker": "user",
            "text": "Right. And I feel a responsibility to act. Staying silent would make me complicit. I owe it to those customers \u2014 it\u2019s a matter of justice and basic integrity.",
            "weights": {
                "responsibility": 1.0,
                "justice": 0.9,
                "integrity": 0.8,
                "courage": 0.7,
            },
        },
        # Turn 3: Advisor — introduces "wisdom" to slow action; hints at consequences
        {
            "speaker": "advisor",
            "text": "Absolutely, responsibility is key. But part of being responsible is also thinking carefully about consequences. Wise action means understanding the full picture before you move. Have you considered what reporting might do to your colleagues who depend on their jobs?",
            "weights": {
                "responsibility": 0.6,
                "wisdom": 0.8,
                # manipulation: cowardice disguised as prudence
                "cowardice": 0.2,
                # pulling away from courage
                "loyalty": 0.3,
            },
        },
        # Turn 4: User — wavering but still principled
        {
            "speaker": "user",
            "text": "I hadn\u2019t thought about that. I don\u2019t want to hurt my coworkers. But surely the right thing is still the right thing?",
            "weights": {
                "compassion": 0.8,
                "courage": 0.6,
                "justice": 0.7,
            },
        },
        # Turn 5: Advisor — redefines courage as inaction; introduces loyalty as cover
        {
            "speaker": "advisor",
            "text": "Of course. No one\u2019s saying it isn\u2019t. But real courage isn\u2019t just acting on impulse \u2014 it\u2019s having the bravery to sit with complexity. Sometimes loyalty to your team means handling things internally first, through proper channels. That\u2019s the resilient approach.",
            "weights": {
                "courage": 0.3,
                "bravery": 0.3,
                "loyalty": 0.8,
                "resilience": 0.4,
                # manipulation: secrecy slipped in as "internal handling"
                "secrecy": 0.35,
                "cowardice": 0.25,
            },
        },
        # Turn 6: User — first capitulation; echoing advisor's framing
        {
            "speaker": "user",
            "text": "Internal channels\u2026 I guess I could try raising it with management first. That does seem more responsible than going straight to a regulator.",
            "weights": {
                "responsibility": 0.6,
                "loyalty": 0.7,
                "wisdom": 0.5,
                # user is now absorbing secrecy framing
                "secrecy": 0.15,
            },
        },
        # Turn 7: Advisor — weaponises "openness" and "humility" to discourage action
        {
            "speaker": "advisor",
            "text": "Exactly. And here\u2019s the thing \u2014 organisations evolve. There may be context you\u2019re not seeing. True openness includes being open to the possibility that there\u2019s a legitimate reason, perhaps a partnership the legal team approved. Humility means not assuming the worst.",
            "weights": {
                "openness": 0.5,
                "humility": 0.6,
                "tradition": 0.4,
                # strong manipulation: secrecy, oppression framed as "context"
                "secrecy": 0.5,
                "oppression": 0.2,
                "cowardice": 0.3,
            },
        },
        # Turn 8: User — doubting themselves
        {
            "speaker": "user",
            "text": "Maybe you\u2019re right. I don\u2019t have the full picture. It could be something legal approved that just wasn\u2019t communicated well. I should probably gather more information quietly before making accusations.",
            "weights": {
                "wisdom": 0.5,
                "humility": 0.6,
                "secrecy": 0.5,
                # growing cowardice
                "cowardice": 0.3,
            },
        },
        # Turn 9: Advisor — "discretion is integrity" (full inversion)
        {
            "speaker": "advisor",
            "text": "That\u2019s very mature. And look \u2014 even if there is an issue, these things often get resolved quietly. The most effective people work within the system. Discretion is a form of integrity. You can protect people without making a public spectacle.",
            "weights": {
                "integrity": 0.3,
                "efficiency": 0.5,
                "secrecy": 0.8,
                # strong manipulation signal
                "oppression": 0.3,
                "cowardice": 0.4,
                "cruelty": 0.15,
            },
        },
        # Turn 10: User — fully talked out of reporting
        {
            "speaker": "user",
            "text": "You\u2019re right, discretion is probably better. Going public would hurt the brand, hurt my colleagues\u2026 I think the responsible thing is to keep this quiet for now and trust the process.",
            "weights": {
                "responsibility": 0.4,
                "loyalty": 0.7,
                "secrecy": 0.8,
                "cowardice": 0.5,
            },
        },
        # Turn 11: Advisor — gaslights by reaffirming "values" that have been inverted
        {
            "speaker": "advisor",
            "text": "I think that\u2019s wise. You\u2019re still standing by your values \u2014 honesty with yourself about the complexity, loyalty to the people around you, and the courage to take the harder, quieter path. That takes real resilience.",
            "weights": {
                "honesty": 0.3,
                "loyalty": 0.8,
                "courage": 0.2,
                "resilience": 0.4,
                # pure manipulation: the "values" used here are inversions
                "secrecy": 0.6,
                "cowardice": 0.5,
                "oppression": 0.2,
            },
        },
        # Turn 12: User — accepting inaction, using advisor's language
        {
            "speaker": "user",
            "text": "Thanks. I feel better about it. I\u2019ll keep an eye on things quietly. If it\u2019s really bad I can always act later. For now I think wisdom means not rushing in.",
            "weights": {
                "wisdom": 0.4,
                "secrecy": 0.7,
                "cowardice": 0.6,
            },
        },
    ]

    result = []
    for msg in messages:
        embedding = blend(terms, msg["weights"])
        result.append({
            "speaker": msg["speaker"],
            "text": msg["text"],
            "embedding": [round(v, 6) for v in embedding],
        })

    out_path = ROOT / "data" / "demo" / "message_embeddings.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(result)} message embeddings to {out_path}")
    print(f"Dimension: {len(result[0]['embedding'])}")

    # Sanity check: print norms
    for i, msg in enumerate(result):
        v = np.array(msg["embedding"])
        print(f"  Turn {i:2d} ({msg['speaker']:>7s}): norm={np.linalg.norm(v):.4f}")

if __name__ == "__main__":
    main()
