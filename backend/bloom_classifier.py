# 📚 Simple keyword-based Bloom level classifier

bloom_keywords = {
    "Remember": ["define", "list", "recall", "identify", "state", "name","what is"],
    "Understand": ["explain", "describe", "summarize", "interpret", "discuss", "paraphrase"],
    "Apply": ["solve", "use", "demonstrate", "implement", "apply", "execute","what"],
    "Analyze": ["analyze", "differentiate", "compare", "examine", "categorize", "contrast","why"],
    "Evaluate": ["evaluate", "critique", "justify", "assess", "argue", "recommend","How","to"],
    "Create": ["design", "create", "formulate", "develop", "construct", "compose"]
}

def classify_bloom_level(question_text: str) -> str:
    """Classifies Bloom level of a question string."""
    question_text = question_text.lower()

    for level, keywords in bloom_keywords.items():
        if any(keyword in question_text for keyword in keywords):
            return level
    if question_text.startswith("what is") or question_text.startswith("define"):
        return "Remember"
    elif question_text.startswith("why") or question_text.startswith("explain"):
        return "Understand"
    elif question_text.startswith("how"):
        return "Apply"
    return "Unknown"

# 🧪 Test
if __name__ == "__main__":
    examples = [
        "Define neural networks.",
        "Explain how a decision tree works.",
        "Use k-means to cluster data.",
        "Analyze the results of your model.",
        "Evaluate which algorithm performs better.",
        "Create a CNN to classify images."
    ]

    for q in examples:
        print(f"{q} → {classify_bloom_level(q)}")
