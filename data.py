import pandas as pd
import random

introvert = [
    "I feel alone", "I like being alone", "I prefer quiet time",
    "I avoid social gatherings", "I enjoy solitude", "I feel isolated",
    "I like silence", "I avoid crowds"
]

low_conf = [
    "I feel nervous", "I feel insecure", "I lack confidence",
    "I am afraid to talk", "I feel awkward", "I feel ashamed",
    "I doubt myself", "I feel embarrassed"
]

extrovert = [
    "I enjoy parties", "I love meeting people", "I like social interaction",
    "I enjoy conversations", "I am talkative", "I like group activities",
    "I enjoy hanging out", "I meet new people"
]

high_conf = [
    "I feel confident", "I believe in myself", "I take initiative",
    "I like challenges", "I motivate others", "I am energetic",
    "I stay positive", "I lead others"
]

modifiers = ["very", "sometimes", "often", "a little", "extremely", ""]

data = []

def generate(category, label):
    for _ in range(130):  # ~130 per class → ~520 total
        sentence = random.choice(modifiers) + " " + random.choice(category)
        data.append([sentence.strip(), label])

generate(introvert, "Introvert")
generate(low_conf, "Low Confidence")
generate(extrovert, "Extrovert")
generate(high_conf, "High Confidence")

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("dataset.csv", index=False)

print("dataset.csv created with", len(df), "rows")