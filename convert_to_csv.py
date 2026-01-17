import pandas as pd

texts = []
emotions = []

with open("train.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if ";" in line:
            text, emotion = line.split(";")
            texts.append(text)
            emotions.append(emotion)

df = pd.DataFrame({
    "text": texts,
    "emotion": emotions
})

df.to_csv("emotions.csv", index=False)

print("✅ emotions.csv created successfully")
print(df.head())
