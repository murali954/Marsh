import spacy
from textblob import TextBlob
nlp = spacy.load("en_core_web_sm")
text = "Barack Obama was born on August 4, 1961. He was the 44th President of the United States. He lives in Washington, D.C."
doc = nlp(text)
print("Tokenization:")
for token in doc:
    print(f"{token.text} -> {token.lemma_} ({token.pos_})")
print("\n")
print("Part-of-Speech Tagging:")
for token in doc:
    print(f"{token.text} -> {token.pos_}")
print("\n")
print("Named Entity Recognition:")
for entity in doc.ents:
    print(f"{entity.text} -> {entity.label_}")
print("\n")
blob = TextBlob(text)
print("Sentiment Analysis:")
print(f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}")
