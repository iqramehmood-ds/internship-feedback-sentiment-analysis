# Task 7: Internship Feedback Sentiment Analysis
# Using NLTK VADER to classify feedback as positive, neutral or negative

import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

# 1. Feedback data
# Note: this is synthetic (sample) data written for practice, because no
# real dataset was provided. It is not real intern feedback.
data = [
    ("Survey", "The mentors were very supportive and always answered my questions."),
    ("Survey", "I learned a lot and enjoyed working with Python and data cleaning."),
    ("Survey", "Tasks were interesting and helped me build real skills."),
    ("Survey", "The task instructions were unclear and no dataset was provided."),
    ("Survey", "Communication was slow and I waited days for a reply."),
    ("Survey", "It was okay, some tasks were average."),
    ("Survey", "The internship portal was easy to use."),
    ("Survey", "The workload was manageable and the tasks were well planned."),
    ("Survey", "Great experience, I would recommend it to other students."),
    ("Survey", "The deadlines were fine and the weekly tasks were clear."),
    ("Comment", "Loved the machine learning task, it was challenging and fun."),
    ("Comment", "Not enough guidance for beginners."),
    ("Comment", "Task 4 was confusing and the fraud detection data was not given."),
    ("Comment", "The certificate and the experience are valuable for my CV."),
    ("Comment", "Average internship, nothing special."),
    ("Comment", "Support team never replied and I was very disappointed."),
    ("Comment", "Very well organized program with clear weekly tasks."),
    ("Comment", "I submitted all tasks on time."),
    ("Comment", "The video task was confusing and a waste of time."),
    ("Comment", "Disappointed that there was no mentor session."),
    ("Social Media", "Just finished my data analytics internship, so proud of what I built!"),
    ("Social Media", "Learning pandas and sklearn through this internship, highly useful."),
    ("Social Media", "Honestly the worst communication I have seen from an internship."),
    ("Social Media", "Internship tasks are going on, week 3 now."),
    ("Social Media", "Amazing opportunity for students to practice real projects."),
    ("Social Media", "Waste of time, tasks have no proper explanation."),
    ("Social Media", "The platform kept crashing and I lost my work."),
    ("Social Media", "Waiting for the next task update."),
    ("Social Media", "Submitted my work today. Waiting for review."),
    ("Social Media", "Terrible experience with the submission portal."),
]
df = pd.DataFrame(data, columns=["source", "feedback"])

# Check the data before analysis
print("Total feedback:", len(df))
print("Missing feedback:", df["feedback"].isna().sum())
print()


# 2. Basic cleaning (remove links and mentions, extra spaces)
def clean(text):
    text = re.sub(r"http\S+|@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


df["clean_feedback"] = df["feedback"].apply(clean)

# 3. Get the compound score for each comment (-1 to +1)
sia = SentimentIntensityAnalyzer()
df["compound"] = df["clean_feedback"].apply(lambda t: sia.polarity_scores(t)["compound"])


# 4. Convert the score into a label (>= 0.05 positive, <= -0.05 negative)
def label(score):
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


df["sentiment"] = df["compound"].apply(label)

# 5. Show results
order = ["Positive", "Neutral", "Negative"]
counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
pct = (counts / len(df) * 100).round(1)
print("Overall sentiment distribution")
print(pd.DataFrame({"count": counts, "percent": pct}))

by_source = pd.crosstab(df["source"], df["sentiment"]).reindex(columns=order, fill_value=0)
print("\nSentiment by source")
print(by_source)

print("\nAverage compound score by source")
print(df.groupby("source")["compound"].mean().round(3))

print("\nMost negative comments")
print(df.nsmallest(5, "compound")[["source", "feedback", "compound"]].to_string(index=False))

print("\nMost positive comments")
print(df.nlargest(5, "compound")[["source", "feedback", "compound"]].to_string(index=False))

# 6. Charts
colors = {"Positive": "#2e8b57", "Neutral": "#9e9e9e", "Negative": "#c0392b"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].bar(counts.index, counts.values, color=[colors[c] for c in counts.index])
ax[0].set_title("Overall Sentiment (synthetic sample data)")
ax[0].set_ylabel("Number of comments")

by_source.plot(kind="bar", stacked=True, ax=ax[1], legend=False,
               color=[colors[c] for c in by_source.columns])
ax[1].legend(by_source.columns, loc="upper left", bbox_to_anchor=(1, 1))
ax[1].set_title("Sentiment by Source (synthetic)")
ax[1].set_xlabel("")
ax[1].tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.savefig("sentiment_charts.png", dpi=150)
plt.show()

# 7. Save the labeled data
df.to_csv("classified_feedback.csv", index=False)
