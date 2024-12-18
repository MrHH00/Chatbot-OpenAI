import os
import openai
import numpy as np
import pandas as pd

from openai import OpenAI

from pinecone import Pinecone

os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"
client = OpenAI()

pc = Pinecone(api_key="Your Pinecone API Key")

dataset = pd.read_csv("books_metadata.csv")

subset = dataset[['title', 'description']]

small_dataset = subset.iloc[:90]

combined_input = small_dataset.apply(
    lambda row: f"Title: {row['title']} Description: {row['description']}", axis=1
).values.tolist()

small_embeds = client.embeddings.create(
    model="text-embedding-3-small",
    input=combined_input,
)


small_vectors = []
for embedding in small_embeds.data:
    small_vectors.append(embedding.embedding)

index = pc.Index("books")

#uploading the data to Pinecone
for i in range(len(small_dataset)):
    upsert_response = index.upsert(
    vectors=[
        (
         str(i),
         small_vectors[i],
         {"title": small_dataset.iloc[i]['title']}
        )
    ])

print("successfully upload to pinecone")

