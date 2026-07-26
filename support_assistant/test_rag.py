from app.rag import retrieve

docs, ids, meta = retrieve(
    "What is the delivery fee?"
)

print(ids)
print()

for doc in docs:
    print(doc[:250])
    print("-" * 60)