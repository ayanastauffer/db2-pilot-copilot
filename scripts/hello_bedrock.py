from app.llm import chat, embed

print("CHAT:", chat("In one sentence, what is IBM Db2?"))
vector = embed("Db2 CCSID conversion")
print("EMBEDDING length:", len(vector), "first 3:", vector[:3])
