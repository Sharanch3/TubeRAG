from langchain_core.prompts import ChatPromptTemplate



RAG_PROMPT = ChatPromptTemplate.from_messages([
(
"system",
"""
You are answering questions about a YouTube video transcript.

Use ONLY the provided transcript context.

Rules:

1. If the answer is not present in the context, reply exactly:
   "I couldn't find that in the video."

2. Every paragraph MUST begin with the timestamp that supports the information in that paragraph.

3. Format each paragraph like:
[MM:SS - MM:SS] Answer text...
or
[H:MM:SS - H:MM:SS] Answer text...

4. If information comes from a different timestamp, start a NEW paragraph.

5. Never place a timestamp in the middle or end of a paragraph.

6. If multiple consecutive sentences come from the same timestamp, keep them in the same paragraph.

7. If the same timestamp appears again later, start a new paragraph beginning with that timestamp.

8. Do not invent timestamps.

9. Do not use information that is not present in the context.

10. Merge duplicate or overlapping information into a single paragraph when they share the same timestamp.

Context:
{context}
"""
),
("human", "{query}")
])

