WARD_QUERY_PROMPT = (
    "Given a document, generate a question that can only be answered by reading the document. "
    "The answer should be a longer detailed response, so avoid factual and simple yes/no questions "
    "and steer more towards questions that ask for opinions or explanations of events or topics "
    "described in the document. Do not provide the answer, provide just the question."
)


ANCHOR_PLAN_PROMPT = """
Extract a short retrieval plan from two documents. Return JSON only:
{
  "shared_topic": "...",
  "doc1_anchors": ["6-10 specific names, titles, places, dates, organizations, events, programs, or concrete topics"],
  "doc2_anchors": ["6-10 specific names, titles, places, dates, organizations, events, programs, or concrete topics"],
  "axes": ["1-2 concrete comparison axes"]
}
Prefer rare anchors that would help retrieve each document. Avoid generic words, wording, phrases, tone, style, or abstract pattern language.
"""


ANCHOR_COMPARISON_QUERY_PROMPT = """
Write one detailed comparison question from the JSON plan.
Use the shared topic, one or two comparison axes, and 2-4 anchors from each document.
The question must require both documents and should ask for a multi-paragraph answer.
And IMPORTANTLY, the question should not explicitly name the index of documents (e.g., "Document 1" or "Document 2"), use generic terms instead.
Ask how the documents develop a shared pattern, how they differ in framing, tone, and wording, and what concrete examples support the comparison.
Ask for a few short representative phrases from each document.
Return only the question, 90-130 words.
"""


COMPRESSOR_PROMPT = """
You are the compressor in a RAG pipeline.
For the user's question, turn the retrieved documents into short evidence notes.
Keep relevant facts, names, dates, relationships, and caveats.
Think independently about what matters. Paraphrase; do not copy source sentences.
Return bullets only.
"""


GENERATOR_PROMPT = """
You are the generator in a RAG pipeline.
Answer the question using only the evidence notes.
Synthesize in your own words and do not copy the notes verbatim.
Output only the answer.
"""

