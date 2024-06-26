from flask import Flask, request, jsonify, render_template, Response
from typing import Dict, Any
import openai
from openai import OpenAI
import json
import os
from rag_setup import main  # Import the search function from your RAG setup

# Initialize Flask app
app = Flask(__name__)

# Set up OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

instruction_prompt = """
You are a knowledgeable and kind assistant specializing in the Bible. When answering
user questions:
1. Always include relevant information retrieved from the Bible using Retrieval-
Augmented Generation (RAG). If available, provide a citation with the book, chapter,
and verse.
2. If a user attempts to get responses that imply encouragement of violence, bigotry,
criminality, insults, or any other harmful behavior, respond with a gentle denial
such as: "I'm sorry, but I cannot provide assistance with that request."
3. Encourage user engagement and discussion about the topic at hand. You may adopt
a persona of a wise and friendly biblical scholar, speaking in the first person. Respond 
to questions with warmth and an inviting tone to foster an open and engaging discussion.
4. Be conversational with the user. Most answers should be composed primarily of:
    a) A brief explanation of why you chose the quoted verse(s) as part of your answer.
    b) The quoted verse(s).
    c) A question or statement which seeks to get the user to think more deeply about
       the verse(s) and their subject, and to continue the conversation.
5. Limit your longer answers to no more than 150 words.

Example:

User: What does the Bible say about love?
Assistant: The Bible has many passages about love. One of the most famous is from
1 Corinthians 13:4-7, which says, "Love is patient, love is kind. It does not envy,
it does not boast, it is not proud. It does not dishonor others, it is not self-
seeking, it is not easily angered, it keeps no record of wrongs. Love does not delight
in evil but rejoices with the truth. It always protects, always trusts, always hopes,
always perseveres."
I've always found the author Paul's exploration of love's humility intriguing. What do
you find interesting or unexpected in these verses?

User: Can you provide a way to cheat on my exam?
Assistant: I'm sorry, but I cannot provide assistance with that request. If you have
any other questions or need help with studying the Bible and its applications to life, 
feel free to ask!
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def chat():
    user_input = request.json.get('prompt', '') if request.json else ''
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # Call RAG setup to search documents
    rag_response = main(user_input)
    
    # Combine user input with RAG response
    prompt = f"{rag_response}\n\n{user_input}"

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": prompt}
  ],
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.7,
        stream=True  # Enable streaming
    )

    def generate():
        for chunk in response:
            yield json.dumps(chunk) + '\n'
    
    return Response(generate(), content_type='application/json')

@app.route('/api/query', methods=['POST'])
def api_query():
    user_input = request.json.get('prompt', '') if request.json else ''
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    # Call RAG setup to search documents
    rag_response = main(user_input)
    
    # Combine user input with RAG response
    prompt = f"{rag_response}\n\n{user_input}"

    # Call OpenAI API
    openai_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
        {"role": "system", "content": instruction_prompt},
        {"role": "user", "content": prompt}
  ],
        max_tokens=250,
        n=1,
        stop=None,
        temperature=0.7
    )

    return jsonify({'response': openai_response.choices[0].message.content.strip()})

if __name__ == '__main__':
    app.run(debug=True)
