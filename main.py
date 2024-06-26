from flask import Flask, request, jsonify, render_template, Response, session
from typing import Dict, Any
import openai
from openai import OpenAI
import json
import os
from rag_setup import main  # Import the search function from your RAG setup
from flask_session import Session

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # For session encryption
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Set up OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

MAX_CONTEXT_LENGTH = 1000  # Adjust based on your needs and model limits
SUMMARY_THRESHOLD = 1500  # When to summarize

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

def summarize_conversation(conversation):
    summary_prompt = f"Summarize the following conversation concisely:\n\n{conversation}"
    summary_response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can use a smaller model for summarization
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=100
    )
    return summary_response.choices[0].message.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    user_input = request.json.get('prompt', '') if request.json else ''
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    if 'conversation' not in session:
        session['conversation'] = []

    # Add user input to conversation history
    session['conversation'].append({"role": "user", "content": user_input})

    # Check if we need to summarize
    full_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session['conversation']])
    if len(full_conversation) > SUMMARY_THRESHOLD:
        summary = summarize_conversation(full_conversation)
        session['conversation'] = [{"role": "system", "content": f"Conversation summary: {summary}"}]

    # Prepare context for the current query
    context = session['conversation'][-MAX_CONTEXT_LENGTH:]

    try:
        # Call RAG setup to search documents
        rag_response = main(user_input)
        
        # Combine user input with RAG response
        prompt = f"{rag_response}\n\n{user_input}"

        messages = [{"role": "system", "content": instruction_prompt}] + context + [{"role": "user", "content": prompt}]

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=250,
            n=1,
            stop=None,
            temperature=0.7,
            stream=True  # Enable streaming
        )

        def generate():
            try:
                last_content = ""
                for chunk in response:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            new_content = content
                            yield json.dumps({"content": new_content, "finished": False}) + '\n'
                            last_content += new_content
                yield json.dumps({"content": "", "finished": True}) + '\n'
            except Exception as e:
                app.logger.error(f"Error in generate function: {str(e)}")
                yield json.dumps({"error": "An error occurred while generating the response"}) + '\n'

        return Response(generate(), content_type='application/json')

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An internal error occurred'}), 500


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


    # user_input = request.json.get('prompt', '') if request.json else ''
