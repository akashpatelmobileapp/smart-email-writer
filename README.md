# Smart Email Writer

A command-line assistant that drafts, polishes, and shortens emails automatically, then lets you ask follow-up questions such as suggesting relevant attachments. Built with [LangChain](https://python.langchain.com/), the [Mistral](https://mistral.ai/) LLM, and [Tavily](https://tavily.com/) web search.

## What it does

1. You provide a **topic** and a **tone** (formal / casual / friendly).
2. The app runs the input through a 3-step chain, with no further interaction:
   - **Draft** — writes an email about the topic in the chosen tone.
   - **Grammar** — fixes and improves the draft.
   - **Shorten** — trims it to under 150 words while keeping the meaning.
3. The final email is printed.
4. An interactive **agent loop** starts so you can request follow-ups. Type `add attachment suggestion` to have the agent search the web (via Tavily) for relevant documents to attach, or `exit` to quit.

## How it works

The core logic lives in [app.py](app.py).

### The email pipeline

Three prompt templates are each turned into a chain (`prompt | llm | parser`) and composed sequentially. Because each step outputs a plain string but the next prompt expects a dict, a small `RunnableLambda` rewraps the string between steps:

```
draft_chain
  | RunnableLambda(lambda x: {"email": x})
  | grammar_chain
  | RunnableLambda(lambda x: {"email": x})
  | shorten_chain
```

The whole pipeline is executed with a single `email_chain.invoke({"topic": topic, "tone": tone})`.

### The follow-up agent

After the email is generated, a LangChain agent is created with `create_agent(model=llm, tools=[search_attachments])`. The agent uses the `messages` format (`SystemMessage` + `HumanMessage`) and decides when to call the `search_attachments` tool.

`search_attachments` uses the Tavily client to search for `relevant documents attachments for <topic> email`, returns up to 5 results, and formats them as a numbered list.

## Requirements

- Python 3.10+
- A [Mistral API key](https://console.mistral.ai/)
- A [Tavily API key](https://app.tavily.com/)

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

```bash
python app.py
```

Example session:

```
Topic : project deadline extension
Tone  (formal/casual/friendly): formal

⚙ Drafting → Fixing Grammar → Shortening...

─── Final Email ────────────────────────────
<generated email>
────────────────────────────────────────────

Commands:
  'add attachment suggestion' → search relevant attachments
  'exit' → quit

You: add attachment suggestion
AI: 📎 <numbered list of suggested attachments>

You: exit
Goodbye!
```

## Project structure

| File               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `app.py`           | Main application — email pipeline + follow-up agent. |
| `requirements.txt` | Python dependencies.                                 |
| `.env`             | API keys (not committed).                            |
