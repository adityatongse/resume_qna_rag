# Quick Start Guide

Get up and running with the CV Question Answering Agent in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- Your CV in PDF format
- OpenAI API key (or Anthropic/OpenRouter)

## Step-by-Step Setup

### 1. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Generate .env file from template

### 2. Configure API Key

Edit the `.env` file:

```bash
nano .env
```

Add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Save and exit (Ctrl+X, then Y, then Enter).

### 3. Add Your CV

```bash
cp /path/to/your/resume.pdf data/cv/resume.pdf
```

### 4. Run the Agent

```bash
python main.py
```

## First Questions to Try

Once the agent starts, try these questions:

1. **"What is my work experience?"**
2. **"Tell me about my education"**
3. **"What programming languages do I know?"**
4. **"What was my most recent role?"** (follow-up question)

## Commands

While using the agent:

- `history` - View conversation history
- `stats` - View session statistics
- `clear` - Clear conversation history
- `help` - Show help message
- `exit` or `quit` - Exit and save conversation

## Troubleshooting

### "CV file not found"
```bash
# Make sure your CV is in the right place
ls data/cv/
# Should show: resume.pdf
```

### "API key not set"
```bash
# Check your .env file
cat .env | grep OPENAI_API_KEY
# Should show: OPENAI_API_KEY=sk-...
```

### Import errors
```bash
# Activate virtual environment
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

## Switching LLM Providers

### Use Anthropic Claude

In `.env`:
```env
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-opus-20240229
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Use OpenRouter

In `.env`:
```env
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=openai/gpt-4-turbo-preview
OPENROUTER_API_KEY=sk-or-your-key-here
```

## What Happens on First Run?

1. **CV Ingestion**: Your CV is parsed and split into chunks
2. **Embedding Generation**: Text chunks are converted to embeddings
3. **Vector Storage**: Embeddings are stored in ChromaDB
4. **Agent Ready**: You can start asking questions!

The ingestion only happens once. On subsequent runs, the agent uses the stored embeddings.

## Output Files

After each session, your conversation is saved to:
```
outputs/conversations/conversation_YYYYMMDD_HHMMSS.txt
```

## Need More Help?

See the full [README.md](README.md) for:
- Detailed architecture
- Advanced configuration
- Troubleshooting guide
- API reference

## Demo Video

For a visual walkthrough, see the demo video.

---

**Ready to start? Run `python main.py` and ask your first question!** 🚀