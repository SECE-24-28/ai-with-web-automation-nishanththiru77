# Hugging Face LLM Chatbot

A simple command-line chatbot built with Python and Hugging Face Transformers. The chatbot uses a pre-trained Large Language Model (LLM) to generate conversational responses and maintain chat history.

## Features

- Interactive command-line chat interface
- Uses Hugging Face pre-trained language models
- Maintains conversation context
- Easy to customize with different models
- Runs locally on your machine

## Requirements

- Python 3.8 or higher
- pip package manager

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/huggingface-chatbot.git
cd huggingface-chatbot
```

### 2. Create a Virtual Environment (Optional but Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install transformers torch
```

## Project Structure

```text
huggingface-chatbot/
│
├── chatbot.py
├── requirements.txt
└── README.md
```

## chatbot.py

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-medium"

print("Loading model... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None

print("\nChatbot Ready!")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt"
    )

    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_input_ids
    )

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )

    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    print("Bot:", response)
```

## requirements.txt

```txt
transformers
torch
```

## Running the Chatbot

Execute the following command:

```bash
python chatbot.py
```

Example:

```text
You: Hello
Bot: Hi! How can I help you today?

You: What is Python?
Bot: Python is a popular programming language used for web development, AI, automation, and more.
```

## Changing the Model

You can replace the model name in `chatbot.py`:

```python
model_name = "microsoft/DialoGPT-medium"
```

Examples:

```python
model_name = "microsoft/DialoGPT-small"
```

```python
model_name = "microsoft/DialoGPT-large"
```

```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## Performance Notes

- Smaller models run faster on CPUs.
- Larger models may require a GPU.
- First run downloads the model automatically.
- Download size depends on the selected model.

## Troubleshooting

### Torch Installation Issues

Visit the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

### Out of Memory Errors

Try a smaller model:

```python
model_name = "microsoft/DialoGPT-small"
```

### Slow Responses

- Use a GPU if available.
- Reduce `max_length`.
- Use a smaller model.

## Future Improvements

- Web interface using Streamlit
- GUI application using Tkinter
- Chat history storage
- Voice input and output
- Integration with Hugging Face Inference API
- Support for modern instruction-tuned LLMs

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files to deal in the Software
without restriction.

## Acknowledgments

- Hugging Face Transformers
- PyTorch
- Open-source AI community
