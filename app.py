import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load real GPT model (distilgpt2, matches book Chapter 4 architecture)
print("Loading model (first time may take 20-40 seconds, then instant)...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

def respond(message, history):
    # Prompt engineered with my research background
    prompt = f"""You are a MIT Mathematics Postdoc. Explain in clear, professional English.

User question: {message}

Answer concisely with mathematical insight:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=180,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean output (remove prompt)
    clean_reply = reply.split("Answer concisely with mathematical insight:")[-1].strip()
    return clean_reply if clean_reply else "Thinking..."

# Create professional web interface
demo = gr.ChatInterface(
    fn=respond,
    title="Math LLM Web Demo – Real GPT Model",
    description="Built from *Build a Large Language Model (From Scratch)* | Try it now!",
    examples=[
        "What is the attention mechanism?"
    ]
)

demo.launch(share=False)