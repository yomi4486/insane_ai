# FastAPIの読み込み
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

async def create_text(input_text):
    inputs = tokenizer.encode_plus(input_text,
                                   return_tensors='pt',
                                   padding="max_length",
                                   truncation=True,
                                   max_length=50)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    # Generate the output
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=128,
        num_return_sequences=random.randint(1,5),
        temperature=random.random(),
        do_sample=True,
        no_repeat_ngram_size=random.randint(1,5)
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(input_text):]

# FastAPIのインスタンスを作成
app = FastAPI(title="あたおかAI",version="1.0.0",description="きちがいです。https://discord.com/oauth2/authorize?client_id=1181054052042297354&permissions=0&scope=bot%20applications.commands")

ip_last_access = {}

# Middleware to handle CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_image(prompt:str=None):
    if not prompt:
        print("Prompt Invalid")
        raise HTTPException(status_code=403, detail="Prompt Invalid")

    return_text = await create_text(prompt)
    
    return JSONResponse(content={"datail":return_text})
