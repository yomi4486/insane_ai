# FastAPIの読み込み
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from os.path import join, dirname
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

#token = os.environ.get("TOKEN")

async def create_text(input_text):
    inputs = tokenizer.encode_plus(input_text,
                                   return_tensors='pt',
                                   padding="max_length",
                                   truncation=True,
                                   max_length=50)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    # Generate the output
    outputs = model.generate(input_ids,
                             attention_mask=attention_mask,
                             max_length=100,
                             num_return_sequences=1,
                             temperature=0.1,
                             no_repeat_ngram_size=2)

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
async def get_image(token:str=None,prompt:str=None):

    json_load = json.load(open('./token/token.json','r'))
    if (not token in json_load) or (not token):
        print("Token Invalid")
        raise HTTPException(status_code=403, detail="Token invalid")
    if not prompt:
        print("Prompt Invalid")
        raise HTTPException(status_code=403, detail="Prompt Invalid")
    return_text = await create_text(prompt)

    return JSONResponse(content={"datail":return_text})
