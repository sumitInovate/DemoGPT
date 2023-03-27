from flask import Flask,request, jsonify
import json
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get("api_key")

app = Flask(__name__)

os.environ['OPENAI_API_KEY'] = API_KEY

# Import JSON File
f = open(os.path.join(os.getcwd(), 'data/prompt.json'))
data = json.load(f)

# Train JSON File
def construct_index(directory_path):
    #set maximum input size
    max_input_size = 4096
    #set number of output tokens
    num_outputs = 500
    #set maximum chunks overlap
    max_chunk_overlap = 20
    #set chunk size limit
    chunk_size_limit = 600


    #define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.3, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index
        
        



@app.route('/')
def index():
    return "Welcome to Ingram Micro Chatbot API"

# GET 
@app.route('/prompts', methods=['GET'])
def get():
    return jsonify({'Data': data})

# POST
@app.route('/prompts', methods=['POST'])
def create():
    prompt = json.loads(request.data)
    construct_index('data')
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(prompt["prompt"], response_mode='compact')
    data.append({"prompt":prompt, "completion": response})
    return jsonify({"Response": response.response})

if __name__ == "__main__":
    app.run(debug=True, host="https://chatcustomgpt-production.up.railway.app/", port=5000)
