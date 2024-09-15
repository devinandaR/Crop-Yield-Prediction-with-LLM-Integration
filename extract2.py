from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.chains import RetrievalQA
from text_extracter import *
from prompts import *
import json
import os
import csv

os.environ["OPENAI_API_KEY"] = ""


def get_answer(pages,query,chain_type):
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type=chain_type)
    answer=chain({"input_documents": pages,"question": query},return_only_outputs=True)
    return answer

def get_full_answer(pages,query):
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer=chain.run(input_documents=pages, question=query)
    return answer

def get_answer_v1(text,prompt):
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer=chain({"text": text,"question": prompt},return_only_outputs=True)
    return answer


def no_tokens(text):
    llm = OpenAI(temperature=0)
    tks=llm.get_num_tokens(text)
    return tks

def get_tokens_by_page(page_dict):
    k={}
    for key in page_dict:
        tks=no_tokens(page_dict[key])
        k[key]=tks
    return k

def create_page_lists(dictionary, limit):
    current_length = 0
    page_lists = []
    current_list = []

    for page, tokens in dictionary.items():
        if current_length + tokens <= limit:
            current_length += tokens
            current_list.append(page)
        else:
            page_lists.append(current_list)
            current_list = [page]
            current_length = tokens

    page_lists.append(current_list)  # Append the last list

    result = {index + 1: pages for index, pages in enumerate(page_lists)}
    return result

def get_limited_text(page_dict,page_lists):
    t=[]
    for key in page_lists:
        list=page_lists[key]
        text=""
        for i in list:
            text+=page_dict[i]
        t.append(text)
    return t



def clean_text(text):
    # Remove newline characters and forward slashes
    cleaned_text = re.sub(r'[\n/]', '', text)

    # Replace sequences of two or more whitespace characters with a single space
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    return cleaned_text




iter=0
pdf_dir = "Task"
for filename in os.listdir(pdf_dir):
    if (iter==1):
        # continue
        break
    if filename.endswith(".pdf") and check_PDF!=None:
        page_dict=get_text_v2(filename)
        if len(page_dict)>0:
            token_dict=(get_tokens_by_page(page_dict))
            page_lists=create_page_lists(token_dict,3500)
            print(page_lists)
            pdf_path=(os.path.join(pdf_dir, filename))
            loader = PyPDFLoader(str(pdf_path))
            pages=loader.load_and_split()
            # Create an empty list to store the batches
            batches = []
            for key in page_lists:
                list=page_lists[key]
                batch = pages[int(list[0])-1:int(list[-1])-1]
                batches.append(batch)
            print(filename,len(batches))
            for batch in batches:
                try:
                    info=get_full_answer(batch,prompt3)
                    # print(type(info))
                    print(info)
                    data = json.loads(info)
                    print(data)
                        # print("----------------------------------------------------------------")
                except json.decoder.JSONDecodeError:
                    pass            
    iter+=1
