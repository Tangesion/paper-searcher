from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from peft import PeftModel
import os
import urllib.request
import urllib.error
import ssl
import requests
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm

def askURL(url):
    head = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        # 创建一个不验证 SSL 证书的上下文
        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, context=context)
        html = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
    return html

def getTitleList(url):
    html = askURL(url)
    print("html get success!")
    soup = BeautifulSoup(html, "html.parser")
    print("parse html success!")
    title_list = []
    pattern = re.compile(r'<span class="title" itemprop="name">(.*?)</span>')
    for item in soup.find_all('span', class_='title', itemprop='name'):
        match = pattern.search(str(item))
        if match:
            title_list.append(match.group(1))
    return title_list

def search_and_download(title):
    os.mkdir("./papers") if not os.path.exists("./papers") else None
    os.mkdir("./papers/not_found") if not os.path.exists("./papers/not_found") else None

    search_url = "https://arxiv.org/search/?query=" + title.replace(" ", "+") + "&searchtype=title"
    response = requests.get(search_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 检查是否存在 "Sorry, your query for" 元素
        no_results = soup.find('p', class_='is-size-4 has-text-warning')
        if no_results:
            tqdm.write(f"{title} No results in arXiv found for the given title, write in json!")
            #print("No results in arXiv found for the given title.")
            if os.path.exists("./papers/not_found/not_found_articles.json"):
                with open("./papers/not_found/not_found_articles.json", "r") as f:
                    not_found_articles = json.load(f)
            else:
                not_found_articles = []

            not_found_articles.append(title)

            with open("./papers/not_found/not_found_articles.json", "w") as f:
                json.dump(not_found_articles, f, indent=4)
            return
        
        # 查找 PDF 下载链接
        result = soup.find('a', href=True, text='pdf')
        if result:
            pdf_url = result['href']
            pdf_response = requests.get(pdf_url)
            if pdf_response.status_code == 200:
                with open(f"./papers/{title}.pdf", "wb") as f:
                    f.write(pdf_response.content)
                tqdm.write(f"{title} Paper downloaded successfully.")
                #print(f"{title} Paper downloaded successfully.")
            else:
                tqdm.write("Failed to download PDF.")
                #print("Failed to download PDF.")
        else:
            tqdm.write("No PDF link found.")
            #print("No PDF link found.")
    else:
        tqdm.write("Failed to search arXiv.")
        #print("Failed to search arXiv.")


def load_fintune_model(model_path, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path,torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    return model, tokenizer, device


if __name__ == "__main__":
    config = json.load(open("config.json"))
    model_path = config["model_path"]
    lora_path = config["lora_path"]
    web_url = config["paper_web_url"]
    model, tokenizer, device = load_fintune_model(model_path, lora_path)
    title_list = getTitleList(web_url)
    for title in tqdm(title_list):
        input_ids = tokenizer.encode(title, return_tensors="pt").to(device)
        logits = model(input_ids).logits
        pred = torch.argmax(logits, dim=1).item()
        if pred:
            search_and_download(title)





