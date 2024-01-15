# 首先导入所需第三方库
from langchain.document_loaders import CSVLoader #文档加载器，采用csv格式存储
from langchain.vectorstores import DocArrayInMemorySearch  #向量存储
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate


folder_path = '/root/code/InternLM/code_data/code_llm_data'  # 文件夹路径
docs = []

# 遍历文件夹下的所有文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):  # 只处理以 .csv 结尾的文件
        file_path = os.path.join(folder_path, file_name)
        
        # 使用 CSVLoader 加载单个文件
        loader = CSVLoader(file_path=file_path)
        docs.extend(loader.load())


# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")

# 构建向量数据库

# 加载数据库
vectordb = FAISS.from_documents(docs, embeddings)

vectordb.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)


print("数据库加载完成！")