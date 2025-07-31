import os
import torch
import tqdm
import pickle
import json
from transformers import AutoModel, AutoProcessor, CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
# import faiss
import numpy as np
from pathlib import Path
# from faiss import write_index, read_index
# import faiss.contrib.torch_utils
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredFileLoader
from langchain_unstructured import UnstructuredLoader
from langchain.schema import Document
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import logging
from langchain.schema import Document
from typing import List
# import faiss
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='my_log.log',       # 指定日志文件路径
    filemode='w'  )
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Knowledge base for system.

    Returns:
        KnowledgeBase
    """

    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)

    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        self.knowledge_base_path = knowledge_base_path
        # knowledge_base save splitted chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        self.knowledge_base = []

    def load_knowledge_base(self):
        """Load the knowledge base."""
        loader = UnstructuredLoader(self.knowledge_base_path)
        document = loader.load()
        #Split article
        all_splits = self.text_splitter.split_documents(document)
        logger.info(f"Finish split chunks, the len of chunks is {len(all_splits)}")
        self.knowledge_base = all_splits

class VectorStore(KnowledgeBase):
    def __init__(self,knowledge_base_path,index_dir:str = "faiss_index"):
        """ Loading kb and enconding them 
        Embedding savd to faiss database
     
        Args:
            knowledge_base_path (_type_): kb path to loading
            index_dir (_type_): faiss saving path for embeddings 
        """
        super().__init__(knowledge_base_path)
        self.load_knowledge_base()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.vector_store = None
        self.emb_model = OllamaEmbeddings(model="bge-m3:latest", base_url="http://localhost:11434")
        self.embedding_dim = 1024
    def getkb(self):
        return self.knowledge_base
    def encode(self,text=None):
        if text:
            return self.emb_model.embed_documents(text)
        return None
    def create_vector_store(self):
        """
            Encoding knowledge base
            Saving to vector store
        """
        if not self.knowledge_base:
            logger.warning("No kb to create vector storage")
        logger.info(f"Start build vector store ...")
        try:
            self.vector_store = FAISS.from_documents(
                documents = self.knowledge_base,
                embedding = self.emb_model,
                normalize_L2=True,
                # index=faiss.IndexFlatIP(self.embedding_dim),
            )
            # self.vector_store.index = faiss.IndexFlatIP(self.embedding_dim)
            self._save_vector_store(self.vector_store)
            logger.info(f"Vector storage was created successfully, containing {len(self.knowledge_base)} document blocks")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to create vector storage: {str(e)}")
            return None
    
    def _save_vector_store(self,vector_store:FAISS):
        """Saving Faiss Database 

        Args:
            vector_store (Faiss): vector/embeddings of kb
        """
        try:
            vector_store.save_local(str(self.index_dir))
            logger.info(f"Vector is saved in :{self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save vector")
    
    def load_vector_store(self):
        """loading vector store
        """
        try:
            if (self.index_dir / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.emb_model,
                    normalize_L2=True, 
                    allow_dangerous_deserialization=True
                )
                vector_count = len(self.vector_store.docstore._dict)
                logger.info(f"Vector storage loading success, total chunks: {vector_count}")
                return self.vector_store
            logger.warning("Vector storage file does not exist")
        except Exception as e:
            logger.error(f"Failed to load vector storage: {str(e)}")
        return None
    def search_documents(self,query:str,threshold: float=0,top_k: int=10):
        """retrieving some high similarity chunks

        Args:
            query (str): query text info , to search in faiss database
            threshold (float, optional): higher threshold higher similarity. Defaults to 0.7.
        """
        if not self.vector_store:
            self.vector_store = self.load_vector_store()
            if not self.vector_store:
                logger.warning("Vector storage is not initialized")
                return []
        try:
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
            )
            results = [doc for doc, score in docs_and_scores if score > threshold]
            logger.info(f"Searched {len(results)} related documents, similarity threshold: {threshold}")
            return results
        except Exception as e:
            logger.error(f"Failed to search for document: {str(e)}")
            return []
    def get_context(self,docs:List[Document]):
        """ retrieved chunks results is a list of document
            document has many key, just concat the page_content
        Args:
            docs (List[Document]): retrieved chunks
        """
        if not docs:
            return ""
        return "\n".join(doc.page_content for doc in docs)
    
    def add_document(self,content:str, metadata:dict):
        """content is the page content, a str
        matedata is the Additional information (optional), such as file name, 
        page number, time, etc., for searching or filtering

        Args:
            content (str): str of added document
            matedata (dict): the addtitional information
            return: Ture  -- add doc succeed
                    False -- Failed to add doc
        """
        if not content:
            logger.warning("The document content is empty and cannot be added")
            return False
        try:
            doc =Document(page_content = content,metadata = metadata or {})
            # if metadata id none or false , will use {} instand
            split_docs = self.text_splitter.split_documents([doc])
            # 如果向量存储不存在，先初始化
            if not self.vector_store:
                self.vector_store = self.load_vector_store()
                if not self.vector_store:
                    # 如果仍不存在，使用当前文档创建新的向量存储
                    self.vector_store = self.create_vector_store([doc])
                    return True
            self.vector_store.add_documents(split_docs)
            self._save_vector_store(self.vector_store)
            logger.info(f"成功添加文档，标题: {metadata.get('source', '未知') if metadata else '未知'}，分块数量: {len(split_docs)}")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    def clear_index(self):
        try:
            for file in self.index_dir.glob("*"):
                file.unlink()
            self.vector_store = None
            logger.info("索引已清除")
        except Exception as e:
            logger.error(f"清除索引失败: {str(e)}")
            raise

if __name__ == "__main__": 
    knowledge_base_path = "/24T/yyy/szx/kb.txt"
    knowledge_base = VectorStore(knowledge_base_path)
    # all_splits = knowledge_base.getkb() #split 后的kb

    # 输出结果
    # print(len(all_splits))
    # print(type(all_splits))
    # print("切分后的文本：")
    # for i, split in enumerate(all_splits):
    #     print(f"Split {i + 1}: {split}")
    #     break
    # emb = knowledge_base.encode(all_splits)
    # print(len(emb))
    # print(len(emb[0]))
    # knowledge_base.clear_index()
    # kbfaiss = knowledge_base.create_vector_store()
    loadfaiss = knowledge_base.load_vector_store()
    # retrieves = knowledge_base.search_documents("鲁菜起源于哪里？")
    # print(knowledge_base.get_context(retrieves))
    knowledge_base.add_document("猪猪猪大侠大战世界！！！，猪猪侠的大名是GGbond",None)
    # knowledge_base.clear_index()
    # knowledge_base.add_document("猪猪猪大侠大战世界！！！，猪猪侠的大名是GGbond",None)
    # loadfaiss = knowledge_base.load_vector_store()
    retrieves = knowledge_base.search_documents("猪猪侠叫什么？")
    print(knowledge_base.get_context(retrieves))

    
    # print(retrieves.page_content)