from copy import deepcopy
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import MarkdownElementNodeParser

class DocumentProcessor:
    def __init__(self, llama_parser_api_key=None):
        self.llama_parser_api_key = llama_parser_api_key

    def transform(self, uploaded_file, llm, embed_model):
        """
        Parse the uploaded file and converts them into nodes
        """
        markdown_doc = self._parse(uploaded_file) # type: ignore
        node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
        nodes = node_parser.get_nodes_from_documents(markdown_doc)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
        page_nodes = self._get_page_nodes(markdown_doc) # type: ignore
        return base_nodes, objects, page_nodes

    def _parse(self, file_path):
        """
        Parse the uploaded file into markdown
        """
        parser = LlamaParse(result_type="markdown", api_key=self.llama_parser_api_key)
        document = parser.load_data(file_path)
        # directly parse the uploaded file object
        # with open(f'./{uploaded_file.name}', "rb") as f:
        #     document = parser.load_data(f)
        return document

    def _get_page_nodes(self, documents, separator="\n---\n"):
        """
        Split each document into page node, by separator
        """
        nodes = []
        for doc in documents:
            page = doc.text.split(separator)
            for text in page:
                node = TextNode(
                    text=text,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)
        return nodes
    
