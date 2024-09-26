from llama_index.core import VectorStoreIndex

def vector_store(nodes):
    recursive_index = VectorStoreIndex(nodes=nodes)
    return recursive_index