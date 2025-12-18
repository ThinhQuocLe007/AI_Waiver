from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(): 
    return HuggingFaceEmbeddings(
        model_name= 'bkai-foundation-models/vietnamese-bi-encoder'

    )