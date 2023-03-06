"""this function generate recommendation for user query in text or image format

this combine both sparse and dense vector search methods
pinecone        -- vector db
dense vector    -- clip model
sparse vector   -- bm25

"""

import os
import pickle

import pinecone
import pinecone_text
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer

# load and set the enviroment variables
load_dotenv()

class SearchEngine:
    """
    Search engine --> input user / output relevant product from the product portfolio

    methodlogy :
        encode user vector into sparse vector (bm25) & dense vector using clip model
        query index the vector DB
        output top 10 results and render them with all the meta data
    """

    # init connection to pinecone
    pinecone.init(
        api_key= os.environ['PINECONE_KEY'] ,  # app.pinecone.io
        environment='us-east1-gcp'  # find next to api key
    )

    # set the pinecone idnex
    index_name = "hybrid-image-search"
    index = pinecone.GRPCIndex(index_name)
    # assignt the acceralator type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, bm25_path ) -> None:
        """class init"""
        # init the trained bm25 spearse vectorizer
        self.bm25_vectorizer = pickle.load( open(bm25_path , "rb") )
        # load a CLIP model from huggingface
        self.clip = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32',
            device=self.device
        )
        # initialize the meta data store
        self.prepare_metadata()


    def prepare_metadata(self):
        """ initialize the meta data store
            product images / product meta data
        """

        # load the dataset from huggingface datasets hub
        fashion = load_dataset(
            "ashraq/fashion-product-images-small",
            split="train"
        )

        # assign the images and metadata to separate variables
        self.images = fashion["image"]
        metadata = fashion.remove_columns("image")
        # convert metadata into a pandas dataframe
        self.metadata = metadata.to_pandas()

    def hybrid_scale(self, dense, sparse, alpha: float):
        """Hybrid vector scaling using a convex combination

        alpha * dense + (1 - alpha) * sparse

        Args:
            dense: Array of floats representing
            sparse: a dict of `indices` and `values`
            alpha: float between 0 and 1 where 0 == sparse only
                and 1 == dense only
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse
    
    def search(self,query, query_img=None, alpha=0.1, filter=None):
        """main search function"""

        # create the sparse vector
        sparse = self.bm25_vectorizer.transform_query(query)
        # check the user input state img / text query
        if(query_img is not None):
            # convert img to PIL format
            query_img = Image.fromarray(query_img)
            # now create the dense vector using the image
            dense = self.clip.encode(query_img).tolist()
            # put more weigth into image dense vector if user input a image ( try to push more product same as input image)
            alpha=0.02
        else:
            dense = self.clip.encode(query).tolist()

        # scale sparse and dense vectors
        hdense, hsparse = self.hybrid_scale(dense, sparse, alpha=alpha)
        # search
        result = self.index.query(
            top_k=10,
            vector=hdense,
            sparse_vector=hsparse,
            include_metadata=True,
            filter=filter  # add to metadata filter
        )
        # use returned product ids to get images
        imgs = [self.images[int(r["id"])] for r in result["matches"]]
        # return the product meta data
        meta_data = [ i['metadata']  for i in result["matches"] ]
        
        return imgs, meta_data
