"""main ecommerce hybrid search engine"""

import os
import random
from base64 import b64encode
from io import BytesIO

import gradio as gr
from datasets import load_dataset
from IPython.core.display import HTML
from transformers import BertTokenizerFast

from search import SearchEngine

# load bert tokenizer from huggingface
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-uncased'
)
# define the text tokenization func
def tokenize_func(text):
    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )['input_ids']
    return tokenizer.convert_ids_to_tokens(token_ids)

# init the search engine
ecommerce_engine=SearchEngine(bm25_path="bm25.pickle")


# # load the dataset from huggingface datasets hub
# fashion = load_dataset(
#     "ashraq/fashion-product-images-small",
#     split="train"
# )

# images = fashion["image"]
# metadata = fashion.remove_columns("image")
# # convert metadata into a pandas dataframe
# metadata = metadata.to_pandas()

# test_meta = metadata.iloc[:4]['productDisplayName'].values.tolist()
# test_category = metadata.iloc[:4]['subCategory'].values.tolist()
# test_imgs = images[:4]

# html css template for item cards
html_css = '''
    <html>
    <head>
        <style>
        .card {
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
            max-width: 300px;
            margin: auto;
            text-align: center;
            font-family: arial;
            }

        .price {
            color: grey;
            font-size: 22px;
            }

        .card button {
            border: none;
            outline: 0;
            padding: 12px;
            color: white;
            background-color: #000;
            text-align: center;
            cursor: pointer;
            width: 100%;
            font-size: 18px;
            }

        .card button:hover {
            opacity: 0.3;
            }
        </style>
    </head>
    <body>
'''

# function to display product images/ information into webpage
def display_card(user_query, query_img):

    #apply filters into search
    filter_feature = None # {"baseColour": "Blue"} 
    # search based on the user input text query / image and filters
    rec_imgs, rec_meta = ecommerce_engine.search(
                            query=user_query,
                            query_img=query_img,
                            filter=filter_feature,
                            alpha=0.1
                        )

    figures = []
    figures.append( html_css )
    for img, meta in zip(rec_imgs, rec_meta):
        b = BytesIO()  
        img.save(b, format='png')
        figures.append(f'''
        <div class="card">
            <img src="data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}" style="width:100%">
            <h1>{meta['subCategory']}</h1>
            <p class="price">$19.99</p>
            <p>{meta['productDisplayName']}.</p>
            <p><button>Add to Cart</button></p>    
        </div>
        ''')
    return f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
        </body>
        </html>
    '''

# -------------------- init gradio application ------------------------------

user_query = gr.Textbox(
        placeholder="Type What's in your'e mind ...",
        show_label=False,
        lines=1,
)

user_img =  gr.Image(label="Input Image of Item and let us help you ....",
                    show_label=True
)

#submit_btn = gr.Button(value="Search")

demo = gr.Interface(
        
        display_card,
        title="Shope ME",
        css=".gradio-container {background-color: lightgray}",
        description="Try New AI Integrated Search Engine .... All at your'e finger tips .........",
        inputs=[user_query, 
                user_img],
        outputs=["html"]

    )

demo.launch(debug=True)
