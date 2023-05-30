from pydantic import BaseModel
from fastapi import FastAPI
from datetime import datetime
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from io import BytesIO
import requests

target_size=(8,8)
app = FastAPI()

class helper(BaseModel):
    url_image_A : str
    url_image_B: str

def array_to_hash(hash_mat):

    return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

def preprocess_image(image,target_size):

    image_pil = image.resize(target_size, Image.LANCZOS)

    return np.array(image_pil).astype('uint8')

def hash_table(file):

    dct_coef = dct(dct(file, axis=0), axis=1)
    coefficient_extract=(8, 8)
    dct_reduced_coef = dct_coef[: coefficient_extract[0], : coefficient_extract[1]]

    median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

    hash_mat = dct_reduced_coef >= median_coef_val
    return hash_mat

def hamming_distance(hash1:str, hash2:str):

    """
    Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
    to be 64 for each hash and then calculates the hamming distance.
    Args:
        hash1: hash string
        hash2: hash string
    Returns:
        hamming_distance: Hamming distance between the two hashes.
    """

    hash1_bin = bin(int(hash1, 16))[2:].zfill(64)
    hash2_bin = bin(int(hash2, 16))[2:].zfill(64)

    distance = np.sum([i != j for i,j in zip(hash1_bin,hash2_bin)])

    return distance

def load_image(helper):

    response_A = requests.get(helper.url_image_A)
    response_B = requests.get(helper.url_image_B)
    if response_A.status_code != 200:
        return print('File A not Found')
    if response_B.status_code != 200:
        return print('File B not Found')

    try:

        img_A = Image.open(BytesIO(response_A.content))
        img_B = Image.open(BytesIO(response_B.content))

        if img_A.mode != 'RGB':
            img_A = img_A.convert('RGBA').convert('RGB')
            img_A = preprocess_image(img_A,target_size)
        else:
            img_A = preprocess_image(img_A,target_size)
            img_A = hash_table(img_A)

        if img_B.mode != 'RGB':
            img_B = img_B.convert('RGBA').convert('RGB')
            img_B = preprocess_image(img_B,target_size)

        else:
            img_B = preprocess_image(img_B,target_size)
            img_B = hash_table(img_B)

        distante_value = hamming_distance(array_to_hash(img_A),array_to_hash(img_B))

        if distante_value<=10:
            return 'Image Duplicated'
        else:
            return 'Not Duplicated'

    except:
        return "Something wrong with images, please check the paths"

@app.get("/check_duplciate_image")
async def read_image(helper: helper):
    print(helper.url_image_A)
    resp = load_image(helper)
    return resp
