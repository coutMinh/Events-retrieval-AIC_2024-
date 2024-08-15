from PIL import Image
import faiss
import numpy as np 
import clip
import torch
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import googletrans
import translate
from langdetect import detect

class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate.
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase
        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation
        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text
class MyFaiss:
  def __init__(self, root_database: str, bin_file: str, json_path: str):
    self.index = self.load_bin_file(bin_file)
    self.id2img_fps = self.load_json_file(json_path)
    self.__device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, preprocess = clip.load("ViT-B/32", device=self.__device)
    self.translater = Translation()
  def load_json_file(self, json_path: str):
      with open(json_path, 'r') as f:
        js = json.loads(f.read())

      return {int(k):v for k,v in js.items()}

  def takeNeighbor(self, id_image, k = 3):
    idx_image = [i for i in range(id_image - k, id_image + k + 1)]
    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]
    return image_paths, idx_image
  def load_bin_file(self, bin_file: str):
    return faiss.read_index(bin_file)


  def image_search(self, id_query, k):
    query_feats = self.index.reconstruct(id_query).reshape(1,-1)

    scores, idx_image = self.index.search(query_feats, k=k)
    idx_image = idx_image.flatten()

    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]

    # print(f"scores: {scores}")
    # print(f"idx: {idx_image}")
    # print(f"paths: {image_paths}")

    return scores, idx_image, infos_query, image_paths

  def text_search(self, text, k):
    if detect(text) == 'vi':
      text = self.translater(text)

    ###### TEXT FEATURES EXACTING ######
    text = clip.tokenize([text]).to(self.__device)
    text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)

    ###### SEARCHING #####
    scores, idx_image = self.index.search(text_features, k=k)
    idx_image = idx_image.flatten()

    ###### GET INFOS KEYFRAMES_ID ######
    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]
    # lst_shot = [info['list_shot_id'] for info in infos_query]

    # print(f"scores: {scores}")
    # print(f"idx: {idx_image}")
    # print(f"paths: {image_paths}")

    return scores, idx_image, infos_query, image_paths