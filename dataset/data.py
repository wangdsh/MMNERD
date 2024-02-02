import os
import logging
import torch
import numpy as np

from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from transformers import ViTFeatureExtractor
from transformers import AutoProcessor
from transformers.tokenization_utils_base import BatchEncoding

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class Instance:
    """自定义的数据结构"""
    def __init__(self, words, labels, img_path):
        self.words = words
        self.labels = labels
        self.img_path = img_path

        # check image path
        if os.path.exists(img_path):
            pass
        else:
            logger.error('{} not exists'.format(img_path))
            exit()

class MyDataset(Dataset):
    def __init__(self, args, tokenizer: AutoTokenizer, vit_fe: AutoProcessor = None, data_mode='train', label2idx: Dict[str, int] = None):
        self.args = args
        self.tokenizer = tokenizer
        self.vit_fe = vit_fe
        self.data_mode = data_mode

        # read data
        files = {'train': args.train_file, 'dev': 'valid_conll_format.txt', 'test': 'test_conll_format.txt'}
        file_path = os.path.join(args.data_dir, files[data_mode])
        self.insts = self.read_data(file_path)

        # build label index if mode is 'train'
        if data_mode == 'train':
            self.label2idx, self.idx2label = self.build_label_idx(self.insts)
        else:
            assert label2idx is not None # not build label2idx for dev/test dataset
            self.label2idx = label2idx

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, index):
        # return self.insts_tensors[index]
        return self.insts[index]

    def read_data(self, file_path):
        """read conll format file"""
        logger.info('read data from %s' % (file_path, ))
        insts = []
        with open(file_path, mode='r', encoding='utf-8') as fin:
            words, labels = [], []
            for line in fin:
                line = line.strip()
                if line.startswith('IMGID:'):
                    imgid = line.split('IMGID:')[1]
                elif line == '':
                    img_path = os.path.join(self.args.image_dir, imgid + '.jpg')
                    insts.append(Instance(words, labels, img_path))
                    words, labels = [], []
                else:
                    word, label = line.split()
                    words.append(word)
                    labels.append(label)
        logger.info('num of sentences: %d' % (len(insts), ))
        return insts

    def build_label_idx(self, insts: List[Instance]):
        """build mapping from label to index and index to label
        """
        label2idx = {}
        idx2label = []
        for inst in insts:
            for label in inst.labels:
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
                    idx2label.append(label)
        logger.info('# labels: {}'.format(label2idx))
        logger.info('# label size: {}'.format(len(label2idx)))
        return label2idx, idx2label

    def collate_fn(self, batch: List[Instance]):
        batch_words = [inst.words for inst in batch]
        # python main.py --batch_size=2
        # print(batch_words, len(batch_words[0]), len(batch_words[1]))
        encoding = self.tokenizer(batch_words,
                                  is_split_into_words=True,
                                  padding='longest',
                                  return_tensors='pt')
        # print(encoding)
        # print(len(encoding['attention_mask'][0]), len(encoding['attention_mask'][1]))

        # 追踪token与词语的映射
        word_to_token_index = []
        words_max_len = max([len(inst.words) for inst in batch])
        for batch_index in range(len(batch)):
            word_index = encoding.word_ids(batch_index)
            word_index = [tmp for tmp in word_index if tmp is not None]
            # print(word_index)
            pre_idx = -1
            word_poi = []
            for i, w_idx in enumerate(word_index):
                if w_idx > pre_idx:
                    word_poi.append(i) # 用位置i对应的token表示代表当前词的表示, 这里采用词对应的第一个token
                    pre_idx = w_idx
            # print('word_poi:', word_poi)
            assert len(word_poi) == len(batch[batch_index].words)
            word_poi = word_poi + [0] * (words_max_len - len(word_poi)) # 如果长度不够，需要补充
            word_to_token_index.append(word_poi)

        # label转为id, 并用-100进行padding
        words_max_len = max([len(inst.words) for inst in batch])
        batch_label = [[self.label2idx[label] for label in inst.labels] for inst in batch]
        batch_label = [labels + [-100] * (words_max_len - len(labels)) for labels in batch_label]

        # prepare for transformer attention
        bert_att_mask = [[1] * len(labels) + [0] * (words_max_len - len(labels)) for labels in batch_label]

        if self.args.use_image:
            # 处理图片 https://huggingface.co/docs/transformers/v4.24.0/en/model_doc/vit
            batch_img = []
            for inst in batch:
                try:
                    img = Image.open(inst.img_path)
                except:
                    img_not_found = os.path.join(self.args.image_dir, '17_06_4705.jpg') # 一个默认的图片
                    img = Image.open(img_not_found)
                finally:
                    batch_img.append(img)
            # some images are png format, mode is RGBA, remove the alpha channel, make it 3 (RGB)
            batch_img = [img if img.mode == 'RGB' else img.convert('RGB') for img in batch_img]

            img_fe = self.vit_fe(images=batch_img, return_tensors='pt')

            # for resnet
            crop_size = 224
            transform = transforms.Compose([
                 transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            batch_img = [img.resize((crop_size, crop_size), Image.BILINEAR) if (img.width<crop_size or img.height<crop_size) else img for img in batch_img]
            resnet_img_fe = [transform(img) for img in batch_img]

            for img in batch_img:
                img.close()

            return encoding, torch.tensor(batch_label), torch.tensor(word_to_token_index), \
                img_fe, torch.tensor(bert_att_mask), torch.stack(resnet_img_fe)
        else:
            return encoding, torch.tensor(batch_label), torch.tensor(word_to_token_index)
