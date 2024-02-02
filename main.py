import argparse
import torch
import logging
import shutil
import random
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from termcolor import colored
from transformers import AutoTokenizer
from transformers import ViTFeatureExtractor
from transformers import AutoProcessor
from transformers import get_scheduler
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from dataset import MyDataset
from model import MMNER
from trainer import Trainer

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--bert_model', type=str, default='./bert-base-multilingual-cased', required=False,
                        help='bert model')
    parser.add_argument('--text_encoder_layer', type=int, default=12, help='only use when use_image is True')
    parser.add_argument('--interaction_layer', type=int, default=6, help='layer number of interaction')
    parser.add_argument('--vit_model', type=str, default='openai/clip-vit-base-patch32',
                        help='openai/clip-vit-base-patch32 or google/vit-base-patch16-224')
    parser.add_argument('--data_dir', type=str, default='./train_valid_test/train_valid_test_all_lang',
                        choices=['./train_valid_test/train_valid_test_all_lang',
                                 './train_valid_test/train_valid_test_en',
                                 './train_valid_test/train_valid_test_fr',
                                 './train_valid_test/train_valid_test_es',
                                 './train_valid_test/train_valid_test_de',
                                 './train_valid_test/train_valid_test_en_plus_de',
                                 './train_valid_test/train_valid_test_en_plus_fr',
                                 './train_valid_test/train_valid_test_en_plus_es',
                                 './train_valid_test/train_valid_test_de_plus_fr',
                                 './train_valid_test/train_valid_test_de_plus_es',
                                 './train_valid_test/train_valid_test_fr_plus_es'], help='train/dev/test directory')
    parser.add_argument('--train_file', type=str, default='train_conll_format.txt', help='train file')
    parser.add_argument('--image_dir', type=str, default='./twitter_and_image_caption_images', help='image folder')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='CPU/GPU devices')
    parser.add_argument('--seed', type=int, default=19, help='random seed')

    parser.add_argument('--batch_size', type=int, default=32, help='default batch size is 32')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='the initial learning rate')
    parser.add_argument('--epochs', type=int, default=8, help='default epochs')
    parser.add_argument('--alpha', type=float, default=0.8, help='loss rate')

    parser.add_argument('--max_no_incre', type=int, default=3,
                        help='early stop when there is n epoch not increasing on dev')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train mode or test mode')
    parser.add_argument('--use_image', default=True, action='store_false', help='use --use_image to set False')
    parser.add_argument('--use_tqdm', type=bool, default=True, help='use tqdm or not')
    parser.add_argument('--use_crf', type=bool, default=True, help='use crf or not')
    parser.add_argument('--mmi', default=True, action='store_false',
                        help='use mmi(multimodal interaction) module or not')
    parser.add_argument('--print_report', default=True, action='store_false',
                        help='whether print report or not')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--lr_scheduler_name', type=str, default='linear', choices=['linear', 'polynomial'], help='学习率调整方式')
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='default num_warmup_steps')

    args = parser.parse_args()
    logger.info(colored('Args:', 'red'))
    for k, v in args.__dict__.items():
        logger.info(k + ': ' + str(v))
    logger.info('')
    return args

def main():
    parser = argparse.ArgumentParser(description='MMNER model implementation')
    args = parse_arguments(parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu')
    logger.info('use device {}'.format(device,))

    # tokenizer = AutoTokenizer.from_pretrained(args.bert_model, add_prefix_space=True, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True)
    # add special tokens (reference: https://github.com/huggingface/tokenizers/issues/247)
    special_tokens_dict = {'additional_special_tokens': ["�", '', '­', '', '', '​', '', '']}
    tokenizer.add_special_tokens(special_tokens_dict)

    vit_fe = AutoProcessor.from_pretrained(args.vit_model) if args.use_image else None

    # build dataset
    train_dataset = MyDataset(args, tokenizer, vit_fe, data_mode='train')
    label2idx = train_dataset.label2idx
    idx2label = train_dataset.idx2label
    dev_dataset = MyDataset(args, tokenizer, vit_fe, data_mode='dev', label2idx=label2idx)
    test_dataset = MyDataset(args, tokenizer, vit_fe, data_mode='test', label2idx=label2idx)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # build model
    model = MMNER(args, label2idx)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    # print(model)
    # for name, param in model.named_parameters():
    #     if name.startswith('vit.vision_model'):
    #         param.requires_grad = False

    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    lr_scheduler = get_scheduler(args.lr_scheduler_name, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                 num_training_steps=args.epochs * len(train_dataloader))

    # train
    trainer = Trainer(args)
    micro_f1_list, reports = [], []
    for epoch in range(args.epochs):
        total_loss = trainer.train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, device)
        # trainer.test_loop(dev_dataloader, model, device, idx2label, mode='dev')
        micro_f1, cr = trainer.test_loop(test_dataloader, model, device, idx2label, mode='test')
        logger.info('epoch: {} done, total loss: {}, micro_f1: {}'.format(epoch, total_loss, micro_f1))
        micro_f1_list.append((epoch, micro_f1))
        reports.append(cr)
    logger.info('micro f1 list: {}'.format(micro_f1_list))
    micro_f1_list = [micro_f1 for epoch, micro_f1 in micro_f1_list]
    best_micro_f1 = max(micro_f1_list)
    epoch = micro_f1_list.index(best_micro_f1)
    logger.info('best micro f1 in epoch {}: {}'.format(epoch, best_micro_f1))
    logger.info('report: {}'.format(reports[epoch]))
    # 如果模型不使用图片，则输出所有的评估报告
    if not args.use_image or args.print_report:
        logger.info('All reports:')
        for epoch, report in enumerate(reports):
            report.pop('weighted avg')
            logger.info('Epoch {}, report: {}'.format(epoch, report))


if __name__ == '__main__':
    main()
    logger.info('Done.')
