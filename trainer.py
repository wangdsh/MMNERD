import logging
import torch

from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args):
        self.args = args

    def train_loop(self, dataloader, model, loss_fn, optimizer, lr_scheduler, device):
        model.train()
        total_loss = 0.
        if self.args.use_tqdm:
            dataloader = tqdm(dataloader, total=len(dataloader))

        for batch in dataloader:
            if self.args.use_image:
                batch_inputs, batch_label, word_to_token_index, img_fe, bert_att_mask, resnet_img_fe = batch
                batch_inputs, batch_label = batch_inputs.to(device), batch_label.to(device)
                word_to_token_index, img_fe = word_to_token_index.to(device), img_fe.to(device)
                bert_att_mask = bert_att_mask.to(device)
                resnet_img_fe = resnet_img_fe.to(device)

                pred, loss = model(batch_inputs, word_to_token_index, batch_label,
                                   img_fe, bert_att_mask, resnet_img_fe) # pred (bs, max_seq_len, label_size)
            else:
                batch_inputs, batch_label, word_to_token_index = batch
                batch_inputs, batch_label = batch_inputs.to(device), batch_label.to(device)
                word_to_token_index = word_to_token_index.to(device)

                pred, loss = model(batch_inputs, word_to_token_index, batch_label)

            if self.args.use_crf:
                pass
            else:
                loss = loss_fn(pred.permute(0, 2, 1), batch_label)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        return total_loss

    def test_loop(self, dataloader, model, device, idx2label, mode='test'):
        model.eval()
        true_labels, true_predictions = [], []
        with torch.no_grad():
            if self.args.use_tqdm:
                dataloader = tqdm(dataloader, total=len(dataloader))
            for batch in dataloader:
                if self.args.use_image:
                    batch_inputs, batch_label, word_to_token_index, img_fe, bert_att_mask, resnet_img_fe = batch
                    batch_inputs, batch_label = batch_inputs.to(device), batch_label.to(device)
                    word_to_token_index, img_fe = word_to_token_index.to(device), img_fe.to(device)
                    bert_att_mask = bert_att_mask.to(device)
                    resnet_img_fe = resnet_img_fe.to(device)

                    pred = model(batch_inputs, word_to_token_index, batch_label,
                                 img_fe, bert_att_mask, resnet_img_fe, mode='eval')
                else:
                    batch_inputs, batch_label, word_to_token_index = batch
                    batch_inputs, batch_label = batch_inputs.to(device), batch_label.to(device)
                    word_to_token_index = word_to_token_index.to(device)

                    pred = model(batch_inputs, word_to_token_index, batch_label, mode='eval')
                labels = batch_label.cpu().numpy().tolist()

                # label id 转为 label
                true_labels += [[idx2label[int(l)] for l in label if l != -100] for label in labels]
                true_predictions += [
                    [idx2label[int(p)] for p, l in zip(prediction, label) if l != -100] 
                    for prediction, label in zip(pred, labels)
                ]
        # print(classification_report(true_labels, true_predictions, mode=None, scheme=IOB2))
        cr = classification_report(true_labels, true_predictions, mode=None, scheme=IOB2, output_dict=True)
        # logger.info(mode)
        micro_f1 = cr['micro avg']['f1-score']
        return micro_f1, cr
