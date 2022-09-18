from torch.utils.data import Dataset
from oscar.utils.tsv_file import TSVFile
import base64
import os.path as op
import torch
import json
import os
from tqdm import tqdm
import numpy as np
class  RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, split='train', is_train=True):
        super(RetrievalDataset, self).__init__()
        self.data_dir = args.data_dir
        self.img_file = args.img_feat_file #the vinvl feature from image
        caption_file = op.join(self.data_dir, '{}_captions.pt'.format(split))
        self.img_tsv = TSVFile(self.img_file)
        self.captions = torch.load(caption_file)
        self.img_keys = list(self.captions.keys())  # img_id as int
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}
        with open(args.image_category,'r',encoding='utf-8') as json_file:
            hashing_label=json.load(json_file)

        # get the image image_id to index map
        imgid2idx_file = op.join(op.dirname(self.img_file), 'imageid2idx.json')
        self.image_id2idx = json.load(open(imgid2idx_file))  # img_id as string 
        if args.add_od_labels:
            label_data_dir = op.dirname(self.img_file)
            label_file = os.path.join(label_data_dir, "predictions.tsv")
            self.label_tsv = TSVFile(label_file)
            self.labels = {}
            for line_no in tqdm(range(self.label_tsv.num_rows())):
                row = self.label_tsv.seek(line_no)
                image_id = row[0]
                if int(image_id) in self.img_keys:
                    results = json.loads(row[1])
                    objects = results['objects'] if type(
                        results) == dict else results
                    self.labels[int(image_id)] = {
                        "image_h": results["image_h"] if type(
                            results) == dict else 600,
                        "image_w": results["image_w"] if type(
                            results) == dict else 800,
                        "class": [cur_d['class'] for cur_d in objects],
                        "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                          dtype=np.float32)
                    }
            self.label_tsv._fp.close()
            self.label_tsv._fp = None
        if(is_train):
            self.num_captions_per_img = args.num_captions_per_img_train
        else:
            self.num_captions_per_img = args.num_captions_per_img_val
            if args.eval_img_keys_file:
                # select a subset of image keys for evaluation. eg. COCO 1k and 5k
                # eval_img_keys_file is a list of image keys saved in tsv file
                with open(op.join(args.data_dir, args.eval_img_keys_file), 'r') as f:
                    img_keys = f.readlines()
                self.img_keys = [int(k.strip()) for k in img_keys]
                self.captions = {k: self.captions[k] for k in self.img_keys}
                if args.add_od_labels:
                    self.labels = {k: self.labels[k] for k in self.img_keys}
        self.hashing_label ={}
        not_in = []
        for ind,k in enumerate(self.img_keys):
            if(str(k) in hashing_label):
                self.hashing_label[k]=hashing_label[str(k)]
            else:
                not_in.append(k)
        for k in not_in:
            self.img_keys.remove(k) 
        
        
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_img_seq_length = args.max_img_seq_length
        self.add_od_labels = args.add_od_labels#att_mask_type
        self.att_mask_type = args.att_mask_type
    def get_image_caption_index(self, index):
        if(self.is_train):
            img_idx = index 
            cap_idx = np.random.choice(self.num_captions_per_img)
            return img_idx, [self.img_keys[img_idx], cap_idx]            
        else:
            img_idx = index // self.num_captions_per_img
            cap_idx = index % self.num_captions_per_img
            return img_idx, [self.img_keys[img_idx], cap_idx]
    def get_od_labels(self, img_key):
        if self.add_od_labels:
            if type(self.labels[img_key]) == str:
                od_labels = self.labels[img_key]
            else:
                od_labels = ' '.join(self.labels[img_key]['class'])
            return od_labels
    def tensorize_example(self, text_a, img_feat, text_b=None, 
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_length - 2:
            tokens_a = tokens_a[:(self.max_seq_length - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
    
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_length   - 2:#a
                tokens_b = tokens_b[: (self.max_seq_length  - 2)]
            tokens_b = [self.tokenizer.cls_token] +tokens_b+ [self.tokenizer.sep_token]
            segment_ids_b = [sequence_b_segment_id] + [sequence_b_segment_id] * (len(tokens_b) -1)
        #a padding
        seq_len_a = len(tokens)
        seq_padding_len_a = self.max_seq_length - seq_len_a
        tokens += [self.tokenizer.pad_token] * seq_padding_len_a
        segment_ids += [pad_token_segment_id] * seq_padding_len_a
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #b padding
        seq_len_b = len(tokens_b)
        seq_padding_len_b = self.max_seq_length - seq_len_b
        tokens_b += [self.tokenizer.pad_token] * seq_padding_len_b
        segment_ids_b += [pad_token_segment_id] * seq_padding_len_b
        input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
        #merge
        input_ids = input_ids+input_ids_b
        segment_ids = segment_ids+segment_ids_b
        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_length:
            img_feat = img_feat[0 : self.max_img_seq_length, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_length - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.att_mask_type
        test_ = self.tokenizer.decode(input_ids)
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len_a + [0] * seq_padding_len_a +[1] * seq_len_b + [0] * seq_padding_len_b +  [1] * img_len + [0] * img_padding_len 
                             
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        
        return (input_ids, attention_mask, segment_ids, img_feat)


    def __getitem__(self, index):

        img_idx, cap_idxs = self.get_image_caption_index(index)
        img_key = self.img_keys[img_idx]
        feature = self.get_image(img_key)
        caption = self.captions[cap_idxs[0]][cap_idxs[1]]
        od_labels = self.get_od_labels(img_key)
        example = self.tensorize_example(caption, feature, text_b=od_labels)
        hashing_label = self.hashing_label[img_key]
        hashing_label = torch.tensor(hashing_label, dtype=torch.long)
        example = example +(hashing_label,)

        return index, example
    def get_test_labels(self):
        all_image = len(self.img_keys)
        all_sample = len(self.img_keys)*self.num_captions_per_img
        #all_label = self.num_captions_per_img
        label = []
        #where index is 1
        for i in tqdm(range(all_image)):
            start = i*self.num_captions_per_img
            label_i = [0 for i in range(all_sample)]
            for j in range(start,start+self.num_captions_per_img):
                label_i[j]=1
            label.extend(label_i)
        return label
    
    def get_image(self, image_id):
        image_idx = self.image_id2idx[str(image_id)]
        row = self.img_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1)).copy()
        t_features = torch.from_numpy(features)
        return t_features

    def __len__(self):
        if(self.is_train):
            return len(self.img_keys) 
        else:
            return len(self.img_keys) * self.num_captions_per_img



