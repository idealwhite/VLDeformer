# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license. 

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import  json
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from oscar.utils.tsv_file import TSVFile
from oscar.utils.logger import setup_logger
from oscar.utils.misc import mkdir, set_seed
from oscar.modeling.modeling_bert import ContrastiveTwins
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule, WarmupCosineSchedule
from oscar.datasets.oscar_dataset import RetrievalDataset,sbu_total_dataset
from torch.utils.data import ConcatDataset
from torch.cuda.amp import GradScaler,autocast

def compute_score_with_logits(logits, labels):
    logits,labels = logits.detach().cpu().numpy(),labels.detach().cpu().numpy()
    i2t_ranks = []
    for lab, sim in zip(labels, logits):
        inds = np.argsort(sim)[::-1]
        rank = logits.shape[1]
        for r, ind in enumerate(inds):
            if ind == lab:
                rank = r
                break
        i2t_ranks.append(rank)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    return i2t_accs

def compute_ranks(dataset, results):
    labels = np.array(dataset.get_test_labels())
    similarities = results

    num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)

    labels = np.swapaxes(labels, 0, 1)
    similarities = np.swapaxes(similarities, 0, 1)
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks

def save_pretrained(model, save_directory,name="model.cpkt"):
    """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
    """
    assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

    # Only save the model it-self if we are using distributed training
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save configuration file
    model_to_save.config.save_pretrained(save_directory)

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, name)

    torch.save(model_to_save.state_dict(), output_model_file)
def save_checkpoint(model, tokenizer, args, epoch, global_step):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, global_step))
    mkdir(checkpoint_dir)
    #model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            
            save_pretrained(model=model,save_directory=checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return

def info_nce_loss(args, feature_t, feature_v, norm=True):
    if norm:
        feature_t = F.normalize(feature_t, dim=1)
        feature_v = F.normalize(feature_v, dim=1)
    batch_size = feature_t.shape[0]
    labels =torch.Tensor(np.arange(batch_size)).to(device=args.contrastive_gpu).long()
    
    matrix = torch.matmul(feature_t, feature_v.T).to(device=args.contrastive_gpu)

    logits_i = matrix / args.temperature_i
    logits_t = matrix / args.temperature_t

    loss_fct = CrossEntropyLoss()
    loss_t2i = loss_fct(logits_i, labels.view(-1))
    loss_i2t = loss_fct(logits_t.T, labels.view(-1))

    return loss_i2t + loss_t2i

def triplet_loss(args, feature_t, feature_i, norm=True):
    if norm:
        feature_t = F.normalize(feature_t, dim=1)
        feature_i = F.normalize(feature_i, dim=1)

    batch_size = feature_t.shape[0]

    # create negatives for the random batch
    negative_idx = torch.range(-1, batch_size-2).long()
    feature_t_neg = feature_t[negative_idx]
    feature_i_neg = feature_i[negative_idx]

    # (no grad before here)
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    loss_i = triplet_loss(feature_i, feature_t, feature_t_neg)
    loss_t = triplet_loss(feature_t, feature_i, feature_i_neg)

    return loss_t + loss_i

def train(args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // \
                args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                * args.num_train_epochs

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    elif args.scheduler == "cos":
        scheduler = WarmupCosineSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model,device_ids=args.gpus)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step, global_loss, global_acc =0,  0.0, 0.0
    model.zero_grad()
    log_json = []
    best_score = 0
    scaler = GradScaler()
    for epoch in range(int(args.num_train_epochs)):
        for step, (_, batch) in enumerate(train_dataloader):
            if( batch[0].shape[0]!=args.train_batch_size):
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            #inputs = cross_batch(batch,args)
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3]
            }
            #inputs["attention_mask"] = inputs["attention_mask"][:,70:]
            #inputs["attention_mask"] = inputs["attention_mask"][:,:70]
            with autocast():
                results_i,results_t = model(**inputs)
                if args.triplet_loss:
                    loss = triplet_loss(args, results_t, results_i)
                else:
                    loss = info_nce_loss(args, results_t, results_i)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            #loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            #batch_acc = batch_score.item() / (args.train_batch_size )
            global_loss += loss.item()
            #global_acc += batch_acc
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                if global_step % args.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f}, " .format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss)
                    )

                if(args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == t_total:
                    save_checkpoint(model, tokenizer, args, epoch, global_step) 
                    # evaluation
                    if args.evaluate_during_training: 
                        logger.info("Perform evaluation at step: %d" % (global_step))
                        test_result = test(args, model, val_dataset)
                        eval_result = evaluate(val_dataset, test_result)
                        rank_accs = eval_result['i2t_retrieval']
                        if rank_accs['R@1'] > best_score:
                            best_score = rank_accs['R@1']
                        epoch_log = {'epoch': epoch, 'global_step': global_step, 
                                     'R1': rank_accs['R@1'], 'R5': rank_accs['R@5'], 
                                     'R10': rank_accs['R@10'], 'best_R1':best_score}
                        log_json.append(epoch_log)
                        with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                            json.dump(log_json, fp) 
    return global_step, global_loss / global_step

import time
def test(args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=args.eval_batch_size, num_workers=args.num_workers)
    
    logger.info("Num examples = {}".format(len(eval_dataset)))
    logger.info("Evaluation batch size = {}".format(args.eval_batch_size))
    model.eval()
    results_t = [] 
    results_i = [] 
    total_time = 0.

    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        start = time.time()
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3]
            }
            logits = model(**inputs)
            results_i.append(logits[0])
            results_t.append(logits[1])
        end = time.time()
        batch_time = end - start
        total_time += batch_time
  
    start = time.time()
    #Note: annote the device transfer and index operations while test time.
    results_i = torch.cat(results_i,0).to(args.cuda_for_test) 
    results_t = torch.cat(results_t,0).to(args.cuda_for_test) 
    results_i = results_i[::args.num_captions_per_img_val]

    if args.cache_representation:
        logger.info("Cache %d samples" % len(results_i))
        torch.save(results_i.detach().cpu(), 'features_i.bin')
        torch.save(results_t.detach().cpu(), 'features_t.bin')

    results_i = F.normalize(results_i, dim=1)
    results_t = F.normalize(results_t, dim=1)
    

    sim = torch.matmul(results_i, results_t.T).cpu().numpy()

    end = time.time()
    batch_time = end - start
    print('*'*10, 'cosine time: %fs '%batch_time ,'*'*10)
    total_time += batch_time
    print('*'*10, 'time: %fs '%total_time ,'*'*10)
    return sim


def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result


def get_predict_file(args):
    cc = []
    data = op.basename(op.join(args.data_dir, '')[:-1])
    if data != 'coco_ir':
        cc.append(data)
    cc.append(args.test_split)
    if args.add_od_labels:
        cc.append('wlabels{}'.format(args.od_label_type))
    return op.join(args.eval_model_dir, '{}.results.pt'.format('.'.join(cc))) 


def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length', 
            'max_img_seq_length', 'add_od_labels', 'od_label_type',
            'use_img_layernorm', 'img_layer_norm_eps']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logger.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='/raid/data_modal/coco_vinvL/coco_ir/', type=str, required=False,
                        help="The input data dir with all required files.")
    parser.add_argument("--img_feat_file", default='datasets/coco_ir/features.tsv', type=str, required=False,
                        help="The absolute address of the image feature file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type. required for training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str, 
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, "
                             "sequences shorter will be padded."
                             "This number is calculated on COCO dataset" 
                             "If add object detection labels, the suggested length should be 70.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--finetune", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="Whether to run inference.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run performance valuation."
                       "do not activate if we want to inference on dataset without gt labels.")
    parser.add_argument("--sbu", action='store_true', help="Whether to train on sbu dataset.")
    parser.add_argument("--two_datasets", action='store_true', help="Whether to train on sbu dataset.")
    parser.add_argument("--test_split", default='test', type=str, help='data split name.')
    parser.add_argument("--eval_img_keys_file", default='', type=str, 
                        help="image key tsv to select a subset of images for evaluation. "
                        "This is useful in 5-folds evaluation. The topn index file is not " 
                        "needed in this case.")
    parser.add_argument("--add_od_labels", default=False, action='store_true', help="Whether to add object detection labels or not.")
    parser.add_argument("--od_label_type", default='vg', type=str, 
                        help="label type, support vg, gt, oid")
    parser.add_argument("--att_mask_type", default='CLR', type=str, 
                        help="attention mask type, support ['CL', 'CR', 'LR', 'CLR']"
                        "C: caption, L: labels, R: image regions; CLR is full attention by default."
                        "CL means attention between caption and labels."
                        "please pay attention to the order CLR, which is the default concat order.")
    parser.add_argument("--do_lower_case", action='store_true', 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--drop_out", default=0.1, type=float, help="Drop out in BERT.")
    parser.add_argument("--max_img_seq_length", default=50, type=int, 
                        help="The maximum total input image sequence length.")
    parser.add_argument("--img_feature_dim", default=2054, type=int, 
                        help="The Image Feature Dimension.")
    parser.add_argument("--img_feature_type", default='frcnn', type=str,
                        help="Image feature type.")
    parser.add_argument("--use_img_layernorm", type=int, default=1,
                        help="Normalize image features with bertlayernorm")
    parser.add_argument("--img_layer_norm_eps", default=1e-12, type=float,
                        help="The eps in image feature laynorm layer")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--cache_representation", action='store_true',  help="cache_representations while test.")
    parser.add_argument("--output_mode", default='classification', type=str,
                        help="output mode, support classification or regression.")
    parser.add_argument("--num_labels", default=2, type=int, 
                        help="num_labels is 2 for classification and 1 for regression.")
    parser.add_argument("--num_captions_per_img_train", default=5, type=int,
                        help="number of positive matched captions for each training image.")
    parser.add_argument("--num_captions_per_img_val", default=5, type=int,
                        help="number of captions for each testing image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before backward.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial lr.")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="Weight deay.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup.")
    parser.add_argument("--scheduler", default='linear', type=str, help="constant or linear.")
    parser.add_argument("--num_workers", default=4, type=int, help="Workers in dataloader.")
    parser.add_argument("--num_train_epochs", default=20, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, 
                        help="Total number of training steps. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X steps.")
    parser.add_argument('--save_steps', type=int, default=-1, 
                        help="Save checkpoint every X steps. Will also perform evaluatin.")
    parser.add_argument("--evaluate_during_training", action='store_true', 
                        help="Run evaluation during training at each save_steps.")
    parser.add_argument("--eval_model_dir", type=str, default='', 
                        help="Model directory for evaluation.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA.")
    parser.add_argument('--seed', type=int, default=88, help="random seed for initialization.")
    parser.add_argument('--logit_gpu', type=str, default="0,1,2,3,4,5,6", help="random seed for initialization.")
    parser.add_argument('--contrastive_gpu', type=int, default=7, help="random seed for initialization.")
    parser.add_argument("--no_pretrain", action='store_true', help="Whether to use oscar pretrain checkpoint.")
    parser.add_argument("--bert_cls", action='store_true', help="Whether to use oscar pretrain checkpoint.")
    parser.add_argument("--triplet_loss", action='store_true', help="Whether to use triplet loss.")
    parser.add_argument('--temperature_t', type=float, default=0.005, help="random seed for initialization.")
    parser.add_argument('--temperature_i', type=float, default=0.005, help="random seed for initialization.")
    args = parser.parse_args()

    global logger
    mkdir(args.output_dir)
    logger = setup_logger("vlpretrain", args.output_dir, 0)

    args.cuda_for_test = torch.device('cuda:{}'.format(args.contrastive_gpu)) 
    args.gpus = [int(i) for i in args.logit_gpu.split(",")]
    args.device = torch.device("cuda", index=args.gpus[0])
    args.n_gpu = len(args.gpus)
    set_seed(args.seed, args.n_gpu)
    logger.warning("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    logger.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))
 
    config_class, tokenizer_class = BertConfig, BertTokenizer
    #model_class = BertImgModel
    if args.do_train and not args.finetune:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')#'vinvl/coco_ir/base/checkpoint-1340000',2
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        #config.loss_type = args.loss_type
        config.img_layer_norm_eps = args.img_layer_norm_eps
        config.use_img_layernorm = args.use_img_layernorm
        config.bert_cls = args.bert_cls
        if(args.no_pretrain):
            model = ContrastiveTwins(None,config)
        else:
            model = ContrastiveTwins(args.model_name_or_path,config)
    elif args.finetune:
        checkpoint = args.model_name_or_path
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Finetune from the following checkpoint: %s", checkpoint)
        model = ContrastiveTwins(None,config)
        sd = torch.load(checkpoint+"/model.cpkt", map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model = ContrastiveTwins(None,config)
        sd = torch.load(args.eval_model_dir+"/model.cpkt", map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
    
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train or args.finetune:
        if args.sbu:
            train_dataset = sbu_total_dataset(tokenizer, args, is_train=True)
        elif args.two_datasets:      
            args_coco = copy.deepcopy(args)      
            args_coco.data_dir = "/raid/data_modal/coco_vinvL/coco_ir/"
            args_coco.img_feat_file = "/raid/data_modal/coco_vinvL/model_0060000/features.tsv"
            args_coco.eval_img_keys_file = '/raid/data_modal/coco_vinvL/coco_ir/test_img_keys.tsv'
            coco = RetrievalDataset(tokenizer, args_coco, 'train', is_train=True)

            args_flickr = copy.deepcopy(args)
            args_flickr.data_dir = "/raid/data_modal/flickr30k_vinvl/captions/"
            args_flickr.img_feat_file = "/raid/data_modal/flickr30k_vinvl/model_0060000/features.tsv"
            args_flickr.eval_img_keys_file = '/raid/data_modal/flickr30k_vinvl/captions/test_img_keys.tsv'
            flickr = RetrievalDataset(tokenizer, args_flickr, 'train', is_train=True)
            train_dataset = ConcatDataset([coco,flickr])
        else:
            train_dataset = RetrievalDataset(tokenizer, args, 'train', is_train=True)
        
        if args.evaluate_during_training:
            val_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
        else:
            val_dataset = None
        global_step, avg_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = RetrievalDataset(tokenizer, args, args.test_split, is_train=False)
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        logger.info("Evaluate the following checkpoint: %s", checkpoint)

        model = ContrastiveTwins(None,config)
        sd = torch.load(args.eval_model_dir+"/model.cpkt", map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.to(args.device)
        model.device =args.device
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model,device_ids=args.gpus)

        # pred_file = get_predict_file(args)
        # if op.isfile(pred_file):
        #     logger.info("Prediction file exist, skip inference.")
        #     if args.do_eval:
        #         test_result = torch.load(pred_file)
        # else:
        test_result = test(args, model, test_dataset)
        #torch.save(test_result, pred_file)
        # logger.info("Prediction results saved to {}.".format(pred_file))

        if args.do_eval:
            eval_result = evaluate(test_dataset, test_result)
            # result_file = op.splitext(pred_file)[0] + '.eval.json'
            # with open(result_file, 'w') as f:
            #     json.dump(eval_result, f)
            # logger.info("Evaluation results saved to {}.".format(result_file))


if __name__ == "__main__":
    main()
