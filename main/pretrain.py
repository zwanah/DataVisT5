__author__ = "Zhuoyue WAN Russ"

import os
from random import shuffle
import pytorch_lightning as pl
import torch
import argparse
import datetime
import wandb
import math

from os.path import join
from torch.utils.data import DataLoader

from transformers import T5ForConditionalGeneration,AutoConfig,AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import concatenate_datasets

from model.utility import HistoryCallback, EpochProgressBar, get_lr_scheduler, get_optimizer
from model.CustomDataset import load_mode_data,MixedDataset
from model.copied_utils import compute_input_and_target_lengths
from model.mymlm_utils import MyDataCollatorForT5MLM, process_dataset,process_dataset_seq_desc
from utilities.functions import check_create_folder,get_library


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, default='/home/russ/LLM4Vis/llm4vis')
    parser.add_argument('--model_type', type=str, default='codet5p-770m'
                        ,choices=['t5-small','t5-base','t5-large','t5-3b','t5-11b',
                                    'codet5-small','codet5-base','codet5-large',
                                    'codet5p-220m','codet5p-770m','codet5p-2b',
                                    'codet5p-6b','codet5p-16b'])
    parser.add_argument('--mode', type=str, default='pretrain',choices=['pretrain','debug'])
    parser.add_argument("--task", type=str, default="pretrain", help="task name",choices=['pretrain'])

    parser.add_argument('--streaming', action='store_true', help='streaming')
    parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size') #16,8

    # exp_version
    parser.add_argument('--exp_version', type=str, default='v0', help='exp_version')

    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs') # 200

    # hyper parameters
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='mlm_probability')
    parser.add_argument('--mean_noise_span_length', type=int, default=3, help='mean_noise_span_length')

    parser.add_argument('--deep_speed_type', type=str, default='deepspeed_stage_2'
                        ,choices=['deepspeed_stage_1','deepspeed_stage_2','deepspeed_stage_2_offload','deepspeed_stage_3','deepspeed_stage_3_offload'])

    parser.add_argument('--acc_grad_batches', type=int, default=1, help='accumulate_grad_batches')
    parser.add_argument('--precision', type=str, default='16-mixed', help='precision',choices=['bf16','16','16-mixed','32'])
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--warmup_rate', type=float, default=0.1, help='warmup rate')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--final_cosine', type=float, default=5e-8, help='final cosine')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(1234)

    # precision
    if args.model_type in ['t5-small','t5-base','t5-large','t5-3b','t5-11b']:
        args.precision = 'bf16'
    elif args.model_type in ['codet5-small','codet5-base','codet5-large','codet5p-220m','codet5p-770m','codet5p-2b','codet5p-6b','codet5p-16b']:
        args.precision = '16-mixed'
        
    if args.deep_speed_type == 'deepspeed_stage_1' or args.deep_speed_type == 'deepspeed_stage_2' or args.deep_speed_type == 'deepspeed_stage_3':
        args.adam_name = 'AdamW'
    elif args.deep_speed_type == 'deepspeed_stage_2_offload' or args.deep_speed_type == 'deepspeed_stage_3_offload':
        args.adam_name = 'DeepSpeedCPUAdam'

    print('Current working task:',args.task)
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d")
    current_time = current_time + '-' + args.exp_version
    
    MODEL_FOLDER = join(args.project_folder, 'ckpt/'+args.task +'/'+ args.model_type+'/'+current_time)
    LOGGER_FOLDER = join(args.project_folder, 'ckpt/'+args.task+'/'+ args.model_type+'/'+current_time+"/logger")

    # check_create_folder(CHECKPOINT_FOLDER,ask_to_rm_if_exists=False)
    check_create_folder(MODEL_FOLDER,ask_to_rm_if_exists=False)
    check_create_folder(LOGGER_FOLDER,ask_to_rm_if_exists=False)

    # including train and val
    text_spilts_datasets, src_tgt_spilts_datasets= load_mode_data('',args.task, args.mode, args.streaming)

    MODEL_TYPE, tokenizer_library= get_library(args.model_type)

    config = AutoConfig.from_pretrained(MODEL_TYPE)
    # model = T5ForConditionalGeneration.from_pretrained(MODEL_TYPE,config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE,use_fast=True)
    tokenizer.model_max_length = int(1e9)
    
    special_tokens_dict = {'additional_special_tokens': ["<NL>", "<VQL>","<schema>","<Question>","<Answer>","<Table>"]}
    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)

    # We increase the input_length, because instead of masking tokens T5 replaces
    # masked spans with a single token, therefore to avoid padding we need to have
    # longer sequences at the start, before masking
    before_mask_input_length, target_length = compute_input_and_target_lengths(
        inputs_length=args.max_input_length,
        noise_density=args.mlm_probability,
        mean_noise_span_length=args.mean_noise_span_length,
    )
    # 将before_mask_input_length和target_length加入args
    args.before_mask_input_length = before_mask_input_length
    args.target_length = target_length
    args.MODEL_TYPE = MODEL_TYPE
    
    if args.streaming:

        NL = process_dataset(text_spilts_datasets['NL'], args, tokenizer)
        VQL = process_dataset(text_spilts_datasets['VQL'], args, tokenizer)
        schema = process_dataset(text_spilts_datasets['schema'], args, tokenizer)
        Table = process_dataset(text_spilts_datasets['Table'], args, tokenizer)
        
        nl2vis_src_trg = process_dataset_seq_desc(src_tgt_spilts_datasets['nl2vis'], args, tokenizer)
        vis2nl_src_trg = process_dataset_seq_desc(src_tgt_spilts_datasets['vis2nl'], args, tokenizer)
        QA_src_trg = process_dataset_seq_desc(src_tgt_spilts_datasets['QA'], args, tokenizer)
        table2nl_src_trg = process_dataset_seq_desc(src_tgt_spilts_datasets['table2nl'], args, tokenizer)
            
        mixed_dataset_split_train = MixedDataset(NL['train'], VQL['train'], schema['train'], Table['train'], 
                                                nl2vis_src_trg['train'], vis2nl_src_trg['train'],
                                                QA_src_trg['train'], table2nl_src_trg['train'])
        mixed_dataset_split_test = MixedDataset(NL['test'], VQL['test'], schema['test'], Table['test'],
                                                nl2vis_src_trg['test'], vis2nl_src_trg['test'],
                                                QA_src_trg['test'], table2nl_src_trg['test'])
        mixed_dataset_split = {'train':mixed_dataset_split_train,'test':mixed_dataset_split_test}
        args.training_set_len = 10000 ########
        args.batches_per_epoch = math.ceil(args.training_set_len / args.batch_size) #34468
    else:
        text_datasets = process_dataset(text_spilts_datasets, args, tokenizer) # 63235,9657 when max_input_length=256
        src_tgt_datasets = process_dataset_seq_desc(src_tgt_spilts_datasets, args, tokenizer) # 106321, 19258
    
        mixed_dataset_split_train = concatenate_datasets([text_datasets['train'], src_tgt_datasets['train']])
        mixed_dataset_split_test = concatenate_datasets([text_datasets['test'], src_tgt_datasets['test']])
        mixed_dataset_split = {'train':mixed_dataset_split_train,'test':mixed_dataset_split_test}
        args.training_set_len = len(mixed_dataset_split['train'])
        args.batches_per_epoch = math.ceil(args.training_set_len / args.batch_size)

    data_collator = MyDataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=args.mlm_probability,
            mean_noise_span_length=args.mean_noise_span_length,
            input_length=args.max_input_length,
            target_length=args.target_length,
            pad_token_id=config.pad_token_id,
        )
    
    dataloaders = {}
    for split in ['train', 'test']:
        if split == 'test':
            batch_size = args.batch_size * 2
            shuffle = False
        else:
            batch_size = args.batch_size
            shuffle = True
        dataloaders[split] = DataLoader(
            mixed_dataset_split[split],
            shuffle=shuffle,
            batch_size=batch_size,
            collate_fn=data_collator,
            pin_memory=True,
            drop_last=False,
        )

    class T5PreTrainer(pl.LightningModule):
        def __init__(self, args):
            super(T5PreTrainer, self).__init__()
            self.args = args
            self.save_hyperparameters(args)
            self.config = AutoConfig.from_pretrained(args.MODEL_TYPE)
            self.model = T5ForConditionalGeneration.from_pretrained(args.MODEL_TYPE,config=self.config)
            self.model.resize_token_embeddings(len(tokenizer))

        def forward(self, input_ids, attention_mask, labels):
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,labels=labels
            )
            return outputs

        def common_step(self, batch, batch_idx, calc_acc=False):
            batch['attention_mask'] = batch['input_ids'] != self.config.pad_token_id
            outputs = self(**batch)

            # The pretrained model aut calcs the loss
            loss = outputs.loss
            if calc_acc:
                correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
                accuracy = correct / batch["labels"].numel()
                return loss, accuracy
            else:
                return loss

        def training_step(self, batch, batch_idx):
            loss = self.common_step(batch, batch_idx)
            self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            last_lr = self.lr_schedulers().get_last_lr()[0]
            self.log("learning_rate", last_lr,on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

            return loss

        def validation_step(self, batch, batch_idx):
            loss, accuracy= self.common_step(batch, batch_idx,calc_acc=True)
            self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return loss
        
        def compute_warmup_steps(self):
            total_training_steps = self.args.num_epochs * self.args.batches_per_epoch
            # warmup_steps = int(np.round(self.args.num_epochs / 3)) * batches_per_epoch
            warmup_steps = int(total_training_steps * self.args.warmup_rate)  # warmup for the first 10% of 

            return warmup_steps, total_training_steps
        
        def configure_optimizers(self):
            warmup_steps, total_training_steps = self.compute_warmup_steps()
            optimizer = get_optimizer(self.args, self.parameters())

            lr_scheduler = get_lr_scheduler('linear',optimizer, warmup_steps, total_training_steps,final_cosine=self.args.final_cosine)

            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        def train_dataloader(self):
            return dataloaders['train']

        def val_dataloader(self):
            return dataloaders['test']

    pt_model = T5PreTrainer(args)
    
    logger = CSVLogger(save_dir=LOGGER_FOLDER,name=f"{current_time}")
    wandb_logger = WandbLogger(
        dir=LOGGER_FOLDER,
        save_dir=LOGGER_FOLDER,
        name= args.task +'_'+ args.model_type+'_'+current_time,
        project="LLM4Vis", 
        group=args.task,
        config=args,
    )

    # Callbacks
    progress_bar = EpochProgressBar()
    history = HistoryCallback()
    # # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        dirpath=MODEL_FOLDER,
        filename='model_best',
        save_top_k=1,
        mode='max'
    )
    # early_stop_callback = EarlyStopping(
    #     monitor='validation_loss',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='min',
    #     check_on_train_epoch_end=True
    # )

    # Trainer
    trainer = Trainer(
        default_root_dir=MODEL_FOLDER,
        callbacks=[progress_bar, history, checkpoint_callback],
        logger=[wandb_logger],
        max_epochs=args.num_epochs,
        limit_train_batches = args.batches_per_epoch,
        precision=args.precision, # T5 -> bf16; CodeT5 -> fp16:16  use 16-mixed
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.acc_grad_batches,
        strategy = args.deep_speed_type,
    )

    trainer.fit(pt_model)

    history.history_dataframe()
    pt_model.model.save_pretrained(MODEL_FOLDER)    
    tokenizer.save_pretrained(join(MODEL_FOLDER, "tokenizer"))

    wandb_logger.experiment.finish()