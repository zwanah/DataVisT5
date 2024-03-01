__author__ = "Zhuoyue WAN Russ"

import os
import pytorch_lightning as pl
import torch
import argparse
import datetime
import wandb

from os.path import join

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.utility import HistoryCallback, EpochProgressBar
from model.T5FineTuner import T5FineTuner
from utilities.functions import check_create_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, default='/root/autodl-tmp/llm4vis')
    parser.add_argument('--model_type', type=str, default='codet5p-770m'
                        ,choices=['t5-small','t5-base','t5-large','t5-3b','t5-11b',
                                    'codet5-small','codet5-base','codet5-large',
                                    'codet5p-220m','codet5p-770m'])
    parser.add_argument('--use_original_ckpt', action='store_true', help='use original ckpt or not')
    parser.add_argument('--ckpt_path', type=str, default=None, help='ckpt path')
    parser.add_argument('--mode', type=str, default='train',choices=['train','debug'])
    parser.add_argument("--task", type=str, default="nl2vis_v0", help="task name",choices=['nl2vis_v0','nl2vis_v1','nl2vis','vis2nl','QA','table2nl','MTF'])

    args, _ = parser.parse_known_args()
    if args.task == 'nl2vis_v0':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=126, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Generate VQL:', help='prefix for the task')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size') #16,8
    elif args.task == 'nl2vis':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length') # A100 use 1024 limited GPU memory
        parser.add_argument('--max_target_length', type=int, default=144, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Generate VQL:', help='prefix for the task')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size') #16,8
    elif args.task == 'vis2nl':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length') # A100 use 512
        parser.add_argument('--max_target_length', type=int, default=81, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Translate VQL to NL:', help='prefix for the task')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size') #16,8
    elif args.task == 'QA':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=240, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Generate Answer:', help='prefix for the task')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size') #16,8         
    elif args.task == 'table2nl':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=512, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Translate Table to NL:', help='prefix for the task')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size') #16,8
    elif args.task == 'MTF':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=512, help='max target token length')
        parser.add_argument('--prefix', type=str, default='None', help='prefix for the task')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size') #16,8

    # # device
    # parser.add_argument('--device', type=str, default='0', help='device number')
    
    # exp_version
    parser.add_argument('--exp_version', type=str, default='v0', help='exp_version')

    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    # hyper parameters
    parser.add_argument('--deep_speed_type', type=str, default='deepspeed_stage_2_offload'
                        ,choices=['deepspeed_stage_1','deepspeed_stage_2','deepspeed_stage_2_offload','deepspeed_stage_3','deepspeed_stage_3_offload'])

    parser.add_argument('--precision', type=str, default='16-mixed', help='precision',choices=['bf16','16','16-mixed'])
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--warmup_rate', type=float, default=0.1, help='warmup rate')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--adam_epsilon', type=float, default=1e-7, help='adam epsilon')
    parser.add_argument('--final_cosine', type=float, default=5e-8, help='final cosine')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(1234)
    ####
    os.environ["WANDB_MODE"] = "offline"

    # os.chdir(args.project_folder)
    if args.mode == 'debug':
        args.exp_version = 'debug'
    # precision
    if args.model_type in ['t5-small','t5-base','t5-large','t5-3b','t5-11b']:
        args.precision = 'bf16'
    elif args.model_type in ['codet5-small','codet5-base','codet5-large','codet5p-220m','codet5p-770m']:
        args.precision = '16-mixed'
    if args.deep_speed_type == 'deepspeed_stage_1' or args.deep_speed_type == 'deepspeed_stage_2' or args.deep_speed_type == 'deepspeed_stage_3':
        args.adam_name = 'AdamW'
    elif args.deep_speed_type == 'deepspeed_stage_2_offload' or args.deep_speed_type == 'deepspeed_stage_3_offload':
        args.adam_name = 'DeepSpeedCPUAdam'
        
    print('Current working task:',args.task)
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d")
    current_time = current_time + '-' + args.exp_version

    if args.use_original_ckpt:
        MODEL_FOLDER = join(args.project_folder, 'ckpt/STF/'+args.task +'/'+ args.model_type+'/'+current_time)
        LOGGER_FOLDER = join(args.project_folder, 'ckpt/STF/'+args.task+'/'+ args.model_type+'/'+current_time+"/logger")
    else:
        MODEL_FOLDER = join(args.project_folder, 'ckpt/pt_STF/'+args.task +'/'+ args.model_type+'/'+current_time)
        LOGGER_FOLDER = join(args.project_folder, 'ckpt/pt_STF/'+args.task+'/'+ args.model_type+'/'+current_time+"/logger")

    check_create_folder(MODEL_FOLDER,ask_to_rm_if_exists=False)
    check_create_folder(LOGGER_FOLDER,ask_to_rm_if_exists=False)

    model = T5FineTuner(args)
    tokenizer = model.tokenizer
    
    logger = CSVLogger(save_dir=LOGGER_FOLDER,name=f"{current_time}")
    wandb_logger = WandbLogger(
        dir=LOGGER_FOLDER,
        save_dir=LOGGER_FOLDER,
        name= args.task +'_'+ args.model_type+'_'+current_time,
        project="LLM4Vis", 
        group=args.task,
        config=args,
        log_model="all"
    )

    # Callbacks
    progress_bar = EpochProgressBar()
    history = HistoryCallback()

    # checkpoint_callback = ModelCheckpoint(
    #     monitor='validation_loss',
    #     dirpath=MODEL_FOLDER,
    #     filename='model_best',
    #     save_top_k=1,
    #     mode='min'
    # )
    # early_stop_callback = EarlyStopping(
    #     monitor='validation_loss',
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode='min'
    # )

    # Trainer
    trainer = Trainer(
        default_root_dir=MODEL_FOLDER,
        callbacks=[progress_bar, history],
        logger=[wandb_logger],
        max_epochs=args.num_epochs,
        precision=args.precision, # T5 -> bf16; CodeT5 -> fp16:16-mixed
        gradient_clip_val=args.gradient_clip_val,
        strategy = args.deep_speed_type
    )

    trainer.fit(model)

    history.history_dataframe()
    model.model.save_pretrained(MODEL_FOLDER)    
    tokenizer.save_pretrained(join(MODEL_FOLDER, "tokenizer"))
    
    wandb_logger.experiment.finish()
    
    

        





    
