import os

import pandas as pd
import torch
import argparse

from os.path import join
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

# from transformers import T5ForConditionalGeneration,AutoModelForSeq2SeqLM,AutoConfig
from pytorch_lightning import Trainer
from utilities.functions import get_library
from model.T5FineTuner import T5FineTuner
from model.eval_support_functions import calculate_accuracies, save_results, save_to_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_folder', type=str, default='/root/autodl-tmp/llm4vis/main')
    parser.add_argument('--model_type', type=str, default='codet5p-770m'
                        ,choices=['t5-small','t5-base','t5-large','t5-3b','t5-11b',
                                  'codet5-small','codet5-base','codet5-large',
                                    'codet5p-220m','codet5p-770m','codet5p-2b'])
    parser.add_argument('--use_original_ckpt', action='store_true', help='use original ckpt or not')
    parser.add_argument('--ckpt_path', type=str, default='None', help='ckpt path')
    parser.add_argument('--mode', type=str, default='test',choices=['test','debug'])
    parser.add_argument("--task", type=str, default="nl2vis_v0", help="task name",choices=['nl2vis_v0','nl2vis','vis2nl','QA','table2nl','MTF'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    args, _ = parser.parse_known_args()
    if args.task == 'nl2vis_v0':
        # parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=126, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Generate VQL:', help='prefix for the task')
        parser.add_argument('--num_beams', type=int, default=1, help='num_beams')
        parser.add_argument('--min_length', type=int, default=16, help='min_length')
        
    elif args.task == 'nl2vis':
        # parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=144, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Generate VQL:', help='prefix for the task')
        parser.add_argument('--num_beams', type=int, default=1, help='num_beams')
        parser.add_argument('--min_length', type=int, default=16, help='min_length')
        
    elif args.task == 'vis2nl':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=81, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Translate VQL to NL:', help='prefix for the task')
        parser.add_argument('--num_beams', type=int, default=4, help='num_beams')
        parser.add_argument('--min_length', type=int, default=8, help='min_length')
    elif args.task == 'QA':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=240, help='max target token length')
        parser.add_argument('--prefix', type=str, default='Generate Answer:', help='prefix for the task')
        parser.add_argument('--num_beams', type=int, default=4, help='num_beams')
        parser.add_argument('--min_length', type=int, default=3, help='min_length')
    elif args.task == 'table2nl':
        parser.add_argument('--max_input_length', type=int, default=512, help='max input token length')
        parser.add_argument('--max_target_length', type=int, default=512, help='max target token length') # 622
        parser.add_argument('--prefix', type=str, default='Translate Table to NL:', help='prefix for the task')
        parser.add_argument('--num_beams', type=int, default=4, help='num_beams')
        parser.add_argument('--min_length', type=int, default=8, help='min_length')
    # elif args.task == 'MTF':
        # parser.add_argument('--max_input_length', type=int, default=126, help='max input token length')
        # parser.add_argument('--max_target_length', type=int, default=512, help='max target token length')
        # parser.add_argument('--prefix', type=str, default='None', help='prefix for the task')
        # parser.add_argument('--num_beams', type=int, default=1, help='num_beams')
        # parser.add_argument('--min_length', type=int, default=15, help='min_length')
    
    # exp_version
    parser.add_argument('--model_dir', type=str, default='2024-01-02', help='model directory')
    parser.add_argument('--exp_version', type=str, default='v0', help='exp_version')

    parser.add_argument('--deep_speed_type', type=str, default='deepspeed_stage_2'
                        ,choices=['deepspeed_stage_1','deepspeed_stage_2','deepspeed_stage_2_offload','deepspeed_stage_3','deepspeed_stage_3_offload'])


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    os.chdir(args.project_folder)
    # device = 'cuda:'+args.device if torch.cuda.is_available() else 'cpu'
    # print('Using device:',device)
    print('Current working task:',args.task)
    # load model
    if args.ckpt_path == 'None' and not args.use_original_ckpt:
        MODEL_FOLDER = join(args.project_folder, args.task +'/'+ args.model_type+'/'+args.model_dir+'-'+args.exp_version)
        
        MODEL_TYPE, tokenizer_library= get_library(args.model_type)
        args.ckpt_path = MODEL_FOLDER
    T5_model = T5FineTuner(args)
    # T5_model.model = T5_model.model.to(device)

    # T5_model.model.eval()
    
    # Step 1 : eaval
    print('-'*50)
    print('Start evaluating...')
    trainer = Trainer(
        logger=False,
        strategy=args.deep_speed_type,
    )
    test_result = trainer.test(T5_model)
    df_results = T5_model.result
    
    # avoid run failed
    save_to_csv(df_results, args)
    
    # Step 2 : calculate accuracies
    print('-'*50)
    print('Start calculating accuracies...')
    df_results = calculate_accuracies(df_results, args.task, T5_model.tokenizer)
    
    # Step 3 : save results
    print('-'*50)
    print('Start saving results...')
    print('Task name:',args.task)
    save_results(df_results, args)