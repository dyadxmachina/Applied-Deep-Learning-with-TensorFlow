import argparse
import tensorflow as tf
import train.model as model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--traindir',
        help='Training data directory or files',
        required=True
    )
    parser.add_argument(
        '--evaldir',
        help='Eval data directory or files',
        required=True
    )
    parser.add_argument(
        '--bucket',
        help='bucket for training/test data',
        required=False,
    )    
    parser.add_argument(
        '--batchsize',
        help='Batch size for training',
        required=True,
        type=int,
        default=256
    )         
    parser.add_argument(
        '--epochs',
        help='Epochs for training',
        required=False,
        type=int,
        default=10                
    )  
    parser.add_argument(
        '--job-dir',
        help='Job dir for ml engine',
        required=True
    )  
    parser.add_argument(
        '--job_dir',
        help='Job dir for ml engine',
        required=False
    )      
    parser.add_argument(
        '--outputdir',
        help='Output dir for training/eval',
        required=True
    )   
    parser.add_argument(
        '--hidden_units',
        help='Output dir for training/eval',
        required=False
    )
    parser.add_argument(
        '--embedding',
        help='Output dir for training/eval',
        required=False
    )
    parser.add_argument(
        '--learn_rate',
        help='Learning rate',
        required=False
    )    
                 
        
    # parse args
    args = parser.parse_args()
    arguments = args.__dict__
    print(arguments)

    model.train_eval(**arguments)