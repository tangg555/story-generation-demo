# Story-Generation-Demo
A simple story generation demo which finetines Huggingface pretrained model to generate stories.

## Instructions
This project is based on [pytorch-lightning](https://www.pytorchlightning.ai/) framework, and the pretrained model [BART](https://aclanthology.org/2020.acl-main.703.pdf) is downloaded from [Hugginface: bart-base](https://huggingface.co/facebook/bart-base).

So if you want to run this code, you must have following preliminaries:
- Python 3 or Anaconda (mine is 3.8)
- [Pytorch](https://pytorch.org/) 
- transformers (a package for [huggingface](https://huggingface.co/facebook/bart-base))
- [pytorch-lightning (a package)](https://www.pytorchlightning.ai/)

## Quick Start

#### 1. Install packages
```shell
python -r requirements.txt
```

#### 2. Collect Datasets and Resources
`datasets` and `resources` are separate from the code, since they are too large. Both of them are offered in ...
#####2.1 Datasets
The structure of `datasets`should be like this:
```markdown
├── datasets
   └── story-generation		# expeirmtn group name
       ├── `roc`        # ROCStories
              └── `train.source.txt`    # leading context of stories
              └── `train.target.txt`       # story corresponding to the leading context
              └── `val.source.txt` 
              └── `val.target.txt` 
              └── `test.source.txt` 
              └── `test.target.txt` 
```
The raw dataset of roc story can be accessed for free. Google and get it. e.g. [homepage](https://cs.rochester.edu/nlp/rocstories/) .

train, val, test are split by the ratio of 0.90, 0.05, 0.05

the example of `test.source.txt` (leading context):

`ken was driving around in the snow .`

the example of `test.target.txt` (story):

`he needed to get home from work . he was driving slowly to avoid accidents . unfortunately the roads were too slick and ken lost control . his tires lost traction and he hit a tree . `
#####2.1 Resources
The structure of resources should be like this:
```markdown
├── resources
   └── external-generation		
       ├── `bart-base`        
              └── `config.json`    
              └── `pytorch_model.bin`       
              └── ...
```
`bart-base` can be downloaded from [here](https://huggingface.co/facebook/bart-base)

#### 3. Fine-tuning BART on ROCStories
I have set all essential parameters, so you can directly run 

`python ./tasks/story-generation/train.py`

**Or** 

If you want to modify parameters, you can run
```shell
python tasks/story-generation/train.py --data_dir=datasets/story-generation/roc-stories\
 --learning_rate=5e-5 \
 --train_batch_size=16 \
 --eval_batch_size=10 \
 --model_name_or_path=resources/external_models/bart-base \
 --output_dir=output/story-generation \
 --model_name leading-bart \
 --experiment_name=leading-bart-roc-stories\
 --val_check_interval=1.0 \
 --limit_val_batches=10 \
 --max_epochs=3 \
 --accum_batches_args=4
```

#### 4. Generating Stories and Evaluation
Same to training. Directly run 

`python ./tasks/story-generation/test.py`

**Or** 

```shell
python tasks/story-generation/test.py --data_dir=datasets/story-generation/roc-stories \
  --eval_batch_size=10 \
  --model_name_or_path=output/story-generation/leading-bart-roc-stories/best_tfmr \
  --output_dir=output/story-generation \
  --model_name leading-bart \
  --experiment_name=leading-bart-roc-stories
```