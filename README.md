# SpecMRL code

Qwen3-0.6B-Base training and  inference. Core implementations:
`train/modeling.py` (SpecMRL), `train/modeling_rawMRL.py` (MRL).

## Install


```bash
conda create -n specmrl python=3.10
conda activate specmrl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation -r requirements.txt
```

## Train


```bash
MODEL_PATH=/path/to/Qwen3-0.6B-Base TRAIN_DATA=/path/to/data bash train.sh SpecMRL
# Replace SpecMRL with rawMRL to train the baseline.
```


## SpecMRL inference / MTEB

```bash
CUDA_VISIBLE_DEVICES=0 python run4mteb_SpecMRL-alldims.py \
    --model_name /path/to/Qwen3-0.6B-Base \
    --trained_model_path ./trained_models/SpecMRL/checkpoint-3000 \
    --pool mean --batch_size 512 --dims 1024 512 256 128 64 \
    --output_dir ./output/SpecMRL-checkpoint-3000
```

## Citation

```bibtex
@misc{chen2026specmrlredesignmatryoshkarepresentation,
      title={SpecMRL: Redesign Matryoshka Representation Learning within Parametric Spectrum}, 
      author={Zhongxin Chen and Songhao Wu and Heng Cui and Cong Li and Xiao Zhou and Rui Yan},
      year={2026},
      eprint={2610.26824},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.26824}, 
}
```

