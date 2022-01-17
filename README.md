# Setup the computing environment
```shell
# Conda Installation
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
# Create the environment
conda create -n <name> python=3.8
conda activate <name>
# Install independencies
pip install -r requirements.txt
```
This repo is built upon **Pytorch** (deep neural networks) and **PytorchGeometrics** (graph learning). For [PyTorch](https://pytorch.org) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) installation, please follow the guide from the websites. Make sure you install the matched the version of Pytorch for PyG.

# Transformer benchmark architecture (Informer -> AAAI'21 best paper)
Our GTA's **Transformer** architecture is built on top of [**Informer**](https://github.com/zhouhaoyi/Informer2020) who won the best paper award of AAAI'21. One may refer to the link to receive more details.

# Usage
For training of GTA for deep anomaly detection (dad), please refer to `main_gta_dad.py` for more information.
We also provided a more detailed and complete cli description for training the model:
```
python -u main_gta_dad.py --model <model> --data <data>
--root_path <root_path> --data_path <data_path> --features <features>
--target <target> --freq <freq> --checkpoints <checkpoints>
--seq_len <seq_len> --label_len <label_len> --pred_len <pred_len>
--enc_in <enc_in> --dec_in <dec_in> --c_out <c_out> --d_model <d_model>
--n_heads <n_heads> --e_layers <e_layers> --d_layers <d_layers>
--s_layers <s_layers> --d_ff <d_ff> --factor <factor> --dropout <dropout> 
--attn <attn> --embed <embed> --activation <activation>
--num_workers <num_workers> --train_epochs <train_epochs> --itr <itr>
--batch_size <batch_size> --patience <patience> --des <des>
--learning_rate <learning_rate> --loss <loss> --lradj <lradj>
--use_gpu <use_gpu> --gpu <gpu>
```
We would update the repo by providing a version that supports multi-gpus using [Pytorch Lightning](https://www.pytorchlightning.ai).

## Citation
If you find this repository useful in your research, please consider citing the following paper:

```
@ARTICLE{zekaietal-gta-2021,
  author    = {Chen, Zekai and
               Chen, Dingshuo and
               Zhang, Xiao and 
               Yuan, Zixuan and 
               Cheng, Xiuzhen},
  journal   = {IEEE Internet of Things Journal}, 
  title     = {Learning Graph Structures with Transformer for Multivariate Time Series Anomaly Detection in IoT}, 
  year      = {2021},
  pages     = {1-1},
  doi       = {10.1109/JIOT.2021.3100509}}
```

## Contact
If you have any questions, feel free to contact Zekai Chen through Email (zekai.chen@bms.com) or Github issues. Pull requests are highly welcomed!
