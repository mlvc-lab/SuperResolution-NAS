# SR-NAS
This project is about Network architecture searching in SR domain.
## Dependencies (Experiment Env)
* Python 3.7.4
* PyTorch = 1.4.0
* numpy
* skimage
* imageio
* matplotlib
* tqdm


## Data Path
Only DIV2K dataset is not uploaded because of the project size restriction. [DIV2K Download link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

```
By using --dir_data option, you can change default data path. 
Place the DIV2K Dataset into Project relative path ~/Dataset/DIV2K/ with HR, LR division.
ex) HRdata format: ~/PycharmProjects/HNAS/Dataset/DIV2K/DIV2K_train_HR/0001.png
    LRdata format: ~/PycharmProjects/HNAS/Dataset/DIV2K/DIV2K_train_LR_bicubic/X2/0001x2.png
```
then, demo script will be executed successfully.


## Quick start 
Place the SR datasets to the path of  'dir_data' as defined in the option.py file.  
Or by using --dir_data option in args parser execution command, change data directory absolute path.  
Run the following command to quick start our project
before you try this, goto **Data Path** Devision in below to download DIV2K dataset.

```bash
    cd src       
    sh demo.sh
```


The HNAS work can be splitted into four procedures:  
1. At search stage, we train the hierarchical controllers for architecture search.  
    ```bash
    CUDA_VISIBLE_DEVICES=0 python search.py --model ENAS --scale 2 --patch_size 96 --save search_model --reset --data_test Set5 --layers 12 --init_channels 8 --entropy_coeff 1 --lr 0.001 --epoch 400 --flops_scale 0.2
    ```

2. At infer stage, we infer some promising architectures.(optional)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python derive.py --data_test Set5 --scale 2 --pre_train  ../experiment/search_model/model/model_best.pt  --test_only --self_ensemble --save_results --save result/ --train_controller False --model ENAS --layer 12 --init_channels 8 --seed 1  
    ```

3. At re-train stage, we re-train the seached architecture from scratch. 
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --model arch --genotype HNAS_A --scale 2 --patch_size 96 --save retrain_result --reset --data_test Set5 --data_range 1-800/801-810 --layers 12 --init_channels 64 --lr 1e-3 --epoch 300 --upsampling_Pos 9 --n_GPUs 1
    ```

3. At test stage, we test our final model on five public standard datasets. 
    ```bash
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test Set5+Set14+B100+Urban100+Manga109 --data_range 801-900 --scale 2 --pre_train  ../experiment/retrain_result/model/model_best.pt  --test_only --self_ensemble --save_results --save result_arch/ --train_controller False --model arch --genotype HNAS_A --layer 12 --init_channels 64 --upsampling_Pos 9
    ```

## Citation

If you use any part of this code in your research, please cite our paper:

```
@article{guo2020hierarchical,
  title={HNAS: Hierarchical Neural Architecture Search for Single Image Super-Resolution},
  author={Guo, Yong and Luo, Yongsheng and He, Zhenhao and Huang, Jin and Chen, Jian},
  journal={arXiv preprint arXiv:2003.04619},
  year={2020}
}
```
