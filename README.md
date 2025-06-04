# GRU-based Deepfake Detection with MiniGPT-4

This project adapts the open-source [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) model for the task of video-based deepfake detection by implementing GRU with our frame analysis.

The model processes multiple frames for each video, encodes the info with MiniGPT's architecture, and uses LoRA fine-tuning for binary classification (real vs fake). 

# Setting Up the Model - Anaconda #
1: Model Weights
Download the Llama 2 7B Model Weights from: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
Place the models weights in the ./Llama-2-7b-chat-hf/ folder.
      
2: Install Requirements and Prepare the Environment
      All python requirements to run the scripts are specified in the requirements.txt file. Make sure that your setup support anaconda, the environment is already prepared for you from the original repository. To activate the environment, you can run:
            conda activate minigptv

3: Prepare Dataset
      To start, follow the instructions here: https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus and then organize the data as such, starting from the root of this repo:
```
      ./data
      ├── train
      │   └── lrs3
      │       └── [video folders and files]
      ├── val
      │   └── lrs3
      │       └── [video folders and files]
```
In the repository, we include two python scripts to build the json file that our dataset loader uses to access the files from. Both follow the same pattern, but are premodified to support file building for the training and validation data. You can modify the directories as you please, but to run these without any modification, place the train_metadata.py inside the /train folder, where it should be sharing the same space as the /lrs3 folder, and the same with the corresponding val_metadata.py inside the /val folder.

4: Training
      From the root of the directory, make sure your anaconda environment is activated, then run:
            python train_deepfake_temporal.py

5: Validation
      From the same root, you can run:
            python eval.py
      which will output the average loss, AUC, and accuracy.

> This repository includes modified source code originally licensed under the [BSD 3-Clause License](LICENSE.md) from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). All original license terms and attributions are preserved.

## 📖 Citation

This project is built on MiniGPT-4, developed by the Vision-CAIR team.

If you use this code or model in your work, please consider citing their original papers:

```bibtex
@article{chen2023minigptv2,
      title={MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning}, 
      author={Chen, Jun and Zhu, Deyao and Shen, Xiaoqian and Li, Xiang and Liu, Zechu and Zhang, Pengchuan and Krishnamoorthi, Raghuraman and Chandra, Vikas and Xiong, Yunyang and Elhoseiny, Mohamed},
      year={2023},
      journal={arXiv preprint arXiv:2310.09478},
}

@article{zhu2023minigpt,
  title={MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models},
  author={Zhu, Deyao and Chen, Jun and Shen, Xiaoqian and Li, Xiang and Elhoseiny, Mohamed},
  journal={arXiv preprint arXiv:2304.10592},
  year={2023}
}
