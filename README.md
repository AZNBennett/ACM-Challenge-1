# GRU-based Deepfake Detection with MiniGPT-4

This project adapts the open-source [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) model for the task of video-based deepfake detection by implementing GRU with our frame analysis.

The model processes multiple frames for each video, encodes the info with MiniGPT's architecture, and uses LoRA fine-tuning for binary classification (real vs fake). The model will also target

> This repository includes modified source code originally licensed under the [BSD 3-Clause License](LICENSE.md) from [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4). All original license terms and attributions are preserved.

## 📖 Citation

This project is built on MiniGPT-4, developed by the Vision-CAIR team.

If you use this code or model in your work, please consider citing their original paper:

```bibtex
@misc{zhu2023minigpt4,
      title={MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models},
      author={Deyao Zhu and Jun Chen and Xiaoqian Shen and Xiang Li and Harry Yang and Xiyang Dai and Yutong Feng and Linjie Li and Jianwei Yang and Pengchuan Zhang and Lu Yuan and Lijuan Wang},
      year={2023},
      eprint={2304.10592},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
