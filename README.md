# GRU-based Deepfake Detection with MiniGPT-4

This project adapts the open-source [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) model for the task of video-based deepfake detection by implementing GRU with our frame analysis.

The model processes multiple frames for each video, encodes the info with MiniGPT's architecture, and uses LoRA fine-tuning for binary classification (real vs fake).

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
