# ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting

[![arXiv](https://img.shields.io/badge/arXiv-2406.19976-b31b1b.svg)](https://arxiv.org/abs/2406.19976)

ScaleBiO is a practical, **first-order bilevel optimization** framework for **data reweighting** in large-scale LLM training. By pairing the algorithm with the **memory-efficient LISA training technique**, the paper demonstrates scaling to ~**30B-parameter** models on **8× H100** GPUs and shows improvements on **instruction following** and **math reasoning** over strong baselines (uniform sampling, influence-aware filtering, and reference-model-based sampling). ([arXiv][1])

---

## ✨ Highlights

* **First scalable instantiation** of modern first-order bilevel optimization for LLMs.
* **Data reweighting** learned end-to-end to up/down-weight training examples.
* **Scales to ~30B** parameters with LISA for memory efficiency.
* **Consistent gains** on instruction-following & math-reasoning tasks across **Llama-3-8B, Gemma-2-9B, Qwen-2-7B, Qwen-2.5-32B**.
* **Theory:** optimality of learned weights and convergence matching prior first-order bilevel methods (smooth & strongly convex settings).

---

## 🧠 What is ScaleBiO?

ScaleBiO optimizes *model parameters* (lower level) and *per-example data weights* (upper level) jointly, but **without second-order differentiation**. This makes bilevel training practical for modern LLMs, where full Hessians or implicit gradients are prohibitive. In practice, ScaleBiO alternates (or interleaves) standard gradient updates with lightweight **weight-update steps** informed by validation feedback, all in a first-order manner. ([arXiv][1])

---

## 🚀 Quick Start

See the instructions in `math` folder.

---

## 📝 Citation

If you use this repo, please cite:

```bibtex
@inproceedings{
  pan2025scalebio,
  title={Scalebio: Scalable bilevel optimization for llm data reweighting},
  author={Pan, Rui and Zhang, Dylan and Zhang, Hanning and Pan, Xingyuan and Xu, Minrui and Zhang, Jipeng and Pi, Renjie and Wang, Xiaoyu and Zhang, Tong},
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2025",
  url = "https://aclanthology.org/2025.acl-long.1543/",
}

@inproceedings{
  pan2024lisa,
  title={{LISA}: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning},
  author={Rui Pan and Xiang Liu and Shizhe Diao and Renjie Pi and Jipeng Zhang and Chi Han and Tong Zhang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=L8ifDX5XNq}
}
```

---


## 🧩 FAQ

**Q:** Do I need second-order derivatives for bilevel training?
**A:** No—ScaleBiO uses a first-order bilevel paradigm, which is what enables scaling to large LLMs.

**Q:** Can I run on fewer than 8× H100?
**A:** Yes, but you’ll likely need to shrink the model/sequence length or enable more aggressive memory-saving knobs; throughput and final quality may differ from the paper’s large-scale setting.

---
