<div align="center">
<p align="center">
  <h1>Is Context the Bottleneck? Causal Decomposition of Failure in Long-Horizon LLM Agents</h1>
  
  [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/)
</p>
</div>

# Overview
Long-horizon LLM agents degrade as trajectory length increases, but is the bottleneck context quality or reasoning capability? This work introduces an oracle state intervention protocol that causally decomposes agent failures.

# Environmental Setup

**1. Create a new conda environment**

```bash
conda create -n agent-decomp python=3.11 -y
conda activate agent-decomp
```

**2. Clone the repository**

```bash
git clone https://github.com/leqijia/agent-decomp.git
cd agent-decomp
pip install -r requirements.txt
```