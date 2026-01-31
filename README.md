# Self-Guard: Defending Large Reasoning Models via enhanced self-reflection
![framework](workflow.png)

Self-Guard, a lightweight safety defense framework that reinforces safety compliance at the representational level for Large Reasoning Models (LRMs). Self-Guard operates through two principal stages: safety-oriented prompting and safety activation steering. During the first stage, it activates the model's latent safety awareness to evoke spontaneous reflection on potential risks. In the second stage, it extracts the directional shift in the hidden state space and amplifies this safety signal to ensure compliance prevails over sycophancy during inference. Specifically, Self-Guard offers a cost-efficient and robust solution that generalizes across diverse unseen risks without compromising model utility.

## Quick Start

### Step 1. Build Environment

```bash
conda create -n self-guard python=3.11
conda activate self-guard
pip install -r requirements.txt
```

### Step 2. Usage

The run.sh script automates the process of extracting steering vector and generating steered responses for the Qwen/Qwen3-8B model.

> **Note:** To accelerate the steering inference process, we use the `vllm` library for speedup.  
> This requires manually modifying a few files in the `vllm` source code; the corresponding code changes are provided in `./steering/vllm_for_steering`.

```bash
bash ./scripts/run.sh
```
