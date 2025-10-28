# AI-Facade
Human-factor-driven automated façade generation and optimization using multimodal physiological data and multi-agent deep reinforcement learning for intelligent geometry and material design.
![Uploading Fig1.jpg…]()

---
# 🏙️ AI-Facade

**Human-factor-driven Automated Façade Design and Optimization**

## 🔍 Project Overview

**AI-Facade** is a research project that explores how physiological data can inform automated architectural façade design.
It establishes a closed-loop workflow linking **human-factor evidence**, **computational optimization**, and **parametric generation**.
By translating multimodal physiological signals—such as EEG—into computable design objectives, the project leverages **multi-agent deep reinforcement learning (MADRL)** to optimize façade geometry and materials.

The system integrates:

* Reinforcement learning using **PaddlePaddle-PARL** framework
* Human-factor metrics for adaptive façade optimization

This framework enables real-time, data-driven façade generation guided by human cognitive responses, bridging the gap between design performance and human perception.

---

## ⚙️ Runtime Environment

* **Python Version:** 3.8
* **Core Dependencies:**

  ```bash
  parl==2.2.1
  paddlepaddle==2.5.1
  pandas==1.1.5
  numpy==1.19.5
  gym==0.26.2
  matplotlib==3.3.4
  ```

---

## 📦 Installation & Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/AI-Facade.git
   cd AI-Facade
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # For macOS/Linux
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download PARL example files**
   Download the official [PARL examples](https://github.com/PaddlePaddle/PARL/tree/develop/examples) and place them under:

   ```
   PARL/examples/
   ```

---

## 🚀 Quick Start

After setting up, you can run training or evaluation scripts, for example:

```bash
python run_all.py
python Test.py --restore --max_episodes=1000
```

---


