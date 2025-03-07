## LLM-Guided Search for Deletion-Correcting Codes

<div align="center">
  <img src="fig/funsearch_overview.png" alt="FunSearch Overview" width="600">
</div>

<p>&nbsp;</p>

This repository provides a **distributed implementation of FunSearch** (Romera et al., 2024) using RabbitMQ for parallelization via asynchronous message passing. The code accompanies the paper *"LLM-Guided Search for Deletion-Correcting Codes"* and is designed for discovering large deletion-correcting codes for any code length and deletion correction capacity.

FunSearch iteratively refines a **priority function** using **evolutionary search** guided by a pretrained **LLM** (default: Starcoder2, with support for GPT-4o Mini via API). 

In each iteration:
- A few-shot prompt is constructed by sampling from the program database.
- The LLM generates a new priority function.
- The function is evaluated by greedily constructing deletion-correcting codes for various code lengths, with a fixed or variable number of deletions.
- If the function is executable and unique, it is stored in the database.

### Modifications for Other Applications
FunSearch can be adapted to different applications with minimal changes:
- **Input format & specification script:** Modify these to adjust the application-specific input format and evaluation logic.
- **LLM model:** You can modify the `checkpoint` parameter in the sampler script to use any open-source LLM that can be loaded from Hugging Face via `transformers.AutoModelForCausalLM`.
---

## **Installation & Setup**

To set up and run **FunSearch**, follow the instructions based on your preferred execution method.

### **1. Clone the Repository**

Clone the FunSearch repository and navigate into the project directory:

```sh
git clone https://github.com/your-username/funsearch.git
cd Funsearch
```

### **2. Choose an Execution Method**

FunSearch can be run in different environments, with or without GPU/API-based LLM inference:

- **Docker Container** – (Recommended for reproducibility and isolation)
- **Local Execution** – (Without Docker)
- **SLURM with Enroot** – (For cluster-based execution)

### **1. Clone the Repository**

Clone the FunSearch repository and navigate into the project directory:

```sh
git clone https://github.com/your-username/funsearch.git
cd Funsearch
```

### **2. Choose an Execution Method**

FunSearch can be run in different environments, with or without GPU/API-based LLM inference:

- **Docker Container** – (Recommended for reproducibility and isolation)
- **Local Execution** – (Without Docker)
- **SLURM with Enroot** – (For cluster-based execution)

### **3. Execution with Docker**

FunSearch uses **Docker Compose (v3.8)** to run two containers:

- **`funsearch-main` (`pytorch/pytorch:2.2.2-cuda11.8-cudnn9-runtime`)** – Runs PyTorch execution tasks with GPU support.
- **`rabbitmq` (`rabbitmq:3.13.4-management`)** – Handles message passing.

You can navigate to the `.devcontainer` directory to start the containers:

```sh
cd .devcontainer
docker-compose up --build -d
```

#### **Docker Networking**
Both containers run inside a **Docker bridge network** (`app-network`) for internal communication:

- **Internal communication** – The main container connects to RabbitMQ via `rabbitmq:5672` (instead of `localhost`).
- **External access** – RabbitMQ’s interface is available at:
  ```
  http://localhost:15672
  ```
  You can modify `docker-compose.yml` to change ports.

#### **3.1. Create and Activate a New Conda Environment (inside Docker)**

We recommend creating a clean Conda environment:

```sh
conda create -n funsearch_env python=3.11 pip numpy==1.26.4 -y
conda activate funsearch_env
```

#### **3.2. Install PyTorch matching the Docker CUDA version (inside Docker)**

Install PyTorch (matching CUDA version `11.8` used by the `funsearch-main` container):

```sh
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **3.3. Install FunSearch package (inside Docker)**

Finally, you can install FunSearch with:

```sh
pip install .
```
---

### **4. Running Locally (Without Docker)**

If you prefer to run FunSearch without Docker, follow these steps:

#### **1. Create a Conda Environment**

Create a clean Conda environment:

```sh
conda create -n funsearch_env python=3.11 pip numpy==1.26.4 -y
conda activate funsearch_env
```

#### **2. Install PyTorch (Match Your System's CUDA Version)**

Check your system CUDA version:

```sh
nvcc --version
```

You can find the compatible PyTorch versions [here](https://pytorch.org/get-started/previous-versions/).

Example installation command for CUDA `11.8`:

```sh
conda install pytorch==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### **3. Install FunSearch package**

Finally, install FunSearch:

```sh
pip install .
```

#### **4. Start RabbitMQ Service**

RabbitMQ must be started before running FunSearch. To start it manually:

```sh
rabbitmq-server -detached
```

You can check if RabbitMQ is running using the following command:

```sh
rabbitmqctl status
```

>>>>>>> bd8b388 (Reinitialized Git after restructuring)
## Command-line Arguments
1. **`--spec-path` (Required)**  
   - **Description**: Path to the specification file from which the prompt is build.  
   - **Usage**:  
     ```bash
     python funsearch.py --spec-path /path/to/specification.txt
     ```

2. **`--backup` (Optional)**  
   - **Description**: Enables backup of all Python files in the working directory before starting the task.  
   - **Usage**:  
     ```bash
     python your_script.py --backup
     ```
   - **Backup Location**:  
     Backups are saved in `/mnt/hdd_pool/userdata/franziska/code_backups` with a timestamped subdirectory.

3. **`--dynamic-scaling` (Optional)**  
   - **Description**: Enables dynamic scaling of evaluators and samplers based on system resources and message queue load.  
   - **Usage**:  
     ```bash
     python your_script.py --dynamic-scaling
     ```

---
## Dynamic Scaling Logic

The **dynamic scaling controller** adjusts the number of evaluator and sampler processes dynamically based on:

### CPU Load
- If the system's 5-minute load average exceeds the number of available CPU cores, processes are scaled down.
- **Evaluator processes** are terminated until:
  - The load drops below or equals the CPU core count.
  - The minimum number of evaluators (`min_evaluators`, default: 1) is reached.

### Queue Metrics

#### Evaluator Queue
- If the queue contains more messages than the `evaluator_threshold` (default: 5), additional evaluator processes are started **if more than 4 CPUs have less than 50% usage**.
- Evaluators are scaled down if:
  - The message count falls below the threshold.
  - A process with CPU usage below 20% is identified.

#### Sampler Queue
- Additional samplers are started if:
  - The queue contains more messages than the `sampler_threshold` (default: 15).
  - A GPU with the following conditions is available:
    - **At least 32 GiB of free memory.**
    - **Less than 50% utilization.**
- **Multiple GPUs Condition**:
  - If no single GPU meets the criteria, multiple GPUs are combined if their total free memory is at least **32 GiB**.
- Samplers are scaled down if:
  - The message count drops below the threshold.
  - A process with GPU utilization below 10% is identified.


