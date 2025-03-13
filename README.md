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
___
## **Installation & Setup**

To set up and run FunSearch, follow the instructions based on your preferred execution method.

### **1. Clone the Repository**

Clone the FunSearch repository and navigate into the project directory:

```sh
git clone https://github.com/your-username/funsearch.git
cd Funsearch
```

### **2. Choose an Execution Method**

FunSearch can be run in different environments, with or without GPU/API-based LLM inference:

- **Docker Container** – (Containerized isolated execution)
- **Local Execution** – (Without Docker)
---

### **3. Execution with Docker**

FunSearch uses **Docker Compose (v3.8)** to run two containers:

- `funsearch-main` (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`) – Runs evolutionary search with GPU support.
- `rabbitmq` (`rabbitmq:3.13.4-management`) – Handles message passing.

You can navigate to the `.devcontainer` directory to start the containers:

```sh
cd .devcontainer
docker-compose up --build -d
```
Both containers run inside a **Docker bridge network** (`app-network`). 

- **Internal communication** – The main container connects to RabbitMQ via `rabbitmq:5672` (instead of `localhost`). The hostname in `/src/experiments/experimentX/config.py` is set to match this configuration by default.
- **External access** – The RabbitMQ Management Interface is a web-based dashboard that allows you to monitor message load, processing rates, and system status across components.  

  The interface is enabled by default in Docker execution and is available at:
  - **Web UI:** [http://localhost:15672](http://localhost:15672)
  - **Login Credentials (default):** `guest / guest`

  If running on a remote server, the Management UI is not directly accessible from your local machine. To access it on your local machine, forward port 15672 (default for management) using an SSH tunnel.  
  Run the following command on your local machine:  
  ```sh
  ssh -J <jump-user>@<jump-server> -L 15672:localhost:15672 <username>@<remote-server> -N -f
  ```

You can modify `docker-compose.yml` to change ports.

#### **3.1. Create and Activate a New Conda Environment (inside Docker)**

We recommend creating a clean Conda environment:

```sh
# Ensure conda is initialized for your shell (needed inside Docker)
conda init bash
source ~/.bashrc 
# Create and activate the Conda environment
conda create -n funsearch_env python=3.11 pip numpy==1.26.4 -y
conda activate funsearch_env
```

#### **3.2. Install PyTorch (inside Docker) *(_Can be skipped if using LLM inference over API_)***

Install PyTorch (matching CUDA version `12.1` used by `funsearch-main` container):

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **3.3. Install FunSearch package (inside Docker)**

Finally, you can install FunSearch with:

```sh
pip install .
```
---
### **4. Execution without Docker**

If you prefer to run FunSearch without Docker, follow these steps:

#### **4.1. Create a Conda Environment**

Create a clean Conda environment:

```sh
conda create -n funsearch_env python=3.11 pip numpy==1.26.4 -y
conda activate funsearch_env
```

#### **4.2. PyTorch Installation Matching CUDA** *(_Can be skipped if using LLM inference over API_)*

You can check your installed CUDA version using `nvidia-smi` and can find compatible PyTorch versions [here](https://pytorch.org/get-started/previous-versions/). For example, to install PyTorch for CUDA `12.1`, use: 

```sh
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

#### **4.4. Start RabbitMQ Service (Root Access Required)**

RabbitMQ must be started before running FunSearch. If RabbitMQ is **not installed** yet, install it using:

```sh
sudo apt update && sudo apt install rabbitmq-server -y
```

After installation, RabbitMQ **automatically starts as a system service**. To check its status:

```sh
sudo systemctl status rabbitmq-server
```

If RabbitMQ is already installed but not running, start it with:

```sh
sudo systemctl start rabbitmq-server
```

To connect FunSearch to RabbitMQ when running **without Docker**, set the RabbitMQ host in `/src/experiments/experimentX/config.py` to:

```sh 
host: str = 'localhost'
```

#### **Optional: Enable the Management Interface (Monitor Load and Processing Rates)**
The RabbitMQ **Management Interface** provides a web-based dashboard for monitoring message load, processing rates, and system status across components. Enable it with:

```sh
sudo rabbitmq-plugins enable rabbitmq_management
sudo systemctl restart rabbitmq-server
```

If running **locally**, you can now access the **Management Interface** at:

- **Web UI:** [http://localhost:15672](http://localhost:15672)
- **Login Credentials (default):** `guest / guest`


If running on a remote server, the Management UI is not directly accessible from your local machine. To access it on your local machine, forward port 15672 (default for management) using an SSH tunnel.  
Run the following command on your local machine:
```sh
ssh -J <jump-user>@<jump-server> -L 15672:localhost:15672 <username>@<remote-server> -N -f
```

#### **4.5. Install FunSearch package**

Finally, install FunSearch:

```sh
pip install .
```

___
## **Usage**
To start an evolutionary search experiment, navigate to your experiment directory (e.g., `experiments/experimentX/`) and run:

```bash
python funsearch.py --config-path experiments/experimentX/config.py 
```

This launches a search using the configurations specified in the directory's `config.py` file, which contains explanations for each argument.

---

## **Command-Line Arguments**
You can specify **general settings, resource management, and termination criteria** via command-line arguments:

#### **General Settings**
- `--config-path /path/to/config`  
  - Path to the configuration file.  
  - Default: `config.py` (inside the directory where the script is run).

- `--save_checkpoints_path /path/to/checkpoints`  
  - Path where checkpoints should be saved.  
  - Default: `Checkpoints/` (inside the directory where the script is run).

- `--checkpoint /path/to/checkpoint`  
  - Path to a checkpoint file from which the search should continue.  
  - Default: `None`.

- `--log-dir /path/to/logs`  
  - Directory where logs will be stored.  
  - Default: `logs/` (inside the directory where the script is run).

#### **Resource Management**
- `--no-dynamic-scaling`  
  - Disables dynamic scaling of evaluators and samplers based on message load.  
  - Default: enabled.

- `--check_interval 120`  
  - Sets the interval (in seconds) for checking resource allocation when dynamic scaling is enabled.  
  - Default: `120s`.

- `--max_evaluators 1000`  and `--max_samplers 1000`  
  - Define the maximum number of evaluators and samplers that can be created dynamically.  
  - Default: a large value, allowing scaling based on resource utilization without hard limits.

#### **Termination Criteria**
- `--prompt_limit 400000`  
  - Sets the maximum number of prompts that can be published.  
  - If set, no new prompts will be published once this limit is reached. However, any remaining messages in the queues will still be processed to ensure exactly `prompt_limit` functions are handled.

- `--optimal_solution_programs 20000`  
  - Defines the number of additional programs to generate after the first optimal solution is found.  
  - If set, execution continues until this number is reached.

- `--target_solutions '{"(6,1)": 8, "(7,1)": 14, "(8,1)": 25}'`  
  - JSON dictionary specifying target solutions for `(n, s_value)`.  
  - If set, the experiment terminates early once a target solution is found.

---
## **Running Multiple Experiments in Parallel**
If you want to run multiple experiments in parallel, you must **assign different RabbitMQ ports**.  
Update both the **TCP listener port** and the **management interface port** in `rabbitmq.conf`.  
Then, update the corresponding ports in your experiment config file (`config.py`) to match the new RabbitMQ settings.
