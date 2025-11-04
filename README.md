## LLM-Guided Search for Deletion-Correcting Codes

<div align="center">
  <img src="fig/overview.png" alt="DeCoSearch Overview" width="600">
</div>

<p>&nbsp;</p>

**DeCoSearch** (Deletion-Correcting Code Search) is a **distributed implementation of FunSearch** (Romera et al., 2024), designed to discover large deletion-correcting codes for an adversarial number of deletions and finite code lengths. It uses RabbitMQ for parallelization via asynchronous message passing and is guided by a pretrained LLM, defaulting to StarCoder2, with support for OpenAI models via API (e.g., GPT-4o Mini via Azure OpenAI).

In each iteration:

- A few-shot prompt is constructed by sampling from the program database, which stores all previously generated functions and their metadata.
- The LLM generates a new priority function.
- The function is evaluated by greedily constructing deletion-correcting codes for user-defined code lengths and number of adversarial deletions.
- If the function is executable and logically distinct from previously stored ones, it is added to the program database along with its evaluation results (i.e., the code sizes achieved).

For more details, see [our paper](https://arxiv.org/abs/2504.00613).


### Modifications for other applications

Our implementation can be adapted to different applications with minimal changes:

- **Input format and evaluation logic:** You can modify the input format of the function to be evolved in `src/experiments/experiment1/config.py` (via the `EvaluatorConfig` class), and optionally set a performance threshold using the `--target_solution` argument in `src/decos/__main__.py` (e.g., to terminate the search once a function surpasses the current best-known solution).  
To adapt how functions are evaluated for your specific application, you can modify the logic in the `src/decos/specifications/` folder.

- **LLM:** You can modify the `checkpoint` parameter in the sampler script (`src/decos/sampler.py`) to use a different open-source LLM that can be loaded from Hugging Face via `transformers.AutoModelForCausalLM`.

___
## **Installation & setup**

To set up and run DeCoSearch, follow the instructions based on your preferred execution environment.

### **1. Clone the repository**

Clone the DeCoSearch repository:

```sh
git clone https://github.com/MLI-lab/DeCoSearch.git
cd DeCoSearch
```

### **2. Choose an execution environment**

Our implementation is designed for Linux and tested on Ubuntu 22.04.6 LTS.
You can execute DeCoSearch in different environments, with or without GPU/API-based LLM inference:

- **Docker Container** – (Containerized isolated execution)
- **Local Execution** – (Without Docker)
- **With Slurm and Enroot** – (Cluster-based execution)
---

### **3. Execution with Docker**

Our implementation uses **Docker Compose (v3.8)** to run two containers:

- `decos-main` (`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`) – Runs the evolutionary search with GPU support.
- `rabbitmq` (`rabbitmq:3.13.4-management`) – Handles message passing.

You can navigate to the `.devcontainer` directory to start the two containers:

```sh
cd .devcontainer
docker compose up --build -d
```
Both containers run inside a **Docker bridge network** (`app-network`). 

- **Internal communication** – The main container `decos-main` connects to RabbitMQ via `rabbitmq:5672`. The hostname in `/src/experiments/experiment1/config.py` is set to match this configuration by default.
- **External access** – The RabbitMQ Management Interface is a web-based dashboard that allows you to monitor message load, processing rates, and system status across system components.  

  The interface is enabled by default in Docker execution and is available at:
  - **Web UI:** [http://localhost:15672](http://localhost:15672)
  - **Login Credentials (default):** `guest / guest`

  If running on a remote server, the Management UI is not directly accessible from your local machine. To access it on your local machine, you can forward port 15672 (default for management) using an SSH tunnel.  
  Run the following command on your local machine:  
  ```sh
  ssh -J <jump-user>@<jump-server> -L 15672:localhost:15672 <username>@<remote-server> -N -f
  ```

To change ports or hostnames, you can modify `docker-compose.yml`.

#### **3.1. Create and activate a new conda environment (inside docker container decos-main)**

We recommend creating a clean Conda environment:

```sh
# Ensure conda is initialized for your shell (needed inside Docker)
conda init bash
source ~/.bashrc 
# Create and activate the Conda environment
conda create -n decos_env python=3.11 pip numpy==1.26.4 -y
conda activate decos_env
```

#### **3.2. Install PyTorch (inside Docker) *(_Can be skipped if using LLM inference over API_)***

You can install PyTorch (matching CUDA `12.1` used by the `decos-main` Docker image) with the following command:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **3.3. Install DeCoSearch package (inside Docker)**

Finally, you can install DeCoSearch with:

```sh
pip install .
```
---
### **4. Execution without Docker**

If you prefer to run DeCoSearch without Docker, follow these steps:

#### **4.1. Create a conda environment**

Create a clean Conda environment:

```sh
conda create -n decos_env python=3.11 pip numpy==1.26.4 -y
conda activate decos_env
```

#### **4.2. PyTorch installation matching CUDA** *(_Can be skipped if using LLM inference over API_)*

You can check your installed CUDA version using `nvidia-smi` and can find compatible PyTorch versions [here](https://pytorch.org/get-started/previous-versions/). For example, to install PyTorch for CUDA `12.1`, use: 

```sh
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

#### **4.4. Start RabbitMQ service (root access required)**

RabbitMQ must be started before running DeCoSearch. If RabbitMQ is **not installed**, you can install it using:

```sh
sudo apt update && sudo apt install -y rabbitmq-server
```

After installation, RabbitMQ **automatically starts as a system service**. To check its status:

```sh
sudo systemctl status rabbitmq-server
```

If RabbitMQ is already installed but not running, start it with:

```sh
sudo systemctl start rabbitmq-server
```

To connect DeCoSearch to RabbitMQ when running **without Docker**, set the RabbitMQ host in `/src/experiments/experimentX/config.py` to:

```sh 
host: str = 'localhost'
```

#### **Optional: Enable the management interface (for monitoring load and processing rates)**
The RabbitMQ **Management Interface** provides a web-based dashboard for monitoring message load, processing rates, and system status across components. You can enable it with:

```sh
sudo rabbitmq-plugins enable rabbitmq_management
sudo systemctl restart rabbitmq-server
```
For local and remote access instructions, see Execution with Docker.

#### **4.5. Install DeCoSearch package**

Finally, install DeCoSearch:

```sh
 pip install . 
```

### **5. Execution with slurm and enroot**

To run DeCoSearch on a Slurm cluster using Enroot containers, follow these steps:

#### **5.1. Pull a PyTorch enroot image**  

You can download and convert a PyTorch image with the required CUDA version to an enroot image.  
For example, to install PyTorch 2.2.2 with CUDA 12.1:  

```sh
enroot import -o /desired/path/custom_name.sqsh docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

#### **5.2. Install RabbitMQ inside the enroot image**  

You can start the image with root privileges to install RabbitMQ, curl, and OpenSSH client:  
```sh
enroot create -n custom_name desired/path/custom_name.sqsh
enroot start --root --rw custom_name
apt update && apt install -y rabbitmq-server curl openssh-client
rabbitmq-plugins enable rabbitmq_management
```

Once the setup is complete, you can exit the Enroot container and save the changes in a new image `custom_name_with_rabbitmq` (after saving, you can delete the original `custom_name` image):
```sh
exit  
enroot export -o desired/path/custom_name_with_rabbitmq.sqsh custom_name
```

#### **5.2. Submit SLURM job**  

You can submit your SLURM job using the previously created Enroot image and a job script (`.sh`).  
For an example of multi-node execution, see `src/experiments/experiment1/exp1.sh`. This script also sets up an SSH reverse tunnel for local access to the RabbitMQ management interface.  

___
## **Usage**

To start an evolutionary search experiment, navigate to your experiment directory (e.g., `src/experiments/experiment1/`) and run:

```bash
cd src/experiments/experiment1
python -m decos
```

This launches a search using the configurations specified in the directory's `config.py` file. The file includes explanations for each argument.

The number of GPUs used is controlled by the `num_samplers` parameter, each sampler runs on a separate GPU.
The number of CPUs used is determined by the `num_evaluators` parameter, each evaluator runs two parallel CPU processes to evaluate generated functions on different inputs.

You can monitor the messages passed between components through the RabbitMQ Management Interface.

**Note:** If stopping an experiment (e.g. via Ctrl+C), shutdown can take 30–60 seconds. During this time, evaluator and sampler processes clean up their connections to RabbitMQ and close all open resources.


Before starting a new run, ensure that all old processes have fully shut down. To close any remaining processes, you can restart the RabbitMQ container. For local execution, you can restart the RabbitMQ service using `sudo systemctl restart rabbitmq-server`. 
If you are using Azure OpenAI (i.e., `gpt=True` in the `SamplerConfig` class), make sure to export the following environment variables before running:

```bash
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_API_VERSION=<your-version>  # e.g., 2024-08-01-preview
```

**(Optional) Downloading the LLM**

Before running the evolutionary search, you can download the LLM from Hugging Face and store it in a cache location. To download StarCoder2, run:

```bash
python load_llm.py
```

The model will be cached in the `/workspace/models/` directory, for both download in advance or when the script first runs.

To change the cache location, you can modify the `TRANSFORMERS_CACHE` environment variable in `src/decos/sampler.py`.

---

## **Command-line arguments**
You can specify **general settings, resource management, and termination criteria** via command-line arguments:

#### **General settings**
- `--config-path /path/to/config`  
  - Path to the configuration file.  
  - Default: `config.py` (inside the directory where the script is run).

- `--save_checkpoints_path /path/to/checkpoints`  
  - Path where checkpoints should be saved.  
  - Default: `Checkpoints/` (inside the directory where the script is run).

- `--checkpoint /path/to/checkpoint`  
  - Path to a checkpoint file from which the search should continue.  
  - Default: `None`.

- `--sandbox_base_path /path/to/sandbox_directory`
  - Directory where function executions are sandboxed. Stores input/output data, serialized function files, and error logs. By default, outputs are deleted after execution to prevent excessive memory usage, as each function execution generates its own stored output.
  - Default: `sandbox/` (inside the directory where the script is run).

- `--log-dir /path/to/logs` 
  - Directory where logs will be stored.  
  - Default: `logs/` (inside the directory where the script is run).

#### **Resource management**
- `--no-dynamic-scaling`  
  - Disables dynamic scaling of evaluators and samplers based on message load.  
  - Default: enabled.

- `--check_interval`  
  - Sets the interval (in seconds) for checking resource allocation when dynamic scaling is enabled.  
  - Default: `120s`.

- `--max_evaluators`  and `--max_samplers`  
  - Sets the maximum number of evaluators and samplers that can be created dynamically.
  - Default: large value (`1000`), allowing scaling based on resource utilization without hard limits.

#### **Termination criteria**
- `--prompt_limit`  
  - Maximum number of prompts that can be published. Once reached, no new prompts are constructed,  but queued messages are still processed.
  - Default: `400_000`

- `--optimal_solution_programs `  
  - Sets the number of additional programs to generate after the first optimal solution is found.  
  - Default: `20_000`

- `--target_solutions '{"(6,1)": 8, "(7,1)": 14, "(8,1)": 25}'`  
  - JSON dictionary specifying target solutions for code length `n` and deletion correction `s` as `(n, s)`.  
  - If set, the experiment terminates early upon finding a target solution.  

___
## **Scaling DeCoSearch across multiple nodes**

Our implementation supports distributed execution by attaching **evaluator** and **sampler** processes to a running script for:

- **Multi-node execution** to increase the rate at which new priority functions are processed (generated, evaluated, and stored).
- **Dynamic scaling** to balance message load at runtime.

### **Attaching additional processes**

You can run the following commands to attach more evaluators and samplers:
```sh
python -m decos.attach_evaluators
python -m decos.attach_samplers
```

These scripts use the same **command-line arguments** as the main script and can be run in the same execution modes, with the difference that **RabbitMQ should not be restarted** if additional processes are attached.

#### **Local execution**

- You can follow the **Execution without docker** steps, skipping the RabbitMQ startup and running the attach scripts instead of the main script (`decos`).

#### **Docker execution**

- You can start only the `decos-main` container (without launching a new RabbitMQ instance) by running:
  ```sh
  cd DeCoSearch/.devcontainer/external/.devcontainer  
  docker-compose up  
  ```
  This starts a `decos-main` container on the new node for running the attach scripts.

#### **SLURM & enroot execution**

- For an example of multi-node SLURM execution, see:
  ```sh
  /DeCoSearch/src/experiments/experiment1/exp1.sh
  ```

### **Configuring RabbitMQ for multi-node execution**

To attach processes from a different node, the new node must be able to connect to the main node running RabbitMQ.

If the nodes can resolve each other’s IP addresses and are in the same network without firewall restrictions (e.g., on a cluster):

- You can set `host` in `config.py` to the **hostname of the main node** (where RabbitMQ runs).

If the nodes **cannot** resolve each other’s IP addresses:

- On the new node, you can establish an SSH tunnel to forward RabbitMQ’s TCP listener port (default: 5672):

  ```sh
  ssh -J <jump-user>@<jump-server> -L 5672:localhost:5672 <username>@<remote-server> -N -f
  ```

- You can then set `host: 'localhost'` on the new node.

## **Running multiple experiments in parallel with SLURM**
If you want to run multiple experiments in parallel, you need to **assign different RabbitMQ ports**.  
You can update both the **TCP listener port** and the **management interface port** in `rabbitmq.conf`.  
Then, update the corresponding ports in your experiment config file (`src/experiments/experiment1/config.py`) to match the new RabbitMQ settings.



