## LLM-Guided Search for Deletion-Correcting Codes

<div align="center">
  <img src="fig/overview.png" alt="FunSearch Overview" width="600">
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
Our implementation can be adapted to different applications with minimal changes:
- **Input format & specification script:** Modify these to adjust the application-specific input format and evaluation logic.
- **LLM:** You can modify the `checkpoint` parameter in the sampler script to use any open-source LLM that can be loaded from Hugging Face via `transformers.AutoModelForCausalLM`.
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

Our implementation is designed for **Linux** and tested on Ubuntu.  
You can execute it in different environments, with or without GPU/API-based LLM inference:

- **Docker Container** – (Containerized isolated execution)
- **Local Execution** – (Without Docker)
- **With Slurm and Enroot** – (Cluster-based execution)
---

### **3. Execution with Docker**

Our implementation uses **Docker Compose (v3.8)** to run two containers:

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

### **5. Execution with Slurm and Enroot**

To run FunSearch on a Slurm cluster using Enroot containers, follow these steps:

#### **5.1. Pull a PyTorch Enroot Image**  

Download and convert a PyTorch image with the required CUDA version into an Enroot image.  
For example, to install PyTorch 2.2.2 with CUDA 12.1:  

```sh
enroot import -o /desired/path/custom_name.sqsh docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

#### **5.2. Install RabbitMQ Inside the Enroot Image**  

Start the image with root privileges to install RabbitMQ, curl, and OpenSSH client:  
```sh
enroot create -n custom_name desired/path/custom_name.sqsh
enroot start --root --rw custom_name
apt update && apt install -y rabbitmq-server curl openssh-client
rabbitmq-plugins enable rabbitmq_management
```

Exit the Enroot container and save the changes in a new image `custom_name_with_rabbitmq` (after saving, you can delete the original `custom_name` image):
```sh
exit  
enroot export -o /dss/dsshome1/02/di38yur/Funsearch/custom_name_with_rabbitmq.sqsh custom_name
```

#### **5.2. Submit SLURM Job**  

Using the previously created Enroot image, you can submit your SLURM job with a job script (`.sh`).  
An example script, `/Funsearch/src/experiments/experiment1/exp1.sh`, supports multi-node execution and sets up an SSH reverse tunnel for local access to the RabbitMQ management interface.  

Submit the job with:  

```sh
sbatch /Funsearch/src/experiments/experiment1/exp1.sh
```
This starts an evolutionary search experiment as explained in the Usage section.



___
## **Usage**

To start an evolutionary search experiment, navigate to your experiment directory (e.g., `experiments/experimentX/`) and run:

```bash
python -m funsearch
```

This launches a search using the configurations specified in the directory's `config.py` file, which contains explanations for each argument.

**Note:** If stopping an experiment, check RabbitMQ to ensure all evaluator and sampler processes are shut down before starting a new one.



**(Optional) Preloading the Model**

Before running the evolutionary search, you can download the model from Hugging Face and store it in a cache location by executing:

```bash
python load_llm.py
```

By default, the model will be stored in the `models/` directory inside your current working directory.

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
  - Directory where function executions are sandboxed. Stores input/output data, serialized function files, and error logs. By default, outputs are deleted after execution to prevent excessive memory usage, as each function execution generates its own stored output.
  - Default: `sandbox/` (inside the directory where the script is run).

- `--sandbox_base_path /path/to/sandbox_directory`  
  - Directory where  will be stored.  
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
___
## **Scaling FunSearch Across Multiple Nodes**

Our implementation supports distributed execution by attaching **evaluator** and **sampler** processes to a running script for:

- **Multi-node execution** to increase the rate at which new priority functions are processed (generated, evaluated, and stored).
- **Dynamic scaling** to balance message load at runtime.

### **Attaching Additional Processes**

You can run the following commands to attach more evaluators and samplers:
```sh
python -m funsearch.attach_evaluators
python -m funsearch.attach_samplers
```

These scripts use the same **command-line arguments** as the main script and can be run in the same execution modes described in the **Installation & Setup** section, with the exception that **RabbitMQ should not be restarted** when attaching additional processes.

#### **Local Execution**

- You can follow the **Execution Without Docker** steps, skipping the RabbitMQ startup and running the attach scripts instead.

#### **Docker Execution**

- You can start only the `funsearch-main` container (without launching a new RabbitMQ instance) by running:
  ```sh
  cd .devcontainer/external/.devcontainer  
  docker-compose up  
  ```
  This starts a `funsearch-main` container on the new node for running the attach scripts.

#### **SLURM & Enroot Execution**

- For an example of multi-node SLURM execution, see:
  ```sh
  /Funsearch/src/experiments/experiment1/exp1.sh
  ```

### **Configuring RabbitMQ for Multi-Node Execution**

To attach processes from a different node, the new node must be able to connect to the main node running RabbitMQ.

If the nodes can resolve each other’s IP addresses:

- You can set `host` in `config.py` to the **IP address of the main node** (where RabbitMQ runs).

If the nodes **cannot** resolve each other’s IP addresses:

- On the new node, you can establish an SSH tunnel to forward RabbitMQ’s TCP listener port (default: 5672):

  ```sh
  ssh -J <jump-user>@<jump-server> -L 5672:localhost:5672 <username>@<remote-server> -N -f
  ```

- You can then set `host: 'localhost'` on the new node.

## **Running Multiple Experiments in Parallel With SLURM**
If you want to run multiple experiments in parallel, you need to **assign different RabbitMQ ports**.  
You can update both the **TCP listener port** and the **management interface port** in `rabbitmq.conf`.  
Then, update the corresponding ports in your experiment config file (`config.py`) to match the new RabbitMQ settings.



