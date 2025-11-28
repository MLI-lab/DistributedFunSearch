# ZIH Capella Setup

## 1. Create Python Environment (once, on login node)

```bash
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

python3 -m venv --system-site-packages /data/horse/ws/frwe188h-disfun/venv
source /data/horse/ws/frwe188h-disfun/venv/bin/activate

pip install --upgrade pip setuptools
cd /home/frwe188h/DistributedFunSearch
pip install -e .

wandb login
```

## 2. Install RabbitMQ + Erlang (once)

RabbitMQ is not available as a module on ZIH, so we install it locally:

```bash
cd /data/horse/ws/frwe188h-disfun

# Install Erlang via conda
module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda create --prefix /data/horse/ws/frwe188h-disfun/erlang erlang -c conda-forge -y

# Download RabbitMQ
wget https://github.com/rabbitmq/rabbitmq-server/releases/download/v3.13.3/rabbitmq-server-generic-unix-3.13.3.tar.xz
tar -xf rabbitmq-server-generic-unix-3.13.3.tar.xz
rm rabbitmq-server-generic-unix-3.13.3.tar.xz

# Test it works
export PATH="/data/horse/ws/frwe188h-disfun/erlang/bin:$PATH"
./rabbitmq_server-3.13.3/sbin/rabbitmq-server --help
```

## 3. Submit Job

```bash
sbatch exp1_zih.sh
```

## 4. Access RabbitMQ Management UI

The script creates a reverse tunnel from the compute node to the login node on port 15672.

**From your local machine (while on VPN):**

```bash
# Create tunnel from local machine to ZIH login node
ssh -L 15672:localhost:15672 frwe188h@login1.capella.hpc.tu-dresden.de
```

Then open http://localhost:15672 in your browser.
- Username: `guest`
- Password: `guest`

## Useful Commands

```bash
# Check job status
squeue -u frwe188h

# Check estimated start time
squeue -u frwe188h --start

# Cancel job
scancel <job_id>

# View live output
tail -f /home/frwe188h/DistributedFunSearch/src/experiments/experiment1/logs/experiment_*.out
```
