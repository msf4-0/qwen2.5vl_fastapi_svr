"""
Date: 11/06/2025
Author: Reuben Lim
Desc: To install dependencies for running Qwen2.5VL with OpenVINO. Also installs and convert the model using Optimum Intel.
      Based on the OpenVINO Notebook: https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen2.5-vl/qwen2.5-vl.ipynb
NOTE: Best to run under Python venv
      """

import subprocess
import platform
from pathlib import Path
subprocess.call(["pip", "install", "requests"])
import requests

"""Install dependencies"""
subprocess.call(["pip", "install", "torch>=2.1", "torchvision", "qwen-vl-utils", "Pillow", "gradio>=4.36", "--extra-index-url", "https://download.pytorch.org/whl/cpu"])
subprocess.call(["pip", "install", "-U", "openvino>=2025.0.0", "openvino-tokenizers>=2025.0.0", "nncf>=2.15.0"])
subprocess.call(["pip", "install", "git+https://github.com/huggingface/optimum-intel.git", "--extra-index-url", "https://download.pytorch.org/whl/cpu"])
subprocess.call(["pip", "install", "transformers>=4.49"])
subprocess.call(["pip", "install", "fastapi", "uvicorn"])
if platform.system() == "Darwin":
    subprocess.call(["pip", "install", "numpy<2.0"])

# if not Path("cmd_helper.py").exists():
#     r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
#     open("cmd_helper.py", "w").write(r.text)


"""Convert and Optimize Model for OpenVINO"""
"""NOTE: CHOOSE MODEL HERE"""
pt_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model_dir = Path(pt_model_id.split("/")[-1])

from cmd_helper import optimum_cli

if not (model_dir / "INT4").exists():
    optimum_cli(pt_model_id, model_dir / "INT4", additional_args={"weight-format": "int4"})

