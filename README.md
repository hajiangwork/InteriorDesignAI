# Introduction 
An Interior design AI demo by stable diffusion and ControlNet.

Reference
- [ControlNet Github link](https://github.com/lllyasviel/ControlNet)
- [HuggingFace ControlNet examples](https://huggingface.co/docs/diffusers/v0.16.0/en/api/pipelines/stable_diffusion/controlnet#diffusers.StableDiffusionControlNetPipeline)

# Getting Started
## Installation
```
conda env create -f environment.yaml
conda activate control
```
## Run
Replace `<azure openai endpoint>` and `<azure openai key>` in `chatgpt.py`, and then
```
python main.py
```
This will deploy an web UI on local machine. By default the address will be http://0.0.0.0:7860/

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)