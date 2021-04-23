## Introduction

In Azure ML studio we can create and manage machine learning assets very easily, it is often advantageous to use a code-based approach to managing resources.

Writing scripts to create and manage resources can help run machine learning tasks from preferred development vironment, automate asset creation and configuration, easure consistency for resources as well as incorporate machine learning asset configuration into developer operations workflows(DevOps).

Azure Machine Learning provides software development kits(SDKs) for Python and R.

## 1. Installing SDK for Python

Open your Azure portal, create or use a machine learning resource, launch the studio. Under the Manage tab, click Compute and find or create a compute instance, open the Jupyter platform. 

![image](https://user-images.githubusercontent.com/71245576/115772882-a508f800-a37d-11eb-84bd-0585407c8597.png)

In Jupyter, create a notebook, the version I chose is Python 3.6-AzureML, that is because the version of tutorial may not support the latest kernel version. If you want to use the latest version Python 3.8-AzureML, see Azure Machine Learning release notes, the URL is https://docs.microsoft.com/en-us/azure/machine-learning/azure-machine-learning-release-notes.

Now in your workspace install the Azure Machine Learning SDK for Python by using the pip package management utility. 
```python
pip install azureml-sdk
```
It took a few seconds and will notice you that you may need to restart the kernel to use updated packages. 

By the way, the SDK includes optional extras that are not required for core operations but can be useful in some scenarios. For example, the notebooks extra include widgets for displaying detailed output in Jupyter Notebooks, the AutoML extra includes packages for automated machine learning training, and the explain extra includes packages for generating model explanations. 

To install extras, specify them in brackets as shown below:

```python
pip install azureml-sdk[notebooks,automl,explain]
```

![image](https://user-images.githubusercontent.com/71245576/115774458-97ed0880-a37f-11eb-9c55-6e590c3cf127.png)

## 2. Connecting to a Workspace

After installing the SDK package in the Python environment we can write code to connect to your workspace and perform machine learning operations. We need to download a configuration file for a workspace from the Azure Machine Learning studio or the Overview page of its blade in the Azure portal.

See how to download config.json in Azure Machine Learning studio:

![image](https://user-images.githubusercontent.com/71245576/115775148-7dfff580-a380-11eb-9590-92b537b9b2a6.png)

To connect to the workspace using the configuration file, we can use the from_config method of the Workspace class in the SDK, as shown here:

```python
from azureml.core import Workspace
ws = Workspace.from_config()
```

Or you can use the get method of the Workspace class explicitly specified subscription, resource group, and workspace details as shown here:
```python
from azureml.core import Workspace

ws = Workspace.get(name='aml-workspace',
                   subscription_id='1234567-abcde-890-fgh...',
                   resource_group='aml-resources')
```
## 3. Working with the Workspace Class
The Workspace class is the starting point for most code operations. For example, you can use its compute_targets attribute to retrieve a dictionary object containing the compute targets defined in the workspace, like this:
```python
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)
```

See some of my compute targets infomation:

![image](https://user-images.githubusercontent.com/71245576/115776169-d4216880-a381-11eb-9c0e-922b9914e2b8.png)

See SDK documentation for more information of Azure ML SDK: https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py

## 4. Azure ML CLI Extension

The Azure command-line interface(CLI) is a cross-platform command-line tool for managing Azure resources. It is an additional package that provides commands for working with Azure Machine Learning.

To install the Azure Machine Learning CLI extension, you should first install the Azure CLI. There is the full installation instructions for all supported platforms: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

To find the installed version and run az version:

![image](https://user-images.githubusercontent.com/71245576/115777686-9a516180-a383-11eb-935e-227ca8a4dac3.png)

For Windows, the Azure CLI is installed via a MSI, whici provides access to the CLI through the Windows Command Prompt or PowerShell.

Now download and install the current release of the Azure CLI. After the installation is complete, you will need to close and reopen any active Windows Command Prompt or PowerShell windows to use the Azure CLI.

![image](https://user-images.githubusercontent.com/71245576/115778759-20ba7300-a385-11eb-8441-9b32095f3e04.png)

After installing it, you can now run the Azure CLI with the az command from either Windows Command Prompt or Powershell. PowerShell offers some tab completion features not available from Windows Command Prompt.

First you should log in, it will open the browser to log you in. 

![image](https://user-images.githubusercontent.com/71245576/115780137-cd492480-a386-11eb-8da1-2c3c12d89403.png)

Now add the Azure Machine Learning CLI extension by running the following commands:
```bash
az extension add -n azure-cli-ml
```
Already installed:

![image](https://user-images.githubusercontent.com/71245576/115780238-e81b9900-a386-11eb-88cb-47bc6444ef3e.png)

To use the Azure Machine Learning CLI extension, run the az ml command with the appropriate parameters for the action you want to perform. For example, to list the compute targets in a workspace, run the following command:

```bash
az ml computetarget list -g 'aml-resources' -w 'aml-workspace'
```

Notice that -g parameter specifies the name of the resource group in which the Azure Machine Learning workspace specified in the -w parameter is defined. These parameters are shortened aliases for --resource-group and --workspace-name.

If there is an exception said that it is a unrecoginized arguments, you can have a try to wait it for a while. When I make sure that my workspace name and resource group name is correct, it still throws me this, I think it is on its way initializing the platform after recent installment.

See my result of compute target list:

![image](https://user-images.githubusercontent.com/71245576/115785633-b823c400-a38d-11eb-809a-57150f8e218d.png)

## 5. Machine Learning experiments

Like any scientific discipline, data science involves running experiments; typically to explore data or to build and evaluate predictive models. In Azure Machine Learning, an experiment is a named process, usually the running of a script or a pipeline, that can generate metrics and outputs and be tracked in the Azure Machine Learning workspace.

An experiment can be run multiple times with different data, code or settings; and Azure Machine Learning tracks each run, enabling you to view run history and compare results for each run..

When you submit an experiment, you use its run context to initialize and end the experiment run that is tracked in Azure Machine Learning, as shown in the following code sample:

```python
from azureml.core import Experiment

# create an experiment variable
experiment = Experiment(workspace = ws, name = "my-experiment")

# start the experiment
run = experiment.start_logging()

# experiment code goes here

# end the experiment
run.complete()
```

You can see the notifications that this experiment is ruunning, you also can check it for more details.

![image](https://user-images.githubusercontent.com/71245576/115786148-6cbde580-a38e-11eb-87bd-cfc7e90f2559.png)


## 6. Logging Metrics and Creating Outputs

Every experiment generates log files that include the messages that would be written to the terminal during interactive execution. This enables you to use simple print statements to write messages to the log. However, if you want to record named metrics for comparison across runs, you can do so by using the Run object; which provides a range of logging functions specifically for this purpose. These include:

![image](https://user-images.githubusercontent.com/71245576/115907651-2c667200-a437-11eb-84dd-c59829662ead.png)

Now let's record the records in a CSV. First we need to create a Azure experiment in the workspace, log the data from the experiment to the dataset then complete it.

```python
from azureml.core import Experiment
import pandas as pd

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace = ws, name = 'my-experiment')

# Start logging data from the experiment
run = experiment.start_logging()

# load the dataset and count the rows
data = pd.read_csv('data.csv')
row_count = (len(data))

# Log the row count
run.log('observations', row_count)

# Complete the experiment
run.complete()
```
You should take care of that the data.csv is the dataset that locates in your local directory of Jupyter. The dataset that I used is a dataset named seeds.csv.


Retrieve and view logged metrics:

```python
from azureml.widgets import RunDetails

RunDetails(run).show()
```

The output logs need to take a while:

![image](https://user-images.githubusercontent.com/71245576/115910427-d1367e80-a43a-11eb-9c9d-4eb8368f20c1.png)

You can also retrieve the metrics using the Run object's get_metrics method, which returns a JSON representation of the metrics, as shown here:

```python
import json

# Get logged metrics
metrics = run.get_metrics()
print(json.dumps(metrics, indent=2))
```

In addition to logging metrics, an experiment can generate output files. Often these are trained machine learning models, but you can save any sort of file and make it available as an output of your experiment run. The output files of an experiment are saved in its outputs folder.

The technique you use to add files to the outputs of an experiment depend on how you're running the experiment. The examples shown so far control the experiment lifecycle inline in your code, and when taking this approach you can upload local files to the run's outputs folder by using the Run object's upload_file method in your experiment code as shown here:

```python
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')
```

you use to run your experiment, you can retrieve a list of output files from the Run object like this:
```python
import json

files = run.get_file_names()
print(json.dumps(files, indent=2))
```
The result should be like this:

![image](https://user-images.githubusercontent.com/71245576/115912775-c16c6980-a43d-11eb-87f4-57acab776f27.png)

You also can check on your notebook file system or jupyter file system, but notice that it takes a while.

## 7. Running a script as an experiment

You can run an experiment inline using the start_logging method of the Experiment object, but it's more common to encapsulate the experiment logic in a script and run the script as an experiment. The script can be run in any valid compute context, making this a more flexible solution for running experiments as scale.

An experiment script is just a Python code file that contains the code you want to run in the experiment. To access the experiment run context (which is needed to log metrics) the script must import the azureml.core.Run class and call its get_context method. The script can then use the run context to log metrics, upload files, and complete the experiment, as shown in the following example:

```python
from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
data = pd.read_csv('data.csv')

# Count the rows and log the result
row_count = (len(data))
run.log('observations', row_count)

# Save a sample of the data
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
```

It will show you like this:

![image](https://user-images.githubusercontent.com/71245576/115913263-6129f780-a43e-11eb-9687-d877bbfc8ea5.png)

To run a script as an experiment, you must define a script configuration that defines the script to be run and the Python environment in which to run it. This is implemented by using a ScriptRunConfig object.

For example:

```python
from azureml.core import Experiment, ScriptRunConfig

# Create a script config
script_config = ScriptRunConfig(source_directory='./',
                                script='experiment.py') 

# submit the experiment
experiment = Experiment(workspace = ws, name = 'my-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
```
You should check on your directory that all of files that you use are in a same directory that you have stated on your commands, such as experiment.py and seeds.csv.

## Reference

Build AI solutions with Azure Machine Learning, retreived from https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/
