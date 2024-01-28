
## Machine Learning Testing

In this workshop, you will take prototype ML code in Python and move it towards production.

## Outline

You will apply the following tools to embed a Machine Learning model in a maintainable software structure:

1. start with a prototype model
2. define a public interface
3. collect training metadata
4. run static code analysis
5. run Unit Tests
6. start a REST API server
7. log information
8. create a Docker container
9. create a package structure
10. monitor aggregate information

----

## Preparations

### Download the code

Copy and unzip or clone the repository.

### Create a virtual environment

Create a virtual environment with Python 3.11.
If you are using Anaconda, type in a terminal:

    conda create -n ml_testing python=3.11

    conda activate ml_testing

Also see [Academis](http://www.academis.eu/advanced_python/getting_started/virtualenv.html).

### Install Python libraries

Install all necessary files from the file `requirements.txt` with:

    pip install -r requirements.txt

----

## 1. The Prototype

Start with a prototype that trains and evaluates a model predicting penguin species.

Download the code in [prototype.py](prototype.py).
Put it in a dedicated folder.

Run the script from the command line:

    python prototype.py


## 2. A public interface

Copy the files `entity.py` and `penguin_predictor.py` from the `tasks/` folder.
They implement the **Facade Pattern** using Pydantic classes for data exchange.

Add the code for the `create_pipeline()` function in `penguin_predictor.py`

#### Questions:

Compare the structure of the public interface to the prototype.
How does the public interface help us maintain a quality model?

Also see: [FacadePattern](https://sourcemaking.com/design_patterns/facade)


## 3. Metadata

The metadata definition is not complete yet.

* collect a few fields that should be part of the metadata.
* add them to the definition of the metadata.
* complete the `ModelMetadata` class in `entity.py`
* complete the metadata creation in `penguin_predictor.py`

#### Question:

Why is the metadata of crucial importance?


## 4. Static Code Analysis

Before we will execute the code, we will check its quality.
Run the following code checking tools:

**black** automatically formats your code to conform with PEP8.
Execute it with:

    python -m black *.py

**isort** automatically sorts the inputs in your Python files.
Execute it with:

    python -m isort *.py

**pylint** does more strict checks against PEP8, including variable usage.
Execute it with:

    python -m pylint

(also see the `.pylintrc` file for configuration)

**mypy** checks consistency of the type hints:

    python -m mypy *.py

(add the `--strict` option for a more rigorous check).

#### Question

What types of bugs could these tools help finding?


## 5. Unit Test the interface

In `tasks/test_penguin_predictor.py` you find Unit Tests against the module `penguin_predictor.py`.
Copy those files alongside with your code.
Also copy the `test_data/` folder.

Run the tests with:

    python -m pytest

You could also include some of the static checks:

    python -m pytest --isort --mypy

#### Exercise:

Add another `assert` to the test that tests a new metadata field.

Make sure the tests passes.

#### Question:

Should your write a test checking the exact training or validation accuracy of a model? Why or why not?


## 6. Rest API Server

In `tasks/server.py` there is code to expose the model to the network.

Start the server with:

    python -m uvicorn server:app --reload --port 8080

Then go to localhost:8080/docs in the browser.
Do the following:

* train the model through the OpenAPI web interface
* check the `models/` folder and copy the id of the model
* add the `model_id` to the `config` dictionary 
* implement the API endpoint `/predict`
* run a prediction with arbitrary values
* unmask the tests in `test_server.py` and make sure the tests pass


## 7. Logging

Inspect the file `loggers.py` in `tasks/` .
Adapt your code to create a few log messages during training and inference.


## 8. Dockerize

Copy the files `Dockerfile` and `docker-compose.yml` from `tasks`.
Build a container with:

    docker compose build

Then start the service with:

    docker compose up


## 9. Package structure

To create a clean package structure that makes it easier to distribute releases of your program, consider the following steps:

* clean up the `requirements.txt` file by moving all development requirements into a separate file `dev_requirements.txt`
* create a folder `pingu_predictor/` for the Python code
* create another folder `tests/` for the test code
* add the `setup.py` from `tasks/` to the main directory of the project
* edit `setup.py` to insert the package name `pingu_predictor`

Then you should be able to locally install the package with:

    pip install -- editable .


## 10. Observing the model

In `data/` you find a test data set.
This can be used to evaluate multiple model versions on the same data.

Run the example script `tasks/monitor_test_data.py` on the models in `models`.

Which of them are good.

Can you find any indication what is wrong with the other ones?


## Further Ideas:

The examination of the model could go a lot further.
Here are some ideas:

* visualize more metadata
* summarize and log key metrics of the data over time
* plot a histogram of the probabilities for the training and test set
* use a data version control tool
* use a tool managing model versions like MLFlow
* use a monitoring tool like Grafana
