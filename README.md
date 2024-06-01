# Iris-Classification
The Iris dataset is a popular choice for classification tasks in machine learning. It contains measurements of four features (sepal length, sepal width, petal length, and petal width) of iris flowers belonging to three different species: setosa, versicolor, and virginica.

Our goal is to build a machine learning model that can accurately classify iris flowers into their respective species based on these four features. This is a supervised learning problem, as we have labeled data with the correct species for each sample. Since we are predicting a categorical variable (the species), this is a classification task.

We will explore different classification algorithms, such as k-nearest neighbors, decision trees, and logistic regression, to determine which model performs best on the Iris dataset. By the end of this project, we aim to build a model that can accurately classify iris flowers based on their measurements.

## Installation
Clone the repository into a directory of your choice (example directory is the `~/dev` directory)
```bash
cd ~/dev/
git clone https://github.com/matthewshawnkehoe/Iris-Classification.git
```

Create a virtual environment and setup Python as the interpreter:
```bash
python3 -m venv ~/.virtualenvs/iris-venv
```
If your project already has a virtual environment, just use that one instead. You may also want to upgrade to the newest version of python 
if your system has one installed. It is good practice to make the virtual environment outside the project directory to avoid accidentally pushing it to Git.

Activate the virtual environment (path will differ if using your own):
```bash
source ~/.virtualenvs/iris-venv/bin/activate
```
Navigate to wherever you cloned the repo:
```bash
cd ~/dev/Iris-Classification
```

Install the required libraries and packages: 
```bash
pip install -r requirements.txt
```
You should now be set up to use code from the Iris classification project in your project.

## Dataset

The data used for this project is the [Iris Flower Clasification Dataset](https://archive.ics.uci.edu/dataset/53/iris).