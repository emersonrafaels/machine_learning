<p align="center">
  	
  <a href="https://www.linkedin.com/in/emerson-rafael/">
    <img alt="Siga no Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
  </a>
	
  
  <a href="https://github.com/emersonrafaels/machine_learning/commits/main">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/emersonrafaels/machine_learning">
  </a>

  <img alt="License" src="https://img.shields.io/badge/license-MIT-brightgreen">
   <a href="https://github.com/emersonrafaels/machine_learning/stargazers">
    <img alt="Stargazers" src="https://img.shields.io/github/stars/emersonrafaels/machine_learning?style=social">
  </a>
</p>


## 💻 Test and Validation

The **error rate on new cases** (not seen in the training step) is called **error of generalization**. 
This step is util to hyperparameters adjust or model selection. 
There are **two alternatives to model validation**: *cross validation holdout* and *kfolds*.

### Holdout

<h1 align="center">
    <img alt="Machine Learning - Tests and Validation - Holdout" title="#TEST_VALIDATION_HOLDOUT" src="./assets/Holdout_i.png" />
</h1>

1. **Train + Validation dataset**: Used to choice the best model and hyperparameters
2. **Test dataset**: Used to evaluate the generalization error.

The step-by-step is:

1. Random selection of datasets: train, validation and test.
2. Choose the best model and hyperparameters using the train dataset and validation dataset.
3. Retrain using complete dataset (Train Dataset + Validation Dataset)
4. Get the generalization error (using any cost function) using test dataset.

### Kfolds

<h1 align="center">
    <img alt="Machine Learning - Tests and Validation - Kfolds" title="#TEST_VALIDATION_KFOLDS" src="./assets/kfolds.png" />
</h1>

*The example above is for any process using 5 folds. The number of folds is chosen by the data scientist.

<h1 align="center">
    <img alt="Machine Learning - Tests and Validation - Kfolds" title="#TEST_VALIDATION_KFOLDS" src="./assets/grid_search_cross_validation.png" />
</h1>

## ✍️  Libs

 - **[Scikit Learn](https://scikit-learn.org/)**

## 🛠  Tech

The following libs were used in building the project:

- [Python]

## 🚀 How execute this project

1. **Installing**: pip install -r requirements.txt

## ➊ Requirement

- Before starting, you will need had installed in your machine this tools: (The download can be done in itself page of Python or Anacondaa):
[Python](https://www.anaconda.com/products/individual).

- Do the install of requirements, getting all the necessary libraries for the execution of the project.

## 📝 License

This project are under the MIT License.

Made with ❤️ by **Emerson Rafael** 👋🏽 [Contact me!](https://www.linkedin.com/in/emerson-rafael/)

[Python]: https://www.python.org/downloads/