- `02_0regression_project.ipynb`: notebook containing the detailed code and explanations of the entire chapter 2. Good for reference when you need a quick reminder of some concept or to get example code.

- `02_exercise_0y.ipynb`: for y in {1, 2, 3, 4, 5, 6} it is the code for each of the exercises from 1 to 6 given at the end of the chapter. This is the solution as given by the author with no roginal code except for some minor changes.

- `02_exercise_0y_b.ipynb`: for y in {1, 3}  it is the code for exercises 1 and 3 extended, that is, with implementations as suggested by the author in the solution notebook. The implementations are mine and the goal is to try a better result than the ones given in the exercises, like implementing a `GridSearch` or `SelectFromModel` or some other suggestion or concept that can add something to our learning experience.

- `02_exercise_0y_b.py`: for y in {1, 3} this is the moduled version of the exercises 1 and 3 extended in pure python. It is meant to be cleaned and self contained: just run it and see the results.

- `downloading_the_data.py`: function to download and read the `housing` dataset

- `preprocessing.py`: contains the whole pipeline a Python Class, allowing it to be properly imported in the `.py` files. In more professional data science projects, the final deliverables are coded in pure python, with functions, classes and other components organized into separate modules. This aproach prevents the cluttering of the main codebase. For instance, pipelines are constructed within classes and then imported as modules into the codebase, just like any python library.
