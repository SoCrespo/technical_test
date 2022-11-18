## Please notice

This repository is intented to be a basis for discussion in the context of a techical test. This is not a complete solution ready to deploy but a baseline approach. 

Specifically, some issues must be addressed:

- somes subclasses are incompatible with previous classes (ex: for "Cups" class, there will be no "Man" / "Woman" subclass): a validation system should be implemented to avoid this

- an option could be to predict the first class, then add it to the features to predict the second, and so on, so we woudn't work with a multi-output model, but with 4 different successive models.  

- more generally, the repo should be cleaned (no notebook) and the code should be refactored to be more readable and maintainable. 

**Since the test was taken in limited time**, all these options have not been explored here, but in real-life they would of course be the next actions to take to produce a robust, explainable and efficient solution.
