r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
Yes, increasing the value of K does increase the model's generalization for unseen data, up to K = 3, after which the generalization is decreasing.
The reason that it's increasing is that as K gets larger, we take into account more and more 'neighbors' of the sample in question. In fact, increasing the size of K serves as a 'protection' from outliers, i.e. the case where the closest neighbor has, for some reason, the wrong label, won't result in a mistake, unless most of the other k - 1 closest neighbors also have a non-matching label - a rather unlikely scenerio, which is why the model works rather well.
The 2 extreme value for K, being K = 1, and K = N_Samples, default into either having each sample being tagged exactly like its closest neighbor, or as the most frequent label in the entire dataset, regarding of which samples are actually closest. From this description we can see in a rather obvious way, why those 2 options are not a good choice.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**The selection of  $\Delta > 0$ is arbitrary, because in order to find the minimal L(w) we derive by w. <br>
 $\Delta$ which is a constant gets a derivation 0. <br>
 and thats why it doesn't affect the optimization results.**
"""

part3_q2 = r"""
**1. First, we can see that for each class the model learned a unique "template" , <br>
which represent some properties of the class. <br>
For example, for the class of digit "3" we can see 3 horizontal lines. <br>**
"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The ideal pattern for the residual plot is that of a straight line, which means that the model can predict precisly the correct values.
Based on the plot that we have recieved we can estimate that the model is fitted relatively well, since although we can see several outliers samples in 
the test set, the great majority of them are relatively close to the fitted line of the model.

Since the plot for the top-5 features only relates to samples included in the training set, we will compare them to those sample points in the CV plot which 
are included in the training set. As we can see, as opposed to the the top-5 features, in which a substantial number of samples where relatively far way from the fitted plane, and outside of the dotted lines range, in the CV plot, almost all of the training samples are contained in the dotted lines, which means that the fitted hyper-plane of the model has a much better fitness to the data then that of the model trained only based on the top-5 features only.
"""

part4_q2 = r"""
The use of the 'np.logspace' function, has opposed to the 'np.linspace' function, is allowing us to search over lambdas in different orders of magnitudes, without sampling extremly large number of points. This entails a hidden assumption of smoothness on the solutions hyper-space, i.e. we assume that for relativly close lambda values, the accuracy of the model won't change much, which is why it's make more sense to search over orders of magnitude in the lambda dimension, otherwise it wouldn't have made much sense, as two very close values of lambda might lead to two very different results (which is of course, not the case).

Overall, the model is fitted K_folds * len(degree_range) * len(lambda_range) number of times, not including the final fit over the entire training data.
"""

# ==============
