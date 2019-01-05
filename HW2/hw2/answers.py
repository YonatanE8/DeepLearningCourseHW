r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr = 0.001
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.001
    lr_vanilla = 0.000001
    lr_momentum = 0.000000075
    lr_rmsprop = 0.00001
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1.0
    lr = 0.000000001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
Well, the answer is both yes and no.

We can clearly see in all graphs that the training hasn't converged yet since both the training and testing accuracy keep rising.
That means that if we would train for more then 30 Epochs, we would probably reach a point in which the training and testing accuracy would
have stopped improving. It is at that point that a comparison between all models should be made. We should note that we have used an
extremely low learning rate - learning rate = $1e-9$, since this is the highest learning rate which hasn't resulted in exploding gradients,
i.e. the loss became NaN, or which resulted in the training collapsing, i.e. reaching a peak accuracy of 25-30 % and then deteriorating
back to accuracy values of ~ 10 %. We have tested learning rate values from $1e-1$ to $1e-10$.

As to the comparison between the 3 plots. We can see that, unlike our initial expectation, the 'no-dropout' model resulted in both the best 
training and testing results. We claim that this result is mis-representative, and would not be so if we let the models train further,
until convergence, and result from the fact that the dropout unit slows down our training process, since in effect, at each batch we
only train those parameters which weren't dropped out. 

As to what do agree with our expectations - We can clearly see that the generalization ability of the models with dropout, (both for p=0.4, and p = 0.8), is not only on par with that of the no-dropout model (for p=0.8), inspite having a slower training process, but in fact we can see that their test accuracies are acutally higher then their training accuracies. While for both the no-dropout, the training accuracy is higher then the test accuracy. 

That clearly implies on a better generalization capability of the models which use dropout. Not only that, we can see that the model with p=0.8 has bot better training and testing accuracy results, as opposed to the model with p=0.4. That results points us to conclude that using a higher rate for the dropout units result in a more stabilized and with a better generalization abilitie. Since both models trained for 30 Epochs, but the model with p=0.8 has converged to a significantly higher results - 34.2 % training accuracy, 38.1 % testing accuracy, as opposed to 28.6 % training accuracy, 35.7 % testing accuracy for the p=0.4 model.
"""

part2_q2 = r"""
Yes, it is possible. 
Let's see it explicitly: 

For the cross-entropy loss, we have a loss matrix L, of size [N, D], with N being the batch number, and D being
the dimensions of the last layer. 
Each element in that loss matrix can have 2 possible values, for simplicity, we will only examine a single row (for example [i, :] in L):

$L[i, j] = \frac{e^(x_{j})}{\sum_{k=1}^{C} e^{x_k}}$ - If j is not the correct class prediction. 
$L[i, j] = -1 + \frac{e^(x_{j})}{\sum_{k=1}^{C} e^{x_k}}$ - If j is the correct class prediction.

Where $x_{k}$ are the scores, not probabilities, of sample i being of class k.

And the total loss is the mean of all of the loss terms.
While the accuracy, on the other hand, is dependent solely on which score is the highest, i.e. $argmax (x_{k})$.

So we can easily envision a situation in which $x_{k=y}$ is very high, while all other scores are very low.
It is now certainly possible that at each epoch, more and more $x_{k=y}$ get the highest over all scores and are being classified correctly, while the scores for other correctly classified samples are decreasing. 

In that situation, the accuracy will increase, because we keep classifying more samples correctly, but the loss will
also increase, since the scores for the correctly classes will decrease, while the scores for the incorrect classes will increase. 

We shall note that this situation can't continue for many epochs, since eventually, if the incorrect scores keep increasing, we will make classification mistakes, and the accuracy will decrease.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
Our basic expectation is that, the deeper the network, the richer representations it can learn, which should translate to better
results.

However, in experiment 1.1, we can see that the best accuracy score was for L=4 and L=2, while L=8 has scored significantly lower, and
the training with L=16 was completely unsuccessful, with L=4 being just slightly better then L=2.

Those findings are true both in the case of K=32, and for K=64.

Overall, we conclude that increasing the depth of the model leads to improved results, up until L=4. From that point on, any additional layers actually leads to a degredation in the training process, and in the case of L=16, our model becomes completely untrainable.

We believe that the main reason leading to those effect is the phenomena called vanishing / exploding gradients.

As the number of layers in each block is growing, we have longer and longer sequences of multiplying numbers which are << 1 (in the case
of vanishing gradients), or >> 1 (in the case of exploding gradients), which leads to numerical instablitiy (overflow to NaN, or underflow to 0 of the gradients' values) in the training process. 
Thus making the model untrainable since we can't properly backpropogate the gradients from the loss to our first layers.

There are several methods which can help us in overcoming the vanishing / exploding gradients phenomena, we will name two of them:
* We can use BatchNormalization layers before each activation - those layers normalize the activations by reducing the mean, and dividing
  them by the std of previous activations of that layer, thus it helps keeping the activations' values in a relatively stable range, which
  then also insures that the gradients' values are also stable and contained in a reasonable range.

* Another possible method is to use skip-connections, those connections significantly shorten elements to be multiplied while we 
  calculate the gradients using the backpropogation algorithm (i.e. calculating the Chain Rule), which was the reason in the first place
  leading to the numerical instability in the case of vanishing gradients, thus we can also make sure that the values are contained to 
  a reasonable range.
"""
# showing image
# <img src="hw2/answers_images/exp1_1_error.png">

part3_q2 = r"""
In experiment 1.2 we can see that for all of the experiment settings, K=32 has gained significantly better results then for all
other K values.

For L=2, we contend that this is the result of insufficient trainins, we can see that the training accuracy for K=32 has reached a 
platue, i.e. the training has converged, while for all other K values, the training accuracy is continue to increase (even linearly),
with the number of epochs, which is why we believe that if we would have continue the training process we would have reached better results for K values which are larger then 32.

The reason for the early stoping (Epoch=9) is that we have used an early-stopping rule, with patience=3 on the loss value, which as we 
can see from the loss graph, deacreses in stages, and probably needed more time for the next significant decrease.
In hindesight, we probably should have set the patience value to be larger then 3.

For L > 2, we can still see that the models with K=32 outperform all others, but in those cases its for different reasons.
While we can see in the training losses plots that no model has reached convergence, K=32 is still the best one, we belive that
this is due to the effect of over-fitting. 

Since now we have increased both L & K, in comparison to K=32, and L=2, so we have reached a model with high enough capacity, so that it
can just memorize the training samples for $L>=4 and K>32$. Another evidence supporting the over-fitting explenation, as that 
if we increase L from 4 to 8, while keeping K=32, we can now see a significant degradtion in the test accuracy score, which
is the classical behavior for over-fitting, resulting from an over-sized model.

And as the models get increasingly large, i.e. $K>=128, and L>=8$ the model training process is completely collapsing. 
We believe that this is due to using a learning rate step which is too large, since that while training for such 
a large model, the loss surface is extremly non-convex, and large learning steps will almost surely result in a diverging training
processes. We believe that using a learning rate which is smaller then $1e-3$ - which is the default for the Adam optimizer, would have shown a better training process, for those large models.

Comparing the results of experiment 1.1 and 1.2, we can see that in general, at least for the limited types of settings that
we have tested here, it is more worthwile to first increase the depth of the model, then the width, since the model with the
best performences was the most narrow one we tested (K=32), but with a medium-sized depth (L=4).
"""

part3_q3 = r"""
In this experiment we can see the presence of several effects which we have witnessed both in experiment 1.1 and 1.2.

First of all, we can see that the best results are again, obtained for the smaller models, with the larger one suffering from
a completely diverging training process. We would like to state that at least in the current settings (no BatchNorms, no skip-connections, learning rate of $1e-3$),
the larger models, i.e. where $L>=3$, are un-trainable. It would have been a fairer comparison, to also try to train all models
with smaller learning rates (and as such not chainging the general model's architecture), in order to at least have a somewhat
converging training process for the larger models.

That being said, we will now focus on analyzing the results we have obtained in the current settings.

We can see that if we set the number of filters to keep increasing at each layer, that we get the best overall results in relation to all
other experiments. We can also see, again, that the smaller model (L=1), has again started to reach a platue in the training accuracy, 
while the model with L=2, seems to continue improving, which again leads us to belive that, the smaller models might indeed be showing
better results, but only since the training process for the larger model (L=2) hasn't platued yet. Again we state that it would
have been worthwhile to train the models again for more epochs.

But, in a general matter, we can see that if we increase the number of convolutional kernels at each layer, we achieve the best
overall results, even with a minimal model depth - i.e. L=1, in comparison to the models from experiment 1.1 & 1.2.

As such we conclude that it is first best to increase the number of kernels from one layer to the next, and then to increase the depth
of the model, since the model with L=2, does show promise of eventually catching up, and possibly surpassing, the performences of the 
model with L=2.
"""


part3_q4 = r"""
1) We added before each activation layer a batch normalization, in order to prevent vanishing gradients. 
In addition, we added after each activation layer, a dropout in order to improve generalization and prevent 
over-fitting.

2) The results of experiment 2 are about 10% better than the results in experiment 1. 
For the models with L=3 and L=4, which have suffered from vanishing gradients in expieriment 1,
are now showing a healthy training process, due to using the batch normalization layers that we have added. 
In addition, we can see that all of the models test accuracies, are ~ monotonically increasing. 
Thus, we can infer that even though we have larger models, we have significantlly stabelized the training process.
We thus contend that, we could have continued the training process, at least untill reaching a platue in the training
accuracy results for the models, which we believe would then ended up with even better test accuracy results then the ones
we have achieved. 

Futrhermore, we can see that the models with L=1, L=2 and L=3 have all achieved results, which are on par with each other (again, we 
believe that this is due to the fact that we have not allowed the training process to reach convergence).

However, the model with L=4 has achieved accuracy scores which are roughly 10% lower.

At first, this seems to be surprising, since the training accuracy for this model, is on par with all of the other models, and since
even though it hasn't converged yet, so did non of the other models.

Why then does this phenomena happens?

We contend that this is a classical manifestation of the over-fitting effect.
While the training accuracy is as high as all other models, the testing accuracy is significantly lower, when we combine that with
the fact that for L=4 the model is exponentially larger then for L=3 (and even more so then for L<3), we can derive the conclusion that 
we have reached a model with high enough capacity in order to memorize training samples.

In order to check this hypothesis, and to prove our point that for L=4 the model is over-fitting, we believe that using a larger 
batch size in the training process would have helped (since more samples are taken into account at each step - we decreasing the chance
of memorizing examples), unfortuntaly, with only 1-GPU, we hadn't had the resources to perform such an experiment.
"""
# ==============
