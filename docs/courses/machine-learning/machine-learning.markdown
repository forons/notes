---
title: "Machine Learning - Notes"
author: [Daniele Foroni]
date: 2020-12-14
subject: "Machine Learning"
keywords: [Machine, Learning, Course]
layout: post
parent: Courses
permalink: /notes/courses/machine-learning/
lang: "en"
---


# Machine Learning - Notes

### Info

Lecturer: Andrew Ng

Offered By: Stanford University

[Course Link](https://www.coursera.org/learn/machine-learning)

[YouTube Link](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)

## Week 1

### What is Machine Learning

- Machine learning is a science that gives computers the ability to learn without explicitly programmed.
- A computer program is said to learn from experience E with respect to some task $T$ and some performance measure $P$ if its performance on $T$, as measured by $P$, improves with experience $E$

### Supervised Learning

We tell the program how to learn from the features.

We give the algorithm a set of labeled data, with the *right/correct answers*, for a set of attribute features.

**Regression.**

An example of supervised learning is regression, which is the task of predicting a continuous valued output.

For example, given the size of a house, which is the attribute feature, we want to know its prize, which is the label.
Thus, we provide to the algorithm the correct prize of a house given its size. Then, the algorithm will suggest the prize of a house of a not yet seen size.

![Regression](https://miro.medium.com/max/4800/1*ezvMPSoAZNomPOKAAdragg.png)

**Classification.**

Another example is classification, which is the task of predicting a discrete valued output (binary if we have two kinds of label, e.g., yes or no, or multi-class if we have more than two kinds of label, e.g., 0, 1, 2, or 3). 

For example, given the size of a tumor, which is the attribute feature, we want to predict if it is malignant, which is the label.
Thus, we provide to the algorithm both labeled data with the size feature of both malignant and non malignant tumors.
Then, the algorithm will identify the malignity of a tumor of a not yet seen size.

![Classification](https://miro.medium.com/max/1400/1*khHP9Jx006XddRRsPyMfmg.png)

### Unsupervised Learning

We give the algorithm non-labeled data, *without correct/right answers*, and it should find pattens/structures within the data.

**Clustering.**
An example of unsupervised learning is clustering, which is the task of grouping together similar objects based on their feature attributes.

For example, given the individuals and information about their genes, which are the features, we identify people with similar genes.
Thus, we do NOT provide to the algorithm any labeled data and the algorithm will identify the groups through the structure/patterns of the features.

It is applied in market segmentation, social network analysis, astronomical data analysis, organize computing clusters, cocktail party (audio algorithm to filter different sources).


### Linear Regression

or univariate (one variable) linear regression.
Supervised learning, because given the *right answer* for each example in the data.

Regression problem, because we predict real-valued output.

More formally, we have a *training set*, and our job is to learn from these labeled data.

- $m$: number of training examples
- $x$: input variable/feature
- $y$: output/target variable
- $(x,y)$: one training example
- $(x^{(i)},y^{(i)})$: i-th training example
- $h$: hypothesis function

    $$\textrm{training set}$$

    $$\downarrow$$

    $$\textrm{learning algorithm}$$

    $$\downarrow$$

$$\textrm{size of house } (x) \quad \rightarrow \quad \textrm{h} \quad \rightarrow \quad \textrm{estimated prize } (y)$$

The training dataset is used by the learning algorithm to create a function $h$, which is the hypothesis. We provide to the hypothesis the $x$ (the input variable) and it will return the y (the estimated value of y).

The function of the hypothesis is the following:

$$h(x)$ or $h_{\theta}(x) = \theta_0 + \theta_1 x$$


In linear regression, we want to minimize:

$$\min_{\theta_0\theta_1} \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

where $h_{\theta}(x^{(i)}) = \theta_0 + \theta_1 x^{(i)}$

By convention, we define a cost function

$$ J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$

and thus the goal is to minimize the cost function, $$ \min_{\theta_0\theta_1} = J(\theta_0, \theta_1)$$, which is also called squared error function.

![Cost Function](https://miro.medium.com/max/1400/1*Vx4bZC0gRHeBMJzG7inNZA.png)

### Gradient Descent

It is an algorithm to find the minimum values of $\theta_0$ and $\theta_1$ for
the cost function in a smart way. But it has much broader applications, not only the minimization of the cost function of linear regression.

The idea of gradient descent, is to start with some parameters (some values of $\theta_0, \theta_1$ for linear regression), and then we keep changing those parameters to reduce the goal function until we hopefully end up at a minimum.

For linear regression, for finding the minimum $\theta_0$ and$\theta_1$, let's repeat the following, until convergence:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta_0, \theta_1), \quad\text{(for } j = 0 \text{ and } j = 1 \text{)} $$

- $\alpha$ is the learning rate: it controls how big are the gradient descent steps (always $\geq 0$). If $\alpha$ is too small, gradient descent can be slow, while, if it is too large, it can overshoot the minimum and it may fail to converge, or even diverge.
- $:=$ means assignment
- for the equation, we want to simultaneously update $\theta_0$ and $\theta_1$

Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed. As we approach a local minimum, gradient descent will automatically take smaller steps (because the derivative term gets smaller). So, there is no need to decrease $\alpha$ over time.



**Gradient Descent for Linear Regression.**

The cost function $J$ for linear regression is a convex function (bowl-shaped function), that has no local optima, except for the global optima, so it will always converge to the global optima.

It turns out that in this case it is "batch" gradient descent, which refers to the fact that each step of gradient descent uses all the (batch) training examples.

![Gradient Descent](https://miro.medium.com/max/1400/1*y4MS1VknE8ZVj46eR9pkwQ.png)


## Week 2

### Linear Algebra Review

A *matrix* is a 2-dimensional array. Its dimension is: no. rows x no. columns. Position $i, j$ refers to $i$-th row and $j$-th column.

A *vector* is a 1-dimensional array, a special case of a matrix, a $n x 1$ matrix.

*Matrix addition* can be performed only with matrix of the same dimension. The result is a $n \textrm{ x } m$ matrix (with the same dimension of the original matrices), where each element is added to the element in the same $i, j$ position. It is commutative.

*Scalar multiplication* can always be performed, and it multiplies. Its result is a $n \textrm{ x } m$ matrix (as the original one), and the result value is each element multiplied by the scalar. It is commutative.

*Matrix by vector multiplication*: $A \textrm{ x } x = y$ with $A$ a $m \textrm{ x } n$ matrix, $x$ a $n \textrm{ x } 1$ matrix and $y$ will be a $m \textrm{ x } 1$ vector. To get $y_i$, multiply $A$'s $i$-th row with all the elements of vector $x$ and add them up.

For our case, to get the prediction of several features, we can perform a matrix multiplication, where $prediction = DataMatrix \textrm{ x } parameters$

*Matrix by matrix multiplication*: $A \textrm{ x } B = C$ with $A$ a $m \textrm{ x } n$ matrix, $B$ a $n \textrm{ x } o$ matrix and $C$ will be a $m \textrm{ x } o$ matrix. To get $i$-th column of matrix $C$, multiply (with matrix by vector multiplication) $A$ with the $i$-th column of $B$. It is **NOT** commutative, but it is associative ($A \textrm{ x } B \textrm{ x } C$, we can compute first ).

*Identity matrix* is denoted as $I$ (or $I_{n \textrm{ x } n}$), and is a square matrix of size $n \textrm{ x } n$, with 1s on the upper-left, bottom-right diagonal, and 0s everywhere else. For any matrix $A$, $A \textrm{ x } I = I \textrm{ x } A = A$. If $A$ is a $m \textrm{ x } n$ matrix, the first identity matrix of course is a $n \textrm{ x } n$, while the second is a $m \textrm{ x } m$ to make sense.

An *inverse* of a matrix, is that matrix that multiplied for itself, gives the identity matrix. If $A$ is a square $m \textrm{ x } m$ matrix, and if it has an *inverse*, $A \textrm{ x } A^{-1} = A^{-1} \textrm{ x } A = I$. Matrices that don't have an inverse are named "singular" or "degenerate".

The *transpose of a matrix* $A$ is denoted as $A^T$. If $A$ is a $m \textrm{ x } n$ matrix and if $B = A^T$, then $B$ is a $n \textrm{ x } m$ matrix, and $B_{ij} = A{ji}$ (the $j$-th row of $A$ becomes the $j$-th column of $B$).

## Week 3

### Linear Regression With Multiple Variables

We may have multiple features, for example, in the case of predicting the house price we saw earlier, we may have the size of the house (feature 1), the number of bedrooms (feature 2), the number of floors (feature 3), and the age of the house (feature 4).
With multiple features, we denote as $n$ the number of features (4 in our example earlier) and $m$ becomes the number of examples in the training set. $x^{(i)}$ is the input features of the $i^{th}$ example, while $x_j^{(i)}$ is the value of feature $j$ in the $i^{th}$ example.

With multiple features, now the hypothesis becomes:

$$h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

For example, we may have that:
$h_{\theta}(x) = 80 + 0.1 x_1 + 0.01 x_2 + 3 x_3 - 2 x_4$, where $x_1$ is the size of the house, $x_2$ is the number of bedrooms, $x_3$ is the number of floors, and $x_4$ is the age of the house.

For convenience of notation, define $x_0 = 1$, thus $h_{\theta}(x) = \theta^Tx$.
Thus, the cost function  $J(\theta_0, \theta_1, \ldots, \theta_n)$ can be easily called $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$.

**Gradient Descent**

The gradient descent with multiple variables should update simultaneously all the parameters of $\theta$:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j}J(\theta), \quad\text{(for } j = 0, \ldots, n \text{)}$$

A suggestion would be to scale the features, so to make sure features are on a similar scale. So the best is to features between 0 and 1, so that the gradient descent can converge faster.

One way to scale features is *mean normalization* of the training set:
$x' = \frac{x - \text{avg(}x\text{)}}{\text{max(}x\text{)} - \text{min(}x\text{)}}$, where $x'$ is the normalized value of the original value $x$. The average, minimum, and maximum are computed on the training set. Instead of the denominator range computation, the standard deviation can be computed.

Ideally, $J(\theta)$ should decrease after every iteration. If it does not work properly (e.g., overshooting), we should use a smaller learning rate $\alpha$. For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration. But, if $\alpha$ is too small, gradient decent can be slow to converge.

A suggestion from Professor Andrew Ng is to try with several $\alpha$ values:

$\ldots, 0.001, 0.003, \ldots, 0.01, 0.03, \ldots, 0.1, 0.3, \ldots, 1, \ldots$


**Features and Polynomial Regression**

The usage of a polynomial function instead of a linear function may fit the data better (e.g., $\theta_0 + \theta_1 x + \theta_2 x^2$).
In this case, we would have $x_0 = 1, x_1 = x, x_2 = X^2$.
Here, since the 3 variables have very different ranges, it is vital to apply feature scaling introduced earlier. The same would work for any other exponential value, not only square.

**Normal Equation**

It is a method to solve the cost function $J(\theta)$ for $\theta$ analytically, so basically we find the convergence in one step and not in several steps as the gradient descent does.

We construct the $m \times (n + 1)$ matrix $X$, where $m$ is the number of examples and $n$ is the number of features (and the + 1 feature is due to$x_0$, which is always equal to 1). We construct also a $m$-dimensional vector $y$, with the actual value of the prediction.
To get the value $\theta$ that minimizes the cost function:
$$\theta = (X^T X)^{-1} X^T y$$

Using normal equation:
- no need to choose $\alpha$
- no need for multiple iterations
- does not work well with large $n$, with large number of features (because it has to compute $(X^T X)^{-1}$), so if $n$ is large it is slow (large $n$ in the order of $10^4$)

It may happen that $X^T X$ is non-invertible when:
- redundant features (linearly dependent, e.g., $x_1$ size in feet$^2$ and $x_2 size in meter$^2$)
- too many features (e.g., $m \leq n$) $\rightarrow$ in this case delete some features or use regularization (that we will see later in this course)


## Week 4

### Logistic Regression

The difference between logistic regression and linear regression is that the value to predict in the latter is a continuous number, while in the former it is a discrete number.

We can label the two classes as 1 and 0 and fit the hypothesis $h_\theta (x) = \theta^T x$. Then, we can use a threshold classifier output for $h_\theta (x)$ at 0.5 $\rightarrow$ if $h_\theta (x) \geq 0.5$ then 1, else 0.
If this seems a cool idea, by adding an extra example we may have issues, thus applying linear regression to classification problems it is not a good idea.

For the logistic regression model, we want $0 \leq h_\theta (x) \leq 1$, and our hypothesis would be: $h_\theta (x) = g(\theta^T x)$, where $g$ is a Sigmoid (or logistic) function defined as $g(z) = \frac{1}{1 + e^{-z}}$. In other terms, $h_\theta (x) = \frac{1}{1 + e^{-\theta^T x}}$.

![Sigmoid Function](https://miro.medium.com/max/700/1*_DonXY5v-xVHGdh3l-TJPA.png)

The output of the hypothesis $h_\theta (x)$ is the estimated probability that $y = 1$ on input $x$, thus $h_\theta (x) = P(y = 1 | x ; \theta) \rightarrow$ the probability that $y=1$ given $x$, parametrized by $\theta$.
Moreover,
$P(y = 0 | x ; \theta) + P(y = 1 | x ; \theta) = 1$ and thus $P(y = 0 | x ; \theta) = 1 - P(y = 1 | x ; \theta)$.

For deciding the boundaries, we have to get the parameters for $h_\theta (x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2)$, lets say that $\theta_0 = -3, \theta_1 = 1, \theta_2 = 2$.
If $-3 + x_1 + x_2 \geq 0$, then $y = 1$, otherwise $y = 0$. The boundary of the two regions, where $-3 + x_1 + x_2$ is exactly equals to $0$, means that $h_\theta (x) = 0.5$.
The same would happen with a non-linear decision boundary.

![Decision Boundary](https://miro.medium.com/max/700/1*DGg-4vBh7YGCregrHzBXaw.png)

But, how do we choose the parameters $\theta$?

We have a **cost function**:
$$J(\theta) = \frac{1}{m}\sum_{i=1}^m cost(h_{\theta} (x^{(i)}, y)$$

where $cost(h\theta (x^{(i)}), y^{(i)}) = \frac{1}{2}(h_{\theta} (x^{(i)}) - y^{(i)})^2$. This is the same of the linear regression function, however, given the nature of $h_{\theta}$, $J$ is a non-convex function. Hence, we use a slightly different cost function:

$$cost(h\theta (x), y) = \begin{cases} -log(h_{\theta}(x)) & \mbox{if } y = 1 \\ -log(1 - h_{\theta}(x)) & \mbox{if } y = 0 \end{cases}$$


