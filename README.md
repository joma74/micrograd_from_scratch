# Setup a Jupyter datascience notebook inside docker

./docker see https://github.com/jupyter/docker-stacks/tree/main/examples/docker-compose/notebook

https://vsupalov.com/docker-arg-vs-env/

https://jupyter-docker-stacks.readthedocs.io/en/latest/using/recipes.html#using-mamba-install-recommended-or-pip-install-in-a-child-docker-image

## Diverse Installation Tips
https://scalingpythonml.com/2020/12/12/deploying-jupyter-lab-notebook-for-dask-on-arm-on-k8s.html

# Content

## Deriving vs Differentiating
Deriving means creating a formula or equation for a given situation or phenomenon. Differentiating means taking the derivative of a function, as in calculus.

## Loss function

L stands for Loss function. https://en.wikipedia.org/wiki/Loss_function
> In mathematical optimization and decision theory, a loss function or cost function (sometimes also called an error function) [1] is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function. An objective function is either a loss function or its opposite (in specific domains, variously called a reward function, a profit function, a utility function, a fitness function, etc.), in which case it is to be maximized. The loss function could include terms from several levels of the hierarchy.

## Affect vs Effect vs Impact

Affect is a verb meaning ‘influence or cause someone or something to change’.

Effect is a noun that means ‘the result of an influence’.

| affect | Affect | effect | Effect | impact | Impact |
| --- | --- | --- | --- | --- | --- |
| Affect as a verb means "to influence." | Affect as a noun has a specialized meaning in medicine and psychology, referring to moods and feeling as distinct from thoughts or knowledge. | Effect as a verb means "to bring about, to produce," or to "accomplish something." | Effect as a noun means "result." | Impact as a verb means "strike with a blow" or "to pack firmly together." | Impact as a noun means "a collision." |

## Chain rule

https://en.wikipedia.org/wiki/Chain_rule#Intuitive_explanation
> The chain rule may also be expressed in Leibniz's notation. If a variable z depends on the variable y, which itself depends on the variable x (that is, y and z are dependent variables), then z depends on x as well, via the intermediate variable y. <br/><br/> Intuitively, the chain rule states that knowing the instantaneous rate of change of z relative to y and that of y relative to x allows one to calculate the instantaneous rate of change of z relative to x as the product of the two rates of change.

Example
> "If a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 × 4 = 8 times as fast as the man."

> Let z, y and x be the (variable) positions of the car, the bicycle, and the walking man, respectively. The rate of change of relative positions of the car and the bicycle is $\frac{dx}{dy} = 2$. Similarly, $\frac{dy}{dx} = 4.$ <br/> So, the rate of change of the relative positions of the car and the walking man is <br/> $\frac{dz}{dx} =\frac{dz}{dx} = {{\frac{dz}{dy}} \cdot {\frac{dz}{dy}}} = 2 \cdot 4 = 8$

## Rise over run formula (or) slope (m)

Consider a line joining two points A(x1,y1) abd B(x2,y2). The slope of this line is the ratio of the difference in y coordinates and the difference in x coordinates.

Rise = difference between the y-coordinates of points A and B

Run = difference between the x-coordinates of points A and B

Rise over run formula (or) slope (m) $= \frac{y2 - y1}{x2 -x1} = \frac{Δy}{Δx} $

## Hyperbolic functions

https://en.wikipedia.org/wiki/Hyperbolic_functions

## Images

https://cs231n.github.io/convolutional-networks/

## Gradient descent

https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/optimizing-multivariable-functions/a/what-is-gradient-descent
> Gradient descent is an algorithm that numerically estimates where a function outputs its lowest values. That means it finds local minima, but not by setting 
$ \nabla f = 0 $ like we've seen before. Instead of finding minima by manipulating symbols, gradient descent approximates the solution with numbers. Furthermore, all it needs in order to run is a function's numerical output, no formula required. ... gradient descent can give us these estimates no matter how elaborate our function is.
>
> Think of a function $f(x, y)$ that defines some hilly terrain when graphed as a height map. 
>
> That might spark an idea for how we could maximize the function: start at a random input, and as many times as we can, take a small step in the direction of the gradient to move uphill. In other words, walk up the hill.
>
> To minimize the function, we can instead follow the negative of the gradient, and thus go in the direction of steepest descent. This is gradient descent.


