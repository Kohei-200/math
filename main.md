## Why Bias reduced with weighted average (Ada Boost) {#why-bias-reduced-with-weighted-average-ada-boost .unnumbered}

- Ada Boost labels take values of -1 or 1

- weight of a sample to be chosen changes

1.  Initialize weights for samples $w_i^{(1)} = \frac{1}{N}$ ($N$ = \#
    of samples)

2.  Repeat steps 3-5 for m-th weak learner ($m = 1 \dots M$)

3.  Train the weak learner to minimize:\
    (error func)
    $E^{(m)} = \frac{\sum_{i=1}^{N} [\hat{y}^{(m)}(x_i) \ne y_i]}{\sum_{i} W_i^{(m)}}$

4.  Determine the weight for the trained learner:\
    $\alpha^{(m)} = \ln \left( \frac{1 - E^{(m)}}{E^{(m)}} \right)$
    ("the amount of say")

5.  Update the weights for training data: $w_i^{(m+1)} = \dots$

**Implication:**

1.  if $E^{(m)} < 0.5$, then $\frac{1 - E^{(m)}}{E^{(m)}} > 1$,
    resulting in $\alpha^{(m)} > 0$\
    *better learners obtain higher positive voting weight*

2.  Given $y_i \in \{-1, 1\}$ and $\hat{y}^{(m)} \in \{-1, 1\}$

    **Case A:**
    $y_i = \hat{y}^{(m)}(x_i) \rightarrow \hat{y}^{(m)}(x_i)y_i = 1$
    ([Correct Case]{style="color: red"}) (Low Bias)\
    then $w_i^{(m+1)} = w_i^{(m)} \exp(-\alpha_m)$\
    since $\alpha_m > 0$, $\exp(-\alpha_m) < 1 \rightarrow$ **[weight
    decreases]{style="color: red"}**

    **Case B:**
    $y_i \ne \hat{y}^{(m)}(x_i) \rightarrow \hat{y}^{(m)}(x_i)y_i = -1$
    ([Highly biased]{style="color: red"})\
    then $w_i^{(m+1)} = w_i^{(m)} \exp(+\alpha_m)$\
    since $\alpha_m > 0$, $\exp(+\alpha_m) > 1 \rightarrow$ **[weight
    increases]{style="color: red"}**

*Each learner targets the localized bias of the previous ensemble.
Sequential.*\
*Other General Boosting doesn't use weighted average.*

## More About Boosting {#more-about-boosting .unnumbered}

[**Ensemble Methods: Foundations and Algorithms**, Zhi-Hua
Zhou](https://doi.org/10.1201/b12207)\
**General procedure**

**Input:** Sample Dist: $D$, Base learning algo: $L$, \# of learning
rounds $T$

1.  $D_1 = D$

2.  for $t = 1 \dots T$

3.  $h_t = L(D_t)$ (train a learner)

4.  $\epsilon_t = P_{x \sim D_t}(h_t(x) \ne f(x))$ (evaluate the error)

5.  if $\epsilon_t > 0.5$ then break

6.  $\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$ -
    odds ratio.

7.  $D_{t+1}(x) = \frac{D_t(x)}{Z_t} \times \begin{cases} \exp(-\alpha_t) & \text{if } h_t = f \\ \exp(\alpha_t) & \text{if } h_t \ne f \end{cases}$\
    $= \frac{D_t(x) \exp(-\alpha_t \cdot f(x) \cdot h_t(x))}{Z_t}$

8.  end

**Output:** $H(x) = \text{Combine-Outputs}(\{h_1(x), \dots, h_T(x)\})$

**Ada Boost:** Additive model, gradient descent on an Exponential loss
(only for classification) $[-1, 1]$

**Output:** $\text{sign}(H(x))$ with model function:
$H(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$

$H(x) =$ continuous real number\
$|H(x)| =$ margin (physical distance of point to the model's decision
boundary)

- Sign of margin = correct / incorrect

- Value of margin = confidence (high/low)

$(e^u)' = e^u \cdot u'$\
Now, exponential loss $l_{exp}(H) = e^{-yH}$\
then the expected exponential loss over a data distribution is:\
$l_{exp}(H|D) = E_{x \sim D}[e^{-f(x)H(x)}]$\
$\frac{\partial l_{exp}(H|D)}{\partial H(x)} = -e^{-H(x)}P(f(x)=1|x) + e^{H(x)}P(f(x)=-1|x)$\
if $H(x)$ can minimize the loss, it can solve by
$\frac{\partial l_{exp}(H|D)}{\partial H(x)} = 0$\
i.e., $H(x) = \frac{1}{2} \ln \frac{P(f(x)=1|x)}{P(f(x)=-1|x)}$

then
$\text{sign}(H(x)) = \begin{cases} 1, & P(f(x)=1|x) > P(f(x)=-1|x) \\ -1, & \text{otherwise} \end{cases}$
(ignore $= \frac{1}{2}$)\
$= \text{argmax}_{y \in \{-1,1\}} P(f(x)=y|x)$\
**Finding the optimal weight $\alpha_t$:**\
Once the classifier $h_t$ is generated from $D_t$, its weight $\alpha_t$
is estimated by minimizing the exponential loss.\
$l_{exp}(\alpha_t h_t | D_t) = e^{-\alpha_t} \cdot P(\text{correct}) + e^{\alpha_t} \cdot P(\text{incorrect})$\
$= e^{-\alpha_t}(1 - \epsilon_t) + e^{\alpha_t} \cdot \epsilon_t$
(weighted error rate under $D_t$)

Set derivative to 0 and get (line 6):\
$\rightarrow$ larger $\alpha_t$ when $\epsilon_t$ is small.\
At iteration $t$, with a sequence of base learners $h_1$ to $h_{t-1}$
and corresponding weights,
$H_{t-1}(x) = \sum_{i=1}^{t-1} \alpha_i h_i(x)$\
and AdaBoost adjusts the sample distribution such that the base learner
$h_t$ can learn from mistakes of $H_{t-1}$.

**Deriving the dist. update**

Notion Required: Taylor expansion:\
$e^u = 1 + u + u^2/2! + u^3/3! + \dots$ (don't need constant to get
critical point)\
the ideal $h_t$ which can correct all mistakes of $H_{t-1}$ should
minimize:\
$l_{exp}(H_{t-1} + h_t | D) = E_{x \sim D}[e^{-f(x)(H_{t-1}(x) + h_t(x))}]$\
$\rightarrow h_t(x) = \text{argmin}_{h} l_{exp}(H_{t-1} + h_t | D)$\
now $E_{x \sim D}[e^{-f(x)(H_{t-1}(x) + h_t(x))}]$\
$= E_{x \sim D}[\underbrace{e^{-f(x)H_{t-1}(x)}}_{\text{constant for } h_t} \cdot \underbrace{e^{-f(x)h_t(x)}}_{u}]$\
$\simeq E_{x \sim D}[e^{-f(x)H_{t-1}(x)} \cdot (1 - f(x)h_t(x) + \frac{1}{2} f^2(x)h_t^2(x))]$\
since $f^2(x) = h_t^2(x) = 1$, $(-f(x)h_t(x))^2 = f^2(x)h_t^2(x) = 1$\
dropping $u^3/3!$ form and after approximates constant - dropped\
$\simeq E_{x \sim D}[e^{-f(x)H_{t-1}(x)} \cdot (1 - f(x)h_t(x) + \frac{1}{2})]$\
then
$h_t(x) = \text{argmax}_{h_t} E_{x \sim D}[e^{-f(x)H_{t-1}(x)} \cdot f(x)h_t(x)]$\
now by definition weighted expectation means:\
$E_{x \sim D}[w(x) \cdot g(x)] = \sum_{x} D(x) \cdot w(x) \cdot g(x)$\
when $D_t(x) = D(x) \cdot w(x)/Z$,

What it did: move the $w$ out of $E$ into dist.\
$= Z \sum_{x} D_t(x) \cdot g(x)$\
$= Z \cdot E_{x \sim D_t}[g(x)]$\
Now apply $w(x) = e^{-f(x)H_{t-1}(x)}$ and $g(x) = f(x)h_t(x)$\
and
$E_{x \sim D}[e^{-f(x)H_{t-1}(x)} \cdot f(x)h_t(x)] = Z \cdot E_{x \sim D_t}[f(x)h_t(x)]$

Where $D_t(x) = \frac{D(x) e^{-f(x)H_{t-1}(x)}}{Z}$ and
$Z = \sum_{x} D(x) \cdot e^{-f(x)H_{t-1}(x)}$\
Since $Z$ is just a positive constant $\rightarrow$ dropped\
and $h_t(x) = \text{argmax}_{h_t} E_{x \sim D_t}[f(x)h_t(x)]$\
here,

  $H_{t-1}$             $f(x)H_{t-1}(x)$   $e^{-f(x)H_{t-1}(x)}$   weight
  --------------------- ------------------ ----------------------- --------
  Correct/Confident     large pos.         small                   low
  uncertain/near zero   near zero          $\sim 1$                medium
  incorrect/confident   large neg.         large                   high

Since $f(x)h_t(x) = 1 - 2 \cdot I[f(x) \ne h_t(x)]$\
$h_t(x) = \text{argmin}_{h_t} E_{x \sim D_t}[I(f(x) \ne h_t(x))]$

**Next iteration prep (recursive form)**\
$D_{t+1}(x) = \frac{D_t(x) \cdot e^{-f(x) \cdot \alpha_t \cdot h_t(x)}}{Z_t}$

which expands to the relationship between $D_{t+1}, D_t$ (residue
minimization):\
$D_{t+1}(x) = \frac{D_t(x)e^{-f(x)H_t(x)}}{E_{x \sim D}[e^{-f(x)H_t(x)}]}$\
$= \frac{D_t(x) e^{-f(x)H_{t-1}(x)} \cdot e^{-f(x)\alpha_t h_t(x)}}{E_{x \sim D}[e^{-f(x)H_t(x)}]}$\
$= \frac{D_t(x) \cdot e^{-f(x)\alpha_t h_t(x)} \cdot E_{x \sim D}[e^{-f(x)H_{t-1}(x)}]}{E_{x \sim D}[e^{-f(x)H_t(x)}]}$

$Z_t = \frac{\sum_{x} D_t(x) \cdot e^{-f(x)\alpha_t h_t(x)}}{E_{x \sim D}[e^{-f(x)H_t(x)}]}$

  $h_t$       $f(x)h_t(x)$   $e^{-f(x)\alpha_t h_t(x)}$   $\times D_t(x)$
  ----------- -------------- ---------------------------- --------------------------------------------
  Correct     +1             $e^{-\alpha_t}$              $e^{-\alpha_t} \text{ if } h_t(x) = f(x)$
  Incorrect   -1             $e^{\alpha_t}$               $e^{\alpha_t} \text{ if } h_t(x) \ne f(x)$

*which yields (line 7)*
