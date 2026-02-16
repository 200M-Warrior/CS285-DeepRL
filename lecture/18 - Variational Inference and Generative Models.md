이번 강의에서는 Model-based RL, Inverse RL, exploration 등 다양한 주제에서 등장하는 variational inference에 대해서 자세히 살펴 본다.
* Probabilisitc latent variable models이 무언인지, 어떤 용도로 쓰이는지 살펴본다.
* Variational inference를 통해 probabilisitc latent variable models을 학습을 통해 어떻게 approximation할 수 있는지 알아본다.
* Amortized variational inference 개념을 활용해 neural network와 같은 approximator를 variational inference와 결합해서 활용하는 방법을 살펴본다.
* 마지막으로 amortized variational inference를 활용하는 모델의 예시를 살펴본다.
  * VAE: Variational Auto Encoder
  * Model-based RL에서 쓰이는 다양한 sequence level models

또한, variational inference는 다음 강의에서 배우는 학습 기반 control과 매우 깊은 연관성을 가지기 때문에 잘 알아두자.

# 1. Probabilistic Latent Variable Models

<p align="center">
  <img src="asset/18/pm.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

Probabilistic model이란 확률 분포를 나타내는 model을 의미한다.
Random variable $x$에 대해 확률 분포 $p(x)$를 modeling한다는 것은 관측된 data(주황색 점들)에 fitting한 분포를 결정한다는 뜻이다.
* 예를 들어, 주황색 data를 표현하기 위해 multivariate normal distribution을 사용할 수 있다.

또한, probabilisitic model은 conditional models이 될 수 있다.
* $p(y|x)$라는 확률 분포가 있을 수 있는데, 이 경우 $x$에 대한 분포 모델링에는 관심이 없고 $x$가 주어졌을 때 $y$의 조건부 분포를 모델링하는 데 집중한다.
* Conditional Gaussian Model을 사용해 fitting할 수 있다.
위의 경우, $y$를 $x$에 대한 선형 함수에 가우시안 노이즈가 더해진 형태로 표현하고 있다.
  * 단순한 gaussian 분포는 고정된 종 모양의 분포이다.
  * 조건부가 붙으면 입력값 $x$에 따라 종 모양의 위치가 달라진다.
	* 즉, 종 모양의 위치를 나타내는 평균 $\mu$은 $x$의 선형 함수로 결정되고, 거기에 가우시안 노이즈(분산 $\sigma^2$)이 더해진 형태이다.
* Policy $\pi(a|s)$는 state $s$가 주어졌을 때 action $a$에 대한 조건부 분포를 제공하는 조건부 확률 모델이다.

<p align="center">
  <img src="asset/18/pm2.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

Latent variable models은 probabilistic models의 특수한 형태로, evidence나 query가 아닌 다른 variables들이 포함된 모델을 뜻한다.
* $p(x)$에서는 evidence는 없고 query는 $x$이다.
* $p(y|x)$에서는 evidence는 $x$이고 query는 $y$이다.
* Query에 관한 확률을 구하기 위해서는 latent variable를 적분해서 없애야 한다.

$p(x)$를 표한하기 위한 latent variable model의 전형적인 예는 Mixture model이다.
* 위의 사진에서 3개의 cluster를 구한다고 할 때, 주어진 데이터를 fitting하는 probabilistic model을 구하길 원할 것이다.
* 가장 편리한 방법은 3개의 multivariable normal distribution으로 구성된 mixture model로 표현하는 것이다.
* 이때 latent variable $z$은 3개의 cluster 중 하나를 나타내는 categorical discrete variable이다.
* $p(x)$를 latent variable model $\sum_z p(x|z)p(z)$로 표현한다.

조건부 모델에서도 똑같은 작업을 할 수 있다.
* $p(y|x) = \sum_z p(y|x,z)p(z)$
* $p(y|x) = \sum_z p(y|x,z)p(z|x)$
  * $z$ 역시 $x$에 의존한다고 설정
* $p(y|x) = \sum_z p(y|z)p(z|x)$
  * $y$에 대한 조건부 분포가 $x$에 의존하지 않음.

Categorical discrete variable $z$의 예시로 mixture density network가 있다.
* 이는 imitation learning을 다룰 때 나무를 피해 운선하는 상황과 같은 multimodal 상황을 처기하기 위해 언급했던 모델이다.

Gaussian mixture로 표현한 분포를 출력하는 neural network에 대해서도 배웠다.
* Neural network는 각 mixture element에 대해 여러 개의 평균 $\mu$, 표준편차 $\sigma$, 가중치 $w$를 출력한다.

위의 clustering 예제는 network 입력을 $x$, 출력을 $y$, latent variable $z$를 cluster의 identity로 둔다.
* 확률 모델 $p(y|x) = \sum_z p(y|x,z)p(z|x)$에서 neural network는 Gaussian들의 평균 $\mu$과 분산 $\sigma^2$을 출력하고 있으며, 각 mixture element의 확률인 $w$도 출력한다.
  * $z$가 $x$에 dependent하다는 것을 heuristic하게 결정되는 확률 모델 디자인 선택이다.
	* 추천 시스템의 예시로 들면 user $x$가 어떤 cluster(segment) $z$에 속하는지 파악하고, 이 둘의 정보를 활용해 특정 content를 소비할 확률 $y$를 구하는 것으로 해석할 수 있다.

<p align="center">
  <img src="asset/18/pm3.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

일반적인 latent variable model은 복잡한 probabilistic model을 단순한 분포들의 곱으로 표현하기 위해 활용된다.
* $p(x)$ 자체는 복잡하지만, $p(x|z)$, $p(z)$는 Gaussian, Categorical 분포처럼 간단할 수 있다.
* 즉, 간단한 분포 곱의 marginalization을 통해 복잡한 분포 $p(x)$를 표현할 수 있게 된다.

<p align="center">
  <img src="asset/18/pm4.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

같은 원리가 conditional case에서도 발생할 수 있다.

Multi-modal policies를 위한 conditional latent variable model의 경우, 출력에 Gaussian mixture를 사용할 수 있지만 더 일반적으로 모델의 입력으로 latent variable $z$를 추가할 수 있다.
* 이전과 같은 원리로 prior 분포 $p(z)$가 있고, 조건부 분포 $p(y|x,z)$는 매우 단순하지만, $p(y|x)$는 복잡할 수 있다.

또 이전에 model-based RL에서 살펴보았다.
* Image $o_t$를 관측하고 action $u_t$에 의존하는 latent state $x_t$를 학습하고자 하는 latent state models을 보았었다.
* 이 예시에서는 다소 복잡한 latent space를 갖는다.
* Observation 분포 $p(o|x)$가 있고, prior 분포 $p(x)$는 dynamics $p(x_{t+1}|x_t)$, $p(x_1)$을 모델링한다.
  * 이러한 latent space는 구조(structure)를 가지고 있기 때문에 prior 분포는 훨씬 더 구조화되어 있고 복잡하다.
* 위 내용은 이번 강의 마지막에 더 자세히 다룰 것이다.

<p align="center">
  <img src="asset/18/pm5.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

Latent variable model은 RL의 다양한 분야에서 볼 수 있다.
* Reward function이 주어졌을 때 최적의 action을 묻는 대신, 무언가를 수행하는 사람의 데이터가 주어졌을 때 그 사람이 무엇을 하려는지 reverse enginnering할 수 있는지를 묻는다.
  * 이것은 imitation learning분야뿐만 아니라 신경과학 및 운동 제어 분야의 인간 행동 연구에서도 흔히 볼 수 있다.
* Exploration에서 information gain을 할 때 variational inference를 사용했고, pseudo counts나 count-based bonus를 할당하기 위해 generative model과 density model을 사용했다.
  
Generative model과 latent variable model은 RL 연구에서 항상 등장한다.
* Generative model은 $x$를 생성하는 모델로, $p(x)$가 generative model이다.
* Latent variable model은 latent variable을 가지는 모델이다.
* 모든 generative 모델이 latent variable 모델인 것은 아니며, 모든 latent variable 모델이 generative 모델인 것도 아니다.
* 하지만, 일반적으로 복잡한 generative model을 간단한 latent models의 곱으로 표현 것이 훨씬 더 쉽다.

<p align="center">
  <img src="asset/18/pm6.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

이제 latent variable model $p_\theta(x)$을 학습시키는 방법을 살펴보자.
* $\theta$는 model의 parameter를 나타낸다.

데이터 $x_1, \cdots, x_N$이 있을 때, 보통 maximul likelihood fitting으로 데이터에 model을 fitting할 것이다.
따라서 자연스러운 objective function은 모든 데이터 point에 대한 $\log p_\theta(x_i)$의 합의 평균을 최대화하는 $\theta$를 구하는 것이다.

여기서 $p(x) = \int p(x|z)p(z)dz$로 주어질 때 이를 objective function에 대입한다.
이때, objective function을 계산하기 상당히 어렵다.
* $z$가 continuous variable이라면 graidnet step을 수행할 때마다 적분을 계산하는 것은 intractable하다.
  * Log 안에 적분이 있기 때문에 미분을 해도 적분 기호가 사라지지 않고, 분모와 분자에 복잡하게 남게 된다.
	* Latent variable $z$가 단순하면 적분 대신 모든 경우의 수를 summation $\sum$하면 된다.
	* 하지만, deep neural network인 경우 보통 $z$는 continuous variable이므로 적분을 수행해야 한다.

* Gaussian mixture model과 같이 경우, 각 gaussian model의 identity $z$는 discrete이므로 summation을 통해 적분을 구할 수 있다.
  * 하지만, 이 경우에도 computer numerical error와 optimization landscape가 까다롭기 때문에 좋지 못한 결과가 나온다는 것이 밝혀졌다.
	* Gaussian 분포의 값은 $\exp\left({-(x-\mu)^2}\right)$ 형태를 가진다. 그렇기 때문에 데이터가 평균 $\mu$에서 조금만 멀어져도 아주 작은 값이 되기 때문에 underflow가 자주 일어난다.
	* Gaussian mixture model은 mode 값이 많은 non-convexity이다.
	그렇기 때문에 local minima에 빠질 위험이 많다.

<p align="center">
  <img src="asset/18/pm7.jpg" alt="Probabilistic Model"  width="800" style="vertical-align:middle;"/>
</p>

Log likelihood와 gradient를 다루기 위해 expected log likelihood를 objective function으로 사용한다.
* 나중에 이번 강의에서 이것이 왜 타당한지 알게 될 것이다.

직관적으로 latent variable을 기본적으로 데이터의 부분적인 관측값으로 생각할 수 있다.
즉, 데이터는 실제로 $x,z$로 구성되어 있는데, $x$는 관측하지만 $z$는 관측하지 못하는 것이다.
따라서 본질적으로 $z$가 무엇이었을지 추측하는 것이다.
* 예를 들어, 데이터 point가 여기 있으니 아마도 이 cluster에 속할 거야라고 추측할 수 있다.

그리고 나서 $x_i$가 실제로 $z$ 값을 가진다는 fake label을 만든 다음, $x_i, z$값에 대해 maximum likelihood를 수행하는 것이다.

특정 $x_i$에 대응하는 $z$를 정확히 알 수 없지만, 그에 대한 분포는 알 수 있다.
따라서 가장 가능성이 높은 $z$값 하나만 취하는 대신, $z$에 대한 전체 분포를 가져와서 해당 $z$의 확률로 가중치를 둔 likelihood의 평균을 구하는 것이 expected log likelihood 계산이다.

기댓값은 샘플을 통해 추정할 수 있고, 기댓값의 unbiased estimate를 얻을 때 모든 $z$를 고려할 필요 없이, posterior $p(z|x_i)$에서 샘플링하고 샘들을의 log 확률을 평균낸다.
* 이를 Monte Carlo extimation이라고 부른다.
* $\log (\mathbb{E}_z)$ 형태의 경우 샘플링을 통해 Monte Calro로 unbiased estimate를 수행할 수 없지만, $\mathbb{E}_z[\log]$ 형태의 경우 적용할 수 있다.
	* Log 함수는 직선(선형성)이 아니기 때문에 $\log (\mathbb{E}_z) \neq \log(\frac{1}{N})$
	* 반면 $\mathbb{E}_z[\log] \approx \frac{1}{N}(\log)$
* Policy gradient에서도 몇 개의 episode를 실행해서 얻은 transition sample로 학습을 진행하였다.

하지만 가장 큰 challenge는 $p(z|x_i)$를 모른다는 것이다.
이것만 파악하면 expected log likelihood를 추정할 수있다.
* $p(z|x_i)$에서는 $x_i$가 주어졌을 때 이를 다시 $z$에 대한 분포로 매핑하는 것이다.

# 2. Variational Inference

## 2.1. The Variational Approximation 

Variation inference는 기본적으로 $x_i$가 주어졌을 때 $z$의 확률인 $p(z|x_i)$를 계산해야 한다.
이 식을 유도하는 과정에서 expected log likelihood가 합리적인 이유를 알게 될 것이다.

<p align="center">
  <img src="asset/18/vlb1.jpg" alt="The variational lower bound"  width="800" style="vertical-align:middle;"/>
</p>

일반적으로 $x_i$는 여러 원인($z_1, z_3, ...$)에 의해서 결정되기 때문에 $p(z|x_i)$는 꽤 복잡한 분포이다.
하지만, 일단 $p(z|x_i)$가 아주 간단한 Gaussian 분포 $q_i(z)$에 근사한다고 가정하자.
* $q_i$와 같이 아래첨자가 붙은 이유는 특정 point $x_i$에 특화된 $z$에 대한 분포이기 때문이다.
* 그림처럼 $p(z)$의 분포가 복잡하고 이를 single peak를 가진 Gaussian 분포 $q_i(z)$로 근사하는게 반드시 잘 작동하지 않을 수 있다.
단순히 최적의 fit을 가지는 Gaussian 분포를 찾는 것이다.

$p(z|x_i)$가 $q_i(z)$에 근사하면 $\log p(x_i)$에 대한 lower bound를 구축할 수 있다는 것이 밝혀졌다.
이 lower bound를 최대화하여 충분히 tight하게(실제 값에 근접하게)만들 수 있다.

위의 사진의 수식에서 어떤 수치에 $q_i(z)$가 곱해진 형태를 볼 수 있는데 이를 $q_i(z)$ 하에서의 기댓값으로 쓸 수 있다.

<p align="center">
  <img src="asset/18/vlb2.jpg" alt="The variational lower bound"  width="800" style="vertical-align:middle;"/>
</p>

Jensen's inequality를 적용하면 위와 같은 수식을 얻을 수 있다.
* Jensen's inequality은 convex/concave 함수에서 linear combination의 관계를 설명한다.
* 이를 통해 lower bound를 만들 수 있게 된 것이다.

위 수식의 장점은 모든 term이 계산 가능하다는 것이다.
* 어떤 $q_i(z)$를 선택하든 성립한다.

모든 $q_i(z)$가 최선의 lower bound를 만들어내지는 않겠지만, 임의의 $q_i(z)$를 골라 샘플링하여 첫 번째 기댓값 term을 계산할 수 있다.

두 번째 기댓값 term은 $q_i(z)$의 entropy와 같다.
Gaussian $q_i(z)$에 대해선 closed form으로 이를 계산할 수 있다.

위의 수식을 최대화하면 $\log p(x_i)$를 최대화할 수 있지만, lower bound가 너무 느슨(loose)하지 않다는 것을 증명해야 한다.
* 즉, lower bound가 실제 $\log p(x_i)$와 가까워야 한다.

이를 증명하기 위해 먼저 Entropy와 KLD에 관해서 잠깐 살펴보자.
이는 variational inference가 무엇을 하는지에 대한 좋은 직관을 제공한다.

<p align="center">
  <img src="asset/18/entropy.jpg" alt="Entropy"  width="800" style="vertical-align:middle;"/>
</p>

Entropy의 첫 번째 직관은 '확률 변수가 얼마나 무작위적인가?'이다.
* 가장 무작위적이고 예측 불가능할 때 가장 높은 entropy를 가진다.

두 번째 직관은 '$p(x)$의 분포하에 $\log p(x)$의 기댓값이 얼마나 큰가?'이다.
* 만약 주로 낮은 log 확률값을 가지면, 비슷한 확률을 할당하는 지점이 많다는 뜻으로 무작위성이 크고 높은 entropy를 가진다.

만약 $p(x_i, z)$의 그래프가 위의 그림의 하단과 같다면, 첫 번째 기댓값 term은 $p(x_i, z)$ 값의 peak에 위치한 $z$에 큰 density를 가지는 분포 $q_i(z)$를 찾게 한다.
하지만, 동시에 두 번째 기댓값 term은 $z$에 대한 분포 $q_i(z)$가 최대한 넓게 퍼지기를 원한다.

이를 통해 $p(x_i, z)$가 높은 지역들을 찾아가되, entropy 덕분에 너무 peak만 찾지 않고 주변을 cover 하는 형태를 띠게 된다.

<p align="center">
  <img src="asset/18/kld.jpg" alt="KL Divergence"  width="800" style="vertical-align:middle;"/>
</p>

두 분포 $p, q$ 사이의 KL Divergence를 풀어 쓰면 위 사진과 같아진다.

KLD의 한 가지 직관은 '두 분포가 얼마나 다른가'이다.
* 두 분포가 같으면, KLD의 값은 0이 된다.

두 번째 직관은 entropy term을 제외 했을 때 '한 분포가 다른 분포에 대해 가지는 expected log probability가 얼마나 작은가?'이다.
* Entorpy term은 VLB (variational lower bound)와 같이 $q$가 단순히 $p$ 아래에서 가장 확률이 높은 지점에서 머무는 것을 방지하여 전체 $p$를 cover하도록 만든다.

<p align="center">
  <img src="asset/18/vlb3.jpg" alt="The variational lower bound"  width="800" style="vertical-align:middle;"/>
</p>

Variational approximation에서는 사진의 최상단의 관계를 ELBO (evidence lower bound) 또는 VLB (variational lower bound)라고 부르며 $\mathcal{L}_i(p,q_i)$로 표기한다.

좋은 $q_i(z)$는 $p(z|x_i)$에 잘 근사해야하고, 이를 통해 가장 tight한 lober bound를 얻을 수 있다.
이는 KLD로 비교할 수 있다.

$q_i(z)$와 $p(z|x_i)$ KLD를 계산하여 정리하면 위와 같은 수식을 얻을 수 있다.
결국 두 분포의 KLD는 음의 VLB $-\mathcal{L}_i(p,q_i)$에 $\log p(x_i)$를 더한 것과 같다.

$\log p(x_i)$는 $q_i$에 영향을 받지 않으므로 식을 재배치하여 $\log p(x_i) = D_\text{KL} + \text{ELBO}$와 같은 형태가 되는 것을 확인할 수 있다.
이는 ELBO를 유도하는 또 다른 방법으로 KLD는 항상 0보다 크거나 같기 때문에 ELBO에서 살펴 본 부등식 형태를 얻을 수 있다.

더 나아가 KLD를 0으로 만들면 ELBO가 정확히 $\log p(x_i)$와 같아지는 것을 볼 수 있다.
즉, KLD를 최소화하는 것이 ELBO를 tight하게 만드는 것이다.
* 이것이 단순한 $q_i(z)$가 $p(z|x_i)$에 근사해야 하는 이유이다.

또한, $\log \int$가 $\mathbb{E}[\log]$로 되는 과정을 통해 왜 expected log likelihood를 사용해도 되는지에 대한 정당한 이유를 제시한다.

<p align="center">
  <img src="asset/18/vlb4.jpg" alt="The variational lower bound"  width="800" style="vertical-align:middle;"/>
</p>

정리하면 KLD를 최소화하는 $q_i$는 동시에 ELBO를 최대화하는 것과 같다진다.
* $\log p(x_i)$는 $q_i$에 영향을 받지 않으므로 무시한다.

따라서 $q_i$에 대한 ELBO를 최대화하면 KLD가 최소화되면서 lower bound가 tight해지고, $\log p(x_i)$도 증가한다.

<p align="center">
  <img src="asset/18/vlb5.jpg" alt="The variational lower bound"  width="800" style="vertical-align:middle;"/>
</p>

ELBO를 최대화하는 구체적인 구현을 살펴보자.

각 data $x_i$에 대해 $q_i(z)$로부터 $z$를 샘플링해 model paramter $\theta$에 대한 gradient를 계산해야 한다.
이때, $p(z)$는 $\theta$에 영향을 받지 않으므로 $p_\theta(x_i|z)$의 gradient만 계산해 $\theta$를 개선한다.

그 다음 동일한 ELBO loss를 최대화하도록 $q_i$를 업데이트한다.
* 이부분을 해결해야 한다.
* 만약 $q_i$가 평균 $\mu_i$와 분산 $\sigma_i$를 가진 Gaussian 분포라고 가정하면 ELBO를 평균과 분산에 대해 미분하여 gradient ascent를 수행할 수 있다.

<p align="center">
  <img src="asset/18/vlb6.jpg" alt="The variational lower bound"  width="800" style="vertical-align:middle;"/>
</p>

모든 data point $x_i$마다 개별 $q_i$를 가지므로 data가 많아질수록 학습해야 할 parameter 개수가 많아지게 된다.
따라서 개별 평균과 분산을 학습하는 대신 $q_i(z)$를 근사하는 별도의 posterior $q_\phi(z|x_i)$를 neural network로 학습시킬 수 있다.
* Posterior parameter인 $\mu_\phi(x)$와 $\sigma_\phi(x)$를 출력하는 neural network를 가지게 된다.

즉, 2개의 neural network $q_\phi: x \rightarrow z$, $p_\theta: z \rightarrow x$를 가지게 된다.
이것이 Amortized Variational Inference의 핵심 아이디어이다.

## 2.2. Amortized Variational Inference

<p align="center">
  <img src="asset/18/avi.jpg" alt="Amortized variational inference"  width="800" style="vertical-align:middle;"/>
</p>

Amortized variational inference에는 generative model $p_\theta(x|z)$와 inference network라고 부르는 $q_\phi(z|x)$가 있다.

어떤 $q$를 사용하든 $\log p(x_i)$의 lower bound를 만들 수 있지만, 이 lower bound는 $q$가 posterior $p(z|x_i)$에 근사할 때 가장 tight해진다.
* 이전에 KLD와 ELBO의 관계로 유도하였다.

ELBO에 대한 gradient ascent를 $\theta, \phi$에서 진행해야 한다.
