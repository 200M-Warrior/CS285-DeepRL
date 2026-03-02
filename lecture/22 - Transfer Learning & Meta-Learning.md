# 1. Transfer Learning

특정 domain의 RL task의 경험을 새로운 downstream task에서 효율적으로 활용하는 방법을 살펴보자.

<p align="center">
  <img src="asset/22/transfer_learning.jpg" alt="Transfer Learning"  width="800" style="vertical-align:middle;"/>
</p>

Montezuma's Revenge 처럼 model은 사람이 가진 사전 지식을 갖추고 있지 않기 때문에 reward 그 자체는 task를 어떻게 해결해야 하는지에 대한 좋은 가이드를 제공하지 않는다.
* 열쇠를 얻고, 문을 열면 reward를 준다.
* 하지만, 열쇠를 줍고 문을 여는 것이 반드시 게임을 이기는 올바른 방법이 아니다.
게임 후반에는 갈래 길로 인해 다른 장소로 이동할 수 있어, 결승점에 도달하기 위해 reward를 주는 모든 단계를 수행할 필요가 없을 수도 있다.
* 사람의 경우 사전 지식을 이용해 점수에 주의를 기울이지 않고 게임을 해결해나갈 수 있다.
  * 해골은 나쁜 것을 의미하고, 열쇠는 문을 여는 데 좋고, 사다리는 오르내릴 수 있다는 것을 알고 있다.
  * 새로운 방에 들어가는 것이 진전을 이루는 좋은 방법이라는 것도 알고 있다. 
  * 이는 세상에 대해 알고 있는 지식을 새로운 MDP로 transfer하는 일종의 transfer learning이다.

Lecture 13에서는 exploration 관점으로 얘기했다면 이번엔 transfer problem 관점에서 Montazuma's Revenge를 바라볼 것이다.

예를 들어 인디아나 존스 영화(source domain)를 보고 그것을 활용해 Montezuma's Revenge(target domain)를 해결하는 알고리즘을 만드는 상상을 할 수 있다.
이는 아주 원대한 목표로 현실적으로 아직 여기까지 도달하지 못했지만, 이를 해결하기 위한 transfer learning 알고리즘이 연구되고 있다.
* Q function, policy, dynamics/reward model 등을 transfer할 수 있다.
  * 물론 다리와 팔을 움직이는 것이 아니라 버튼을 눌러서 이동하는 것이기 때문에 유용하지 않은 actions을 배제하면 더 좋을 수 있다.
* 또한 feature나 hidden state를 transfer할 수 있다.
  * 시각적인 특징을 활용해 해골/사다리/열쇠 등 중요한 것들을 어느 정도 파악할 수 있다.
  * Visual pre-training은 실제로 큰 도움이 된다.

<p align="center">
  <img src="asset/22/transfer_learning2.jpg" alt="Transfer Learning"  width="800" style="vertical-align:middle;"/>
</p>

Transfer learning은 source와 target domain이 뭔지에 따라 활용해야 하는 알고리즘이 다르다.
* 예를 들어, 인디아나 존스 영화로 Montezuma's Revenge를 해결하는 알고리즘과 로봇이 많은 물체를 잡도록 학습시킨 후 새로운 물체를 잡도록 배포하는 알고리즘은 매우 다를 것이다.

하지만, 공통적으로 사용하는 몇 가지 아이어들이 있으며 이를 강의에서 다루겠다.
* Forward Transfer
  * 효과적으로 policy를 transfer하는 방법을 학습한다.
  * Source domain에서 학습 후 target domain에 바로 실행하거나 fine-tuning한다.
  * Source, target domain이 꽤 유사할 것을 요구한다.
  * 많은 연구들은 source domain을 target domain처럼 보이게 만들거나, source domain에서 학습된 policy가 traget domain으로 효율적으로 transfer할 방법을 찾는데 집중한다.
* Multi-task transfer
  * 여러 다른 tasks에서 학습 후 새로운 task로 전이한다.
  * 직관적으로 target domain이 source domains의 convex hull 안에 있는 것과 같다.
  * 하나의 물체에 대해서만 학습한 것보다 많은 다른 물체들로 학습했다면, 로봇이 잡아야 할 새로운 물체가 학습한 범위와 유사해 보일 수 있다.
  * Model의 특성 layer를 공유하거나 task representation을 학습해 새로운 task에 즉시 일반화하는 등 다양한 방법이 있다.
* Meta-learning
  * Transfer learning의 확장으로 나중에 새로운 target domain에 적응할 것이라는 사실을 인식하는 방식으로 source domain에서 특별한 방식으로 학습한다.
  * 종종 "Learning to Learn" 문제로 구성된다.


Meta-learning에 더 많은 원칙과 공통 알고리즘이 존재하기 때문에, 이번 강의에서는 forward transfer와 multi-task transfer는 간단하게 살펴보고 Meta-learning에 집중할 것이다.

Forward transfer와 multi-task transfer 연구들은 다소 산발적인 경향이 있어 소수의 원칙들로 정리하기 어렵다.
그렇기 때문에 광범위하게 적용 가능한 원칙들만 살펴 볼 것이다.

# 2. Forward Transfer

<p align="center">
  <img src="asset/22/forward_transfer.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

Transfer learning에서 마주칠 수 있는 문제는 위와 같다.
* Domain shift
  * 예를 들어, 시뮬레이터와 실제 환경의 시각적 차이(조명, 텍스처 등) 등이 있다.
  * Source domain에서 할 수 있는 것들이 target domain에서 전혀 불가능할 수도 있다.
* Difference in the MDP
  * Mechanism적으로 다르지만, source domain에서 상속을 받을 수 있을 만큼 구조적으로 유사한 경우를 다룬다.
    * Domain shift는 mechanism으로는 유사하다.
* Fine-tuning issues in RL
  * Source domain에서 학습된 optimal policy는 deterministic한 경향이 있어, target domain에서 exploration을 하지 않을 수 있다.
  * Exploration 부족으로 인해 새로운 환경에 적응하는 속도가 매우 느려지거나 실패할 수 있다.

<p align="center">
  <img src="asset/22/forward_transfer2.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

컴퓨터 비전에서의 domain shift의 예시를 살펴보자.
* Vision에 관한 것이지만, 고차원 representation에서의 일반적인 문제로 생각할 수 있다.

시뮬레이터 image로 학습하고 실제 세계의 image가 제시될 때 정책이 잘 작동하기를 원한다.
실제 세계의 image가 없을 수도 있지만, 몇 장의 image가 있다고 가정하자.
* 실제 세게의 action은 모르고 단지 image만 있다.

Domain shift가 존재하는 경우, 시뮬레이터에서 학습된 모델이 실제 세계에서 잘 작동하지 않을 수 있다.
이것을 완화하기 위해서 invariance assumption을 가정한다.
* Invariance assumption이란 두 domain에서 발생한 차이가 task를 수행하는 데 중요하지 않다는 가정이다.
  * 예를 들어, 현실 세계에서 비가 오더라도 근본적으로 운전하는 방법은 변하지 않는다.
* Invariance assumption을 수식으로 표현하면 다음과 같다.
  * $p(x_1) \neq p(x_2)$
  * $z = f(x_1) = f(x_2)$
  * $\therefore p(y|x_1) = p(y|z) = p(y|x_2)$
* Invariance assumption는 완벽하지 않을 수 있다.
  * 예를 들어, 현실 세계에서 비가 오면 좋은 운전 습관이 달라지기 때문에 완벽하게 invariance asssumption을 만족하지 않는다.

Invariance representation $z$를 얻는 방법은 Domain confusion, Domain adversarial neural networks 등 여러 가지가 있지만, 기본 아이디어는 유사하다.
* Neural networks에서 중간 layer를 가져온다. 주로 convolution layer의 출력을 사용한다.
* 그런 다음 그 layer를 invariance하게 만들기 위한 loss term을 추가한다.
  * 즉, $f(x_1) = z = f(x_2)$로 만든다.
  * 그렇기 때문에 실제 세계의 image가 몇 장 필요하다.
* 예를 들어, 그 layer 이후 descriminator $D_\phi(z)$를 도입해 두 domain을 구별 못하게 학습할 수 있다.
  * Invariance representation $z$를 만들기 위해 discriminator를 학습 하는 구조는 방법은 다양하다.
* 반드시 Target domain에서 RL을 실행할 필요는 없다.

위의 아이디어에서 주의할 점은 target domain의 데이터가 좋지 않을 경우이다.
* 예를 들어, 실제 환경에서 초보 운전자의 데이터만 있다고 하자.
* 그럼 $z$는 좋고 나쁨에 대해서도 invariance해지기 때문에 $z$에서 좋은 운전 데이터가 사라질 것이다.

<p align="center">
  <img src="asset/22/forward_transfer3.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

Dynamics가 다른 경우를 살펴보자.
이땐 invariance를 강제하는 것은 좋은 아이디어가 아니다.

한 가지 방법은 target domain에서 할 수 없는 일을 하는 source domain의 agent에 penalty를 주도록 reward function을 변경하는 것이다.
* 출발점에서 목표지접까지 갈 때 source domain에서는 벽에 막혀 있지 않지만, 실제 환경에서는 벽에 막혀 있다고 가정하자.
* Target domain에서의 경험을 기반으로 target domain에서 불가능한 일을 source domain에서 하는 것에 대해 매우 큰 음수 reward을 제공하도록 reward function을 변경할 수 있다.
* 이것은 domain adaptation과 매우 유사한 아이디어이지만, representation을 invariance하게 변경하는 대신 behavior를 invariance하게 변경한다.

이전의 invairance 기법과 매우 유사한 개념들이 실제로 reward function을 계산하는 데 사용될 수 있다. 
* 원하는 최적 action으로 이어지는 reward function은 target domain에서 transition이 발생할 확률의 로그 확률에서 source domain의 로그 확률을 뺀 것이다.
* 즉, target domain에서 발생할 가능성이 있는 transition의 확률을 높이는 것이다.
* 이 quantity를 근사하는 방법은 여러 가지가 있고 그 중 하는 discriminator를 학습하는 것이다.
  * 2개의 discriminator가 필요하다. 하나는 $(s, a, s^\prime)$에 대한 것이고, 다른 하나는 $(s, a)$에 대한 것이다.
  * Reward function은 $(s, a, s^\prime)$ discriminator의 출력값에서 $(s, a)$ discriminator의 출력값을 뺀 것이다.
  * 이는 조건부 확률 $p(s^\prime|s,a) = \frac{p(s,a,s^\prime)}{p(s,a)}$이기 때문에, 로그를 취하면 분모인 $p(s,a)$ term을 빼줘야 한다.
* 자세한 내용은 논문을 참고하자.

핵심은 invariance 아이디어를 기반으로 dynamics shift를 다룰 수 있다는 것이다.

하지만, source domain이 target domain에서 필요한 behavior를 허용하지 않을 수 있다.
* 예를 들어, source domain에서 벽에 막혀있지만, target domain은 뚫려 있는 경우를 다루지 못한다.
* 가장 이상적인 것은 두 domain의 교집합을 학습하는 것이다.

<p align="center">
  <img src="asset/22/forward_transfer4.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

Supervised learning에서의 fine-tuning보다 RL에서의 fine-tuning을 더 어렵게 만드는 몇 가지 문제들이 있다.
* RL task는 일반적으로 덜 다양하다.
  * 컴퓨터 비전이나 자연어 처리에서의 pre-train 및 fine-tuning은 일반적으로 수백만 개의 image나 수십억 줄의 text와 같은 매우 광범위한 설정에서 pre-train하는 시나리오에 의존 후, 훨씬 더 좁은 domain에서 fine-tuning한다.
  * RL에서는 일반적으로 그렇게 작동하지 않는다. RL에서는 pre-train할 tasks가 훨씬 더 좁을 수 있다.
* 완전히 관찰된 MDP에서의 최적 정책은 결정론적인 경향이 있다.
  * 학습할수록 policy은 더 deterministic하게 된다.
  * 이것이 첫 번째 문제와 결합되면 큰 문제가 될 수 있다. 
    * 첫 번재에서 pre-train RL task가 더 좁기 때문에 overfitting이 발생할 가능성이 높고 이것에 policy가 deterministic하면 fine-tuning이 더 어려워질 것이다.
  * Policy가 더 결정론적이 될수록 exploration을 덜 하게 된다.
  * 따라서 새로운 설정에 매우 느리게 적응하게 된다.

위의 문제들로 SL에서의 방법을 그대로 적용하는 것은 효과적이지 않고, 더 다양한 것을 고려해야 한다.
* 이것을 위한 단 하나의 기법은 없다.
* 한 가지 가능한 것은 lecture 14 unsupervised skill discovery와 lecture 19 control inference에서의 max-ent RL 방법으로 사전에 다양한 task를 해결할 수 있도록 policy를 학습하는 방법이 있다.

하지만 안타깝게도 여기에 매우 일반적이고 광범위하게 적용 가능한 원칙들이 없기 때문에 이것에 대해 더 자세히 다루지는 않겠다.

<p align="center">
  <img src="asset/22/forward_transfer5.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

Transfer learning에서 다른 유용한 방법은 source domain을 약간 조작하는 것이다.
이것의 직관은 source domain의 다양성을 높여 target domain으로의 zero-shot generalization을 높이는 것이다.
* 항상 가능한 것은 아니지만, source domain에 어느 정도 제어권이 있다면 transfer를 효율적으로 하는 몇 가지 방법이 있다.

한 가지 방법은 randomization이다.
다양한 물리적 parameter의 변동성에 대한 policy의 robustness와 generalization을 높이기 위해 source domain에 더 많은 무작위성을 추가하는 것이다.

<p align="center">
  <img src="asset/22/forward_transfer6.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

위의 논문은 deep RL에 물리적 parameter에 randomization을 적용한 연구이다.
* 한 hoper에서 학습하고 다른 질량 parameter를 가진 hopper에서 테스트 한다.
* 단일 setting으로 하면 transfer가 잘 안될 수 있으니 다양한 다른 parameter setting에서 학습하여 transfer를 더 효율적으로 한다.

실험 결과를 살펴 보자.
* 단일 질량으로만 학습하면 질량이 달라질수록 성능이 떨어진다.
* 여러 질량을 고려해 학습하면 성능이 유지된다. 즉, robust한 policy가 학습된다.

Robustness와 optimality 사이에 trade-off가 있을 것이라 생각할 수 있는데, 많은 경우에 그렇지 않다는 것을 살펴 볼 수 있다.
* 약간의 free lunch를 예상할 수 있는 이유 중 하나는 deep neural network가 강력하기 때문이다.

또 하나의 흥미로운 관찰은 충분한 parameters를 무작위화하면 실제로 무작위화되지 않은 다른 parameters들에 대한 robustness가 생긴다는 것이다.
* 위 연구에서는 4가지 물리적 parameter에 대해 하나를 제외하고 무작위화를 했다.
* 질량은 항상 동일하지만 마찰, 관절 감쇠 및 전기자를 변경했고, 그 결과 실제로 질량에 대해 꽤 robust한 것을 발견했다.
* 이는 물리적 parameter가 redundant effects를 가지기 때문이다.
  * 질량을 줄이는 것을 마찰을 줄이는 것과 같다.
  * 따라서 마찰을 변경하는 것이 질량을 변경하는 것과 정확히 같지는 않지만, 마찰을 무작위화하면 질량에 대해 조금 더 robust해진다.
* 따라서 충분한 parameters를 변화시키면, 변화시키지 않은 parameters들에 대해서도 여전히 약간 robust할 수 있다.

Target domain에서 약간의 경험을 얻으면 약간 적응하고, parameters 분포를 target domain에 더 가깝게 변경하면 더 잘 작동할 것이다. 
Radomization 아이디어는 RL transfer learning에서 매우 광범위하게 사용된다.
* 드론 날리기, 보행 policy 학습하기 등에서 인기 있다.

<p align="center">
  <img src="asset/22/forward_transfer7.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

<p align="center">
  <img src="asset/22/forward_transfer8.jpg" alt="Forward Transfer"  width="800" style="vertical-align:middle;"/>
</p>

# 3. Multi-task Transfer

Multi-task transfer에 대한 간단한 아이디어를 살펴보자.
* 나중에 meta learning을 논의할 때 더 많이 논의할 것이다.

기본 아이디어는 여러 tasks를 학습함으로써 더 빠르게 학습하고 더 잘 transfer할 수 있다는 것이다. 
* Tasks를 개별적으로 학습할 수 있지만, tasks가 일부 공통 구조를 공유할 가능성이 매우 높다.
* Randomization과 유사한 직관으로, 더 다양한 학습 상황이 있을수록 테스트 상황이 어느 정도 익숙해 보일 가능성이 더 높다.

Multi-task transfer를 위한 다양한 기법들이 제안되었지만, 전반적으로 작동하는 단 하나의 킬러 기법은 없다.

<p align="center">
  <img src="asset/22/multi_task_transfer.jpg" alt="Multi-task Learning"  width="800" style="vertical-align:middle;"/>
</p>

Multi-task RL은 joing MDP에서의 single-task RL에 해당한다.
* 표준 RL setting에서는 initial state $s_0$를 $p(s_0)$에서 샘플링하고 policy를 roll-out한다.
* Multi-task 문제에서는 initial state distribution만 변경하면 된다.
  * 예를 들어, 동일한 policy로 여러 다른 Atari 게임을 학습하는 경우, 일반 Atari 게임에서 initial state 분포는 그 게임의 시작이다. 
  * Multi-task Atari MDP에서 initial state 분포는 게임들의 분포이다.
  즉, 첫 번째 time step에서 게임을 샘플링(선택)하고, 그 이후에 그 게임을 플레이한다.

Atari 게임은 화면을 보면 어떤 게임을 하고 있는지 알 수 있다.
하지만, 집에 로봇이 있고 로봇이 동일한 initial state에서 빨래를 하거나 설거지를 할 수 있다고 하자.
Multi-task learning을 표준 RL 문제로 인스턴스화하려면 agent에게 어떤 task를 해야 하는지 나타내기 위한 context를 할당해야 한다.
* One-hot vector, discriminator, target image or text 등이 될 수 있다.
* 이를 contextual policy라고 한다.

<p align="center">
  <img src="asset/22/multi_task_transfer2.jpg" alt="Multi-task Learning"  width="800" style="vertical-align:middle;"/>
</p>

표준 policy가 $\pi(a|s)$라면 contextual policy는 $\pi(a|s, \omega)$이다.
* Actor-critic 방법이나 Q learning 방법으로 학습한다면 Q function도 omega $\omega$를 입력으로 받는다.
* Context를 state에 추가하기 위해 state space를 증강하기만 하면 된다.
  * $\omega$를 state에 추가하여 MDP 정의만 변경하면 기존 알고리즘을 변경할 필요가 없다.

<p align="center">
  <img src="asset/22/multi_task_transfer3.jpg" alt="Multi-task Learning"  width="800" style="vertical-align:middle;"/>
</p>

Contextual policy의 가장 일반적인 형태는 goal-conditioned policy이다.
* 여기서 context $\omega$는 goal state를 의미한다. 
* Reward는 현재 state가 목표 state(context)에 도달했는지 여부로 정의된다.
* 즉, 여러 task 중 원하는 task (goal state)를 하도록 policy를 학습하는 것이다.
* Reward를 수동으로 정의할 필요가 없기 때문에 특히 편리할 수 있다.
* 새로운 task가 다른 goal state로 표현되면 zero-shot transfer를 할 수 있다.

하지만 몇 가지 단점이 있다.
* 일반적으로 goal-conditioned policy를 학습하기 어렵다.
  * Lecture 14의 예시와 같이 빨간 원을 피하면서 초록 위치에 도달해야 할 경우, 이를 설명하는 single goal이 없다.
* 문제 정의는 간단하지만, 잘 작동하게 만들려면 다양한 트릭이 필요하다.
* 강의 범위를 벗어나므로 자세한 것은 다양한 논문을 참곻자.

# 4. Meta Learning

Meta learning은 multi-task learning의 확장된 개념이다.
* Multi-task learning은 단순히 여러 task를 잘 해결하는 방법을 학습한다.
* Meta learning은 여러 tasks를 활용해 새로운 task를 더 빠르게 학습하는 방법을 배운다.
 
## 4.1. Meta Learning in Supervised Learning

<p align="center">
  <img src="asset/22/meta_learning.jpg" alt="Meta Learning"  width="800" style="vertical-align:middle;"/>
</p>

Meta learning은 여러 task를 학습 과정을 일반화해서 새로운 task 습득을 효율적으로 하는 것이다.
* 따라서 meta learning은 '학습하는 방법을 학습하는 것'이다.
* Optimzer 학습, 경험을 통해 새로운 task를 해결하는 RNN 학습, 새로운 task에 빠르게 fine-tuning될 수 있는 representation 학습 등이 있다.
  * 매우 다른 것처럼 보이지만, 동일한 원리에 서로 다른 아케틱처를 선택한 것이다.
* RL 관점에서는 새로운 task를 빠르게 습득하기 위한 exploration 방법을 학습할 수 있고, network가 새로운 task에 맞게 feature representation을 수정할 수 있도록 훈련되었다면 쓸모있는 또는 쓸모없는 behavior를 빠르게 파악할 수 있다.

<p align="center">
  <img src="asset/22/meta_learning2.jpg" alt="Meta Learning"  width="800" style="vertical-align:middle;"/>
</p>

Computer vision에서의 task를 통해 supervised learning에서의 meta learning의 원리를 살펴보자.
* 매우 유사한 원리가 RL에서도 실제로 작동한다.

Meta-training dataset은 source domain이고 meta-testing dataset은 target domain이다.
* 각 task마다 classification 하는 종류가 다르다.
  * 첫 번째 task의 경우 클래스 0은 새, 클래스 1은 버섯, 클래스 2는 개, 클래스 3은 사람, 클래스 4는 피아노이고, 두 번째 task의 경우 클래스 0은 체조 선수, 클래스 1은 풍경, 클래스 2는 탱크, 클래스 3은 배럴 등 일 수 있다.
  * 이러한 할당은 수동으로 하거나 임의로 무작위화할 수 있으며, 위의 경우는 무작위이다.
* $\mathcal{D}^\text{tr}$은 meta-training와 meta-testing의 모든 trainin data를 의미한다.

Train set을 처리하기 위해 RNN, transformer와 같은 구조가 잘 작동할 수 있다.
* Few-shot으로 train image-label tuple을 입력하고 test image를 읽어 test label을 예측할 수 있다.
* 자세한 내용은 추후에 살펴 볼 것이다.

<p align="center">
  <img src="asset/22/meta_learning3.jpg" alt="Meta Learning"  width="800" style="vertical-align:middle;"/>
</p>

먼저, meta learning으로 무엇이 학습되고 있는지 좀 더 구체적으로 이야기 해보자.
학습하는 방법을 학습한다고 했을 때 '학습'이란 무엇일까?

일반적인 SL은 train set에서 loss function을 최소화하도록 parameter $\theta$를 학습한다.

일반적으로 meta learning은 parameter $\phi$에 대한 test set의 loss를 최소화하도록 parameter $\theta$를 학습한다.
* Train set를 입력으로 neural network $f_\theta$이 parameter $\phi$를 생성한다.
* 학습되는 parameter는 $\theta$이다.
* 즉, few-shot train set을 바탕으로 parameter $\phi$를 생성하고, 이 parameter $\phi$로 test set에 대해 예측을 진행한다.

<p align="center">
  <img src="asset/22/meta_learning4.jpg" alt="Meta Learning"  width="800" style="vertical-align:middle;"/>
</p>

RNN의 구조에서 $\theta$는 RNN과 classifier parameter를 의미한다.
* $\phi$는 학습 데이터로 얻은 hidden state와 작은 classifier network의 parameter 인 $\thetha_p$를 의미한다.
* Image와 label을 RNN encoder에 입력해 hidden state $h_i$를 얻고, label을 예측하는 작은 classifier에 전달된다.

Target domain에서도 유사하게, train data image (few-shot)를 바탕으로 target domain의 hidden state를 생성하고 classifier로 예측을 진행한다.

## 4.2. Meta Learning in RL