#import "@preview/codly:1.1.1": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#import "@preview/zero:0.3.1": *

#let title = [
  Course report: Improving a CNN with residual connections and feature recalibration
]

#set page(
  paper: "a4",
  header: align(right, title),
  numbering: "1",
  columns: 2,
)
#set par(justify: true)
#set text(
  font: "Libertinus Serif",
  size: 11pt,
)
#show heading.where(
  level: 1
): it => block(width: 100%)[
  #set align(center)
  #set text(13pt, weight: "regular")
  #smallcaps(it.body)
]

#show heading.where(
  level: 2
): it => text(
  size: 11pt,
  weight: "bold",
  style: "normal",
  it.body + [.],
)

#place(
  top + center,
  scope: "parent",
  float: true,
  clearance: 2em,
)[
  #align(center, text(17pt)[
    *#title*
  ])

  #grid(
    columns: (1fr),
    align(center)[
      Rong Bao \
      Student ID: 23060827 \
      #link("mailto:rong.bao@hdu.edu.cn")
    ]
  )
]

#place(
  auto,
  scope: "parent",
  float: true,
  clearance: 2em,
)[
  #figure(
    image("assets/structure.png"),
    caption: [
      Structure of our network, assuming a trivial batch size $N_B = 1$.
    ]
  ) <fig-structure>
]

= Introduction
In this course report, we are given a boilerplate convolutional neural network (CNN) that recognizes handwritten digits from MNIST dataset #cite(label("6296535")), and are tasked to 1) add more convolutional layers so that the total number of layers is at least 6, then 2) introduce residual connections to the network, then 3) add some attention mechanism to the model, and finally 4) evaluate the improved performance of this network. By completing the required steps and incorporating the Squeeze-and-Excitation feature recalibration mechanism in the model, we are able achieve higher-than-baseline performance at a low cost.

== Residual networks
Training very deep convolutional networks is considered challenging for both slow convergence and performance degradation. Residual networks (ResNets) #cite(label("DBLP:journals/corr/HeZRS15")) utilize skip connections to address the problem of vanishing gradients and facilitate the training of very deep architectures. Utilizing skip connections, the model learns to optimize the function $accent(bold(y), hat) = F(bold(x); bold(#sym.theta)) + bold(x)$, learning the distance vector between a predicted output and its location on the sample manifold. ResNets allow the gradient to flow directly from earlier layers to later layers, thus improving the training process.

== Feature recalibration
Convolutional networks capture features of an input at different levels of abstraction, providing both spatial correlations and invariance through the use of a single kernel each feature channel. However, this model of computation does not address potential relationships between different feature channels. Squeeze-and-Excitation modules (SE blocks) #cite(label("DBLP:journals/corr/abs-1709-01507")) explicitly model this inter-feature correlations with a low overhead by first "squeeze" each feature channel map into a scalar with global average pooling, then use two fully-connected layers to produce a per-channel weight $bold(k)$ that "excites" original features. By calculating $accent(bold(x), tilde) = bold(k)(bold(x); bold(#sym.theta)) #sym.dot.circle bold(x)$, where $#sym.dot.circle$ stands for Hadamard product, SE blocks emphasize the important channels and suppress less important ones, thus recalibrating across feature maps.

= Method

== Anatomy of handout code
The provided boilerplate is a openly-available example file provided by #cite(label("PytorchExamples2017")), and uses PyTorch #cite(label("DBLP:journals/corr/abs-1912-01703")) as its framework of implementation. The baseline network is composed of two convolutional layers and two linear layers for feature extraction and summarization. For training, the code uses Adadelta optimizer #cite(label("DBLP:journals/corr/abs-1212-5701")) with initial learning rate $l_0 = 1$ and scheduling $l' = 0.7l$ for each epoch. The training and evaluation set are traken from the MNIST dataset after normalization to $#sym.mu = 0.1307$, $#sym.sigma = 0.3081$. The images are grouped into batches of $N_B = 64$ and $1000$ for training and evaluation respectively.

We implement all enhancements to the network in PyTorch. We keep the optimizing algorithm and learning rate scheduling as-is, and modify none of the hyperparameters during our experiments. The effect of choosing different sets of hyperparameters is out of scope for this report.

== Implementation of SE blocks
As per #cite(label("DBLP:journals/corr/abs-1709-01507")), SE blocks learn to attend to important channels by optimizing on a per-channel factor. This is done by implementing a fully-connected, single-layer autoencoder operating on the global average of each feature map. However, in PyTorch, the fully-connected (`torch.nn.Linear`) layer only operates on the last dimension of the input tensor. Therefore, after obtaining feature map average in $RR^(N_B #sym.times C #sym.times 1 #sym.times 1)$, we either permute the dimensions into $RR^(N_B #sym.times 1 #sym.times 1 #sym.times C)$, or squeeze out the last two dimensions, making it $RR^(N_B #sym.times C)$. In @code-se-block, we choose the former approach to implement a generic SE block that is used in our network.

#figure([
  #codly(languages: codly-languages, zebra-fill: none)
  ```python
  class SEBlock(nn.Module):
      def __init__(self, h: int, w: int, c: int, r: int):
          super(SEBlock, self).__init__()
          self.h, self.w, self.c, self.r = h, w, c, r
          self.fc1 = nn.Linear(c, c // r)
          self.fc2 = nn.Linear(c // r, c)

      def forward(self, x: torch.Tensor):
          k = F.avg_pool2d(x, (self.h, self.w))
          k = k.permute(0, 2, 3, 1)
          k = self.fc1(k)
          k = F.relu(k)
          k = self.fc2(k)
          k = F.sigmoid(k)
          k = k.permute(0, 3, 1, 2)
          k = F.interpolate(k, (self.h, self.w), mode="nearest")
          return k * x
  ```],
  caption: [
    Implementation of generic SE blocks in PyTorch.
  ]
) <code-se-block>

== Network architecture
@fig-structure shows the outline of our final network architecture. First, we add one more unpadded convolutional layer to the model to extract more high-level features. Then, we use SE-ResNet structure as proposed in #cite(label("DBLP:journals/corr/abs-1709-01507")) to recalibrate these feature maps. Each SE-ResNet block consists of a zero-padded convolutional layer activated by ReLU units #cite(label("DBLP:journals/corr/abs-1803-08375")), an SE block that performs the recalibration, and a skip connection for residual learning. The final features are aggregated and summarized in the same approach as the original handout code.

= Results
We train both our network and the handout baseline network for 30 epochs, recording the negative log-likelihood loss (NLL loss) and accuracy of each epoch, as displayed in @fig-result-loss and @fig-result-acc. We observe that while our network converges at a similar speed to baseline, our model achieves significantly higher accuracy and lower loss compared to baseline model when plateaued.

#figure(
  image("assets/result_loss.png"),
  caption: [
    Loss curve of our network.
  ]
) <fig-result-loss>

#figure(
  image("assets/result_acc.png"),
  caption: [
    Accuracy curve of our network.
  ]
) <fig-result-acc>

We measure run time of both scripts for a total of 30 epochs. This measurement involves training time, evaluation time, and time spent on other auxiliary operations. We present the timing result alongside performance metrics stated above in @table-result-comp, from which we can see that our model outperforms the baseline at a comparatively low time overhead.

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(stroke: 0.08em),
    table.header([Network], [NLL loss], [Accuracy], [Training time (s)]),
    table.hline(stroke: 0.05em),
    [Baseline], [#num(0.020182653331756593, round: (mode: "places", precision: (6)))], [#num(0.994)], [*#num(278.0604258, round: (mode: "places", precision: (3)))*],
    [Ours], [*#num(0.0161860077857971, round: (mode: "places", precision: (6)))*], [*#num(0.9956)*], [#num(712.96537, round: (mode: "places", precision: (3)))],
    table.hline(stroke: 0.08em)
  ),
  caption: [
    Performance metrics of networks at epoch 30. Winner of each metric is bolded.
  ]
) <table-result-comp>

== Ablation study
We evaluate the effectiveness of structures introduced into the network in our experiments by ablating modules then comparing the loss of these ablated networks. As depicted in @fig-ablation-loss, an all-ablated network performs the worst in terms of convergence speed and plateaued loss. When ablated of SE blocks only, the network suffers from a high plateaued loss similar to that of a double-ablated network. When ablated of residual connections only, the plateaued loss of network is not affected, and the model even _benefits_ from this change as it converges faster. Surprisingly, if we compare a no-SE network and a no-SE-, no-Res network, we do observe that skip connections aid in helping the network converge.

#figure(
  image("assets/ablation_loss.png"),
  caption: [
    Loss curve of networks ablated of different modules.
  ]
) <fig-ablation-loss>

#bibliography("works.bib")
