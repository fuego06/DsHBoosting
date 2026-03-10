# HBoost-2e2boosting-


The repository contains everything about the HBoost(2e2boosting) model

Node classification on real-world graphs represents a challenging task due to both heterophily,
namely when connected nodes have different labels, and optimization issues, such as oversmoothing
and instability in deep message passing. In this work, we propose DsHBoosting,
a deep supervised end-to-end boosted structurally hierarchical GNN that targets heterophilous
graphs by learning a stage-wise tradeoff between masked structural adjacency and a cosine-based
Top-k KNN graph constructed in the learned feature space. The model follows a unified
Stagewise Training Scheme (STS) that combines (i) internal stage-wise residual logit refinement
across T stages, (ii) deep supervision via auxiliary softmax heads on intermediate stages, and
(iii) an external node reweighting loop across R rounds that upweights hard training nodes only.
In addition, we introduce a nodewise structural hierarchy driven by graph priors (depth and
community consistency) that modulates propagation and residual refinement through hierarchy-aware
gating and attention. Experiments on 9 benchmark datasets spanning both homophilous
and heterophilous settings show that DsHBoosting achieves strong and stable performance and
yields clear gains on highly heterophilous graphs. Ablation studies further confirm that STS
and KNN–adjacency blending are the primary contributors, while structural hierarchy provides
complementary benefits on graphs with strong structural organization.
Keywords: Graph Neural Networks; Heterophily; Boosting; Graph Structure Learning; Node
Classification



#### The Hierarchy in the Proposed Model
<img width="375" height="340" alt="heatmap_good3" src="https://github.com/user-attachments/assets/041c1977-3b5b-4301-9d36-0cc244364d50" />
<img width="375" height="540" alt="heatmap_good" src="https://github.com/user-attachments/assets/fbfeb46c-f41d-43b2-903c-43dbb84408e9" />
<img width="375" height="557" alt="structure" src="https://github.com/user-attachments/assets/997e8e39-6917-47a1-ab08-ffd154b23707" />
<img width="375" height="540" alt="boosting_arc.png" src="https://github.com/user-attachments/assets/d95373c5-e555-4cc6-8444-29ad7cae246f" />

```latex

\chapter{Methodology}

\section{Preliminaries}
In this section, we introduce the problem setup and describe the notations. Secondly, we explain the graph priors used in the model. Finally, the feature transformation method used before training the model is illustrated.


\subsection{Problem Setup and Notation}
We consider a standard transductive semi-supervised node classification task on a single attributed graph. We report a transductive regime; Let $\mathcal{V}_{\text{train}}, \mathcal{V}_{\text{val}}, \mathcal{V}_{\text{test}} \subseteq V$ denote the training, validation, and test node sets, respectively.

\begin{equation}
G = (V, E, X, \mathbf{y}),
\end{equation}

where $V$ is the set of nodes in the graph, $E$ is the set of undirected edges describing the relations among nodes, $X$ is the node feature matrix, where each row corresponds to the feature vector of a node, and $\mathbf{y}$ is the node label vector, containing the class labels for the subset of annotated nodes.

More precisely, assume
\begin{equation}
 X \in \mathbb{R}^{N \times F},
 \qquad
 \mathbf{y} \in \{0,\ldots,K-1\}^N,
\end{equation}
where $N = |V|$ denotes the number of nodes and $F$ is the number of features per node. The structural component $(V,E)$ is coded by the adjacency matrix
\begin{equation}
 A \in \{0,1\}^{N \times N}.
\end{equation}
The model assumes a simple undirected graph with
\begin{equation}
 A_{ij} = A_{ji},
 \qquad
 A_{ii} = 0,
 \quad
 \forall i,j \in \{1,\ldots,N\}.
\end{equation}

Consider also a depth vector, $\mathbf{D} \in [0,1]^N$ and a community-based centrality vector $\mathbf{C} \in [0,1]^N$ computed by using the full graph $A$. These graph priors are described in Section~\ref{subsec:preproc}. For simplicity, element-wise (Hadamard) products for matrices or vectors are denoted by the symbol ``$\odot$''.

The model uses a binary mask $M \in \{0,1\}^{N \times N}$, which controls which node pairs are allowed in the learned similarity graph:
\begin{equation}
 M_{ij} =
 \begin{cases}
 0, & \text{edge between } i \text{ and } j \text{ is hindered},\\[0.3ex]
 1, & \text{edge between } i \text{ and } j \text{ is allowed}.
 \end{cases}
\end{equation}
In all stages, the learned similarity operator is multiplied element-wise by $M$ so that forbidden edges never appear. We apply $M$ to both the structural adjacency and the cosine-KNN operator in the chosen evaluation regime.

In standard Graph Neural Networks (GNNs), node representations
are updated by aggregating information from neighboring nodes.
Formally, a propagation operator $\mathcal{P}$ defines how node
features are combined using the graph structure.

In classical GCN-style models, this operator is typically based on
a normalized adjacency matrix $\tilde{A}$ and performs local
aggregation of 1-hop neighbor features.



Let $h_i \in \mathbb{R}^d$ denote the learned representation
of node $i$ after propagation.

The classifier produces a logit vector
\[
z_i = W h_i,
\]
where $W \in \mathbb{R}^{C \times d}$ is a learnable weight matrix.
The vector $z_i \in \mathbb{R}^C$ contains the raw, unnormalized
class scores (logits) for node $i$.

Predicted class probabilities are obtained by applying the softmax function:
\[
\hat{y}_i = \text{softmax}(z_i).
\]

In this thesis, the term "logits" refers to the raw class scores
$z_i$ before applying softmax.



Training is performed by minimizing a classification loss over the labeled nodes:
\[
\mathcal{L} = \sum_{i \in V_L} \ell(\hat{y}_i, y_i),
\]
where $\ell(\cdot)$ denotes the cross-entropy loss
(or focal loss when applied).


\subsection{Preprocessing and Graph Priors}
\label{subsec:preproc}

In this subsection, we introduce feature engineering applied to node features and graph-based priors used in the model. All benchmark graph datasets used in the paper are available in the \texttt{torch\_geometric.datasets} library \cite{51}. The library provides a numeric feature matrix $X \in \mathbb{R}^{N \times F}$ for each dataset and contains term frequencies, according to bag-of-words representation. Before training the model, we applied the TF-IDF \cite{37} style reweighting. For each feature f, we compute its document
frequency, $df_f$ , defined as the number of nodes in which $f$
appears. If a feature is seen in fewer nodes, a larger weight is assigned to that feature. Let N be the number of nodes.
The TF-IDF style reweight for feature $f$ is

\begin{equation}
 w_f = \log\!\left( \frac{N + 1}{\mathrm{df}_f + 1} \right) + 1.
\end{equation}
Every feature value at a node is multiplied by its own weight, resulting in a reweighted feature matrix $\widehat{X}$ with
\begin{equation}
 \widehat{X}_{i,f} = w_f \, X_{i,f}.
\end{equation}

Secondly, every node’s feature vector is normalized to have a unit Euclidean norm:
\begin{equation}
 \big\| \widehat{X}_{i,:} \big\|_2 = 1,
 \qquad
 \forall i \in \{1,\ldots,N\},
\end{equation}
which avoids scale imbalances across nodes.
Third, dimensionality reduction is applied via Principal Component Analysis (PCA) \cite{38} and the number of retained components can be found at Section~\ref{subsec:exp set}. Crucially, the PCA transformation is conducted only on the training nodes (i.e., $i \in \mathcal{V}_{\text{train}}$) and then applied to all nodes to avoid any feature-space leakage from validation or test nodes into the training procedure. Denoting the PCA mapping fitted on training nodes by $\mathrm{PCA}_{\mathrm{train}}(\cdot)$, we obtain reduced features
\begin{equation}
 \widetilde{X}_{i,:}
 =
 \mathrm{PCA}_{\mathrm{train}}\big(\widehat{X}_{i,:}\big),
 \quad \forall i.
\end{equation}

This process, TF-IDF-style reweighting, $\ell_2$-normalization, and PCA fitted only on training nodes, is used in all benchmark datasets (Cora, CiteSeer, Chameleon, etc.). For the dataset WikiCS \cite{23}, richer structural augmentation is achieved by using multi-hop structural augmentation \cite{40} to show that the model fits well in different feature settings.

Graph-based priors used in the model as input variables are computed as follows. Let $\mathcal{N}(i)$ denote the neighbor set of node $i$ in the undirected graph $A$ and let $c_i$ be community label of node $i$ obtained from a community detection algorithm. In this paper, we used the Louvain algorithm \cite{41}. The community-based centrality prior $C_i$ is defined by
\begin{equation}
 C_i =
 \begin{cases}
 \displaystyle
 \frac{1}{|\mathcal{N}(i)|}
 \sum_{j \in \mathcal{N}(i)} \mathbf{1}\{c_j = c_i\},
 & |\mathcal{N}(i)| > 0, \\[2ex]
 0, & |\mathcal{N}(i)| = 0,
 \end{cases}
 \label{eq:hier_C}
\end{equation}
where $\mathbf{1}\{\cdot\}$ is the indicator function and $C_i \in [0,1]$ measures how well node $i$ agrees with its local community.

Let $e_i$ show the eccentricity of node $i$, and define $e_{\min}$ as minimum $e_i$, $e_{\max}$ as maximum $e_i$. $D^{depth}_i$ is defined as follows:
\begin{equation}
 D_i^{\text{depth}} =
 \begin{cases}
 \displaystyle
 \frac{e_i - e_{\min}}{e_{\max} - e_{\min}},
 & e_{\max} > e_{\min}, \\[2ex]
 0, & e_{\max} = e_{\min},
 \end{cases}
 \label{eq:hier_D}
\end{equation}
so that nodes close to the structural core have smaller $D_i^{\text{depth}}$, while peripheral nodes have larger values. Both priors are computed purely from topology (no labels) on the full graph, consistent with the transductive protocol. 

\section{Proposed Method}
This section provides a detailed illustration of the suggested methodology. Firstly, the model layers are introduced, and how they interact to form a hierarchy-aware boosting architecture is explained. Secondly, the trade-off between the feature-driven cosine KNN graph and masked structural adjacency in the propagation operator is shown. The deep supervision idea of the model is driven by the goal of making each intermediate stage a valid predictor and ensuring stable gradient flow so that stage-wise residual refinements are learned effectively. The boosting architecture of the model, the loss function used for the training, and how the model prevents leakage are discussed. Finally, relationship between model components is explained.

\begin{figure}[t]
\centering
\includegraphics[width=\FigW]{overview.png}
\caption{Overview of the proposed DsHBoosting framework and the Stagewise Training Scheme (STS): masked structural adjacency and cosine Top-$k$ KNN are blended at each stage, hierarchy signals modulate propagation and refinement, intermediate heads provide deep supervision, and an external reweighting loop upweights hard training nodes across rounds.}
\label{fig:overview_dshboosting}
\end{figure}
\FloatBarrier

Figure~\ref{fig:overview_dshboosting} summarizes the complete workflow of the proposed method. The model starts from the input graph structure $A$ and node features $X$, and constructs a feature-based cosine Top-$k$ KNN neighborhood that complements the observed edges. At each internal STS stage, a learnable trade-off blends the masked adjacency with the KNN operator to form the stage-specific propagation view, while hierarchy information derived from graph priors modulates how strongly nodes aggregate and how aggressively logits are refined. The stage outputs are supervised through auxiliary softmax heads to stabilize training and make intermediate stages predictive. Finally, an external reweighting loop across rounds increases the influence of hard training nodes so that later optimization focuses on ambiguous and heterophilous regions.

\textbf{Big picture (one pass of STS):}

In our unified Stagewise Training Scheme (STS), the model is trained end-to-end while combining:
(i) internal stage-wise residual logit refinement across $T$ stages,
(ii) deep supervision through auxiliary softmax heads on intermediate stages,
and (iii) an external node reweighting loop across $R$ rounds that upweights hard training nodes only.
At each stage $t$, the model constructs a blended propagation operator $B^{(t)}$ (from masked adjacency and cosine KNN), computes hierarchy signals, performs hierarchy-aware message passing, produces a residual logit correction $\Delta^{(t)}$, and refines the accumulated logits $F^{(t)}$ by a learnable shrinkage.


\noindent\textbf{STS stage summary.}
For $t=1,\dots,T$ each stage, it first constructs a propagation operator that determines how node features are aggregated from neighbors in each stage-specific blended adjacency matrix:
\[
\text{build } S^{(t)} \;\to\; \text{blend } B^{(t)}.
\]
Then it performs hierarchy-aware refinement and updates the logits:
\[
\text{hierarchy-gated message passing} \;\to\; \Delta^{(t)} \;\to\; F^{(t)}.
\]

For $t=1,\dots,T$, we (i) build a cosine Top-$k$ graph $S^{(t)}$ and blend it with masked adjacency to form $B^{(t)}$,
then (ii) perform hierarchy-gated propagation to obtain a residual update $\Delta^{(t)}$,
and (iii) update logits to $F^{(t)}$. Across boosting rounds $r=1,\dots,R$, the training loss reweights only training nodes using $w_i^{(r)}$.


\section{Model Layers and Structure}
\label{subsec:model_layers}
In this section, how the main layers are designed, how
they work, and how they help each other to create a structural hierarchy and depth-aware architecture is presented. Throughout this section, let
\begin{equation}
Z \in \mathbb{R}^{N \times d}
\label{eq:Z_def}
\end{equation}
denote the current node embedding matrix at a given stage, and let
$Z_i \in \mathbb{R}^d$ be the embedding of node $i$.

\subsection{Row-normalized adjacency: AdjRowNorm}
The \emph{AdjRowNorm} layer produces a row-stochastic matrix where each
row sums to one. Given an adjacency-like operator
$A \in \mathbb{R}^{N \times N}$, the normalized matrix
$\widetilde{A}$ is
\begin{equation}
\widetilde{A}_{ij}
=
\frac{A_{ij}}{\sum_{k=1}^{N} A_{ik} + \varepsilon},
\qquad \varepsilon > 0,
\label{eq:adjrownorm}
\end{equation}
so that each row of $\widetilde{A}$ sums approximately to one. This
layer is crucial for stabilizing the message passing and normalizing
the learned similarity operators so that they can be interpreted as
probability kernels.

\subsection{Cosine KNN Graph}
The \emph{CosineTopKNN} layer is a feature-based topological layer. It
constructs a feature-driven graph on top of the current node embeddings.
At each stage, the embedding of each node is taken, and the cosine
similarities are computed with all other nodes:
\begin{equation}
S_{ij}
=
\frac{\langle Z_i, Z_j \rangle}
{\|Z_i\|_2\,\|Z_j\|_2},
\qquad
\forall i,j \in \{1,\ldots,N\}.
\label{eq:cosine_sim}
\end{equation}
For each node $i$, the top-$k$, which is a fixed hyperparameter that determines how many feature-similar neighbors are retained per node at each stage, most similar neighbors are kept; denote
this set by $\mathcal{N}_k(i)$. A row-wise softmax is applied so that
their weights sum to one:
\begin{equation}
\widetilde{S}_{ij}
=
\frac{\exp(S_{ij}/\tau)\,\mathbf{I}\{j \in \mathcal{N}_k(i)\}}
{\sum_{m \in \mathcal{N}_k(i)} \exp(S_{im}/\tau)},
\label{eq:topk_softmax}
\end{equation}
where $\tau > 0$ is a temperature parameter. The KNN graph is dynamic:
as embeddings are updated during training, the similarity graph also
changes. On heterophilous datasets, the feature-driven KNN graph can connect nodes that are not adjacent in the original topology, potentially improving label consistency.

\subsection{Learnable adjacency blending: BlendAdjAlpha}
The \emph{BlendAdjAlpha} layer is used for combining the original
adjacency and the KNN-based similarity graph into a single propagation
operator. Both components are row-normalized (see \eqref{eq:adjrownorm}). A learned scalar
parameter $\alpha \in (0,1)$ determines the relative weight:
\begin{equation}
B = \alpha\,\widetilde{A} + (1 - \alpha)\,\widetilde{S}.
\label{eq:blend_basic}
\end{equation}
If $\alpha \approx 1$, the model has confidence in the original graph
and effectively ignores the KNN graph. If $\alpha \approx 0$, the model
relies heavily on the learned similarity structure. The blending
parameter is learned end-to-end and shared across nodes at each stage
of the model. In later sections, we show how this learnable adjacency
blending and the cosine KNN graph layer create a trade-off between
original topology and feature-based similarity.

\subsection{Parent Bias}
The \emph{ParentBias} layer determines one scalar weight per node
based on the squared norm of its current embedding. We empirically test whether this signal aligns with uncertainty (entropy/max-probability relation). A typical form is
\begin{equation}
\beta_i
=
\sigma\big( \gamma \,\|Z_i\|_2^2 \big),
\label{eq:parentbias}
\end{equation}
where $\gamma>0$ is a learnable scale and $\sigma(\cdot)$ is the
logistic function. This value is then added into the attention logits. This term can bias attention toward nodes whose embeddings have larger norm; we evaluate its empirical relation to uncertainty in Section~\ref{pbs_rlp}. Thus, $ParentBias$ serves as a learnable, embedding-dependent source-importance term in the attention mechanism.

\subsection{Hierarchy Gate: depth-aware node gate}
The \emph{Hierarchy Gate} component takes three types of inputs for
each node: preprocessed node features $\widetilde{X}$, a scalar centrality score
$C_i$, and a normalized depth score $D_i$. These inputs are passed
through a small neural network to produce a gate value.
\begin{equation}
g_i = g(X_i, C_i, D_i) \in (0,1).
\label{eq:hiergate_simple}
\end{equation}
The gate $g_i$ is computed from ($X_i,C_i,D_i$). In hierarchy validation, we report correlations between $g_i$ and the priors to verify that the learned gate is consistent with these structural signals.

\subsection{DendroELU: depth-modulated activation}
The \emph{DendroELU} layer modulates nonlinear activations using the node depth prior by using the ELU activation function \cite{42}.
For node $i$, the depth-modulated activation is
\begin{equation}
\varphi(Z_i, D_i)
=
\mathrm{ELU}(Z_i)\,\exp(-\lambda D_i),
\label{eq:dendroelu}
\end{equation}
where $\lambda \in \mathbb{R}$ is a learnable scalar and $D_i \in [0,1]$ is the normalized depth.
Because $\lambda$ is not constrained to be positive in the implementation, this term can either \emph{dampen} peripheral nodes when $\lambda>0$ or \emph{amplify} them when $\lambda<0$; the sign and magnitude are learned from data under the training objective.


\subsection{Hierarchy-Aware Attention (depth interaction term)}
The hierarchy-aware attention layer forms attention logits by combining a feature interaction term, a structural prior from the blended operator, a depth-difference term, and a node-importance (ParentBias) term.
A stage-wise attention logit of the form used in our implementation is
\begin{equation}
\ell_{ij}
=
\phi(Z_i, Z_j)
+
\log B_{ij}
+
\lambda_D \bigl|D_i^{\text{depth}} - D_j^{\text{depth}}\bigr|
+
\beta_j,
\label{eq:hier_attention_logits_impl}
\end{equation}
where $\phi(\cdot,\cdot)$ is a learnable feature interaction, $\beta_j$ is the ParentBias term, and $\lambda_D\in\mathbb{R}$ is a learnable scalar coefficient.
Since $\lambda_D$ is not constrained in the implementation, the model can learn to either \emph{penalize} cross-depth interactions (when $\lambda_D<0$) or \emph{encourage} them (when $\lambda_D>0$), depending on what best reduces the training loss.

\subsection{DeepHierarchyBlockV2: gated residual refinement with ELU mixing}
The \emph{DeepHierarchyBlockV2} refines node representations using a gate $g_i\in(0,1)$ (output of the HierarchyGate) that controls the strength and form of the residual update.
Let $\Delta_i^{(\ell)}$ denote the pre-activation update at internal layer $\ell$ (after affine mapping, normalization, and dropout).
The block computes a gated nonlinear update by mixing two ELU-transformed paths:
\begin{equation}
u_i^{(\ell)}
=
g_i\,\mathrm{ELU}\!\big(\Delta_i^{(\ell)}\big)
+
(1-g_i)\,\mathrm{ELU}\!\big(\alpha\,\Delta_i^{(\ell)}\big),
\qquad \alpha\in(0,1),
\label{eq:deepblock_gated_elu}
\end{equation}
and applies it as a residual refinement:
\begin{equation}
h_i^{(\ell+1)} = h_i^{(\ell)} + u_i^{(\ell)}.
\label{eq:deepblock_residual}
\end{equation}
Thus, high-gate nodes receive a stronger direct nonlinear update, while low-gate nodes follow a more conservative scaled path, improving stability on unreliable or noisy nodes.

\subsection{ClusterGate: channel gating based on prototypes}
In the latent space, prototype-driven channel gating is carried out by
the \emph{ClusterGate} layer. After learning a limited number of
prototypes, it calculates each node's soft assignment to these
prototypes and transforms them into per-channel gate values that
either selectively amplify or suppress feature dimensions.

Let $p_i \in \mathbb{R}^K$ be the soft assignment vector of node $i$
over $K$ prototypes. Here $K$ illustrates the number of learned prototypes. We set $K$ equal to the number of classes each data has in our experiments. The per-channel gate vector is
\begin{equation}
\mathbf{g}_i^{\text{chan}}
=
\sigma\big( p_i W_g + \mathbf{b}_g \big),
\label{eq:clustergate_vec}
\end{equation}
and the gated embedding is
\begin{equation}
Z'_i = \mathbf{g}_i^{\text{chan}} \odot Z_i,
\label{eq:clustergate_apply}
\end{equation}
where $W_g$ and $\mathbf{b}_g$ are learnable parameters and
$\sigma(\cdot)$ acts element-wise. This produces sharper, more
class-discriminative node representations prior to the boosting stage
by emphasizing channels that are consistent with a node's most likely
structural/semantic pattern and downweighting irrelevant or noisy
channels.

\begin{table}[t]
\centering
\caption{Layer-level summary of DsHBoosting and how each component supports STS.}
\label{tab:method_layer_summary}
\TableStd
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{p{0.22\linewidth} p{0.30\linewidth} p{0.23\linewidth} p{0.23\linewidth}}
\toprule
\textbf{Component} & \textbf{Role in the model} & \textbf{Main parameters} & \textbf{Connected STS element} \\
\midrule
AdjRowNorm & Row-normalize adjacency-like operators & $\varepsilon$ & Stabilizes propagation inside each stage \\
CosineTopKNN & Build feature-driven Top-$k$ similarity graph $S^{(t)}$ & $k, \tau$ & Heterophily handling inside each stage \\
BlendAdjAlpha & Blend masked adjacency and KNN similarity into $B^{(t)}$ & $\alpha_t$ & Stage-wise learned trade-off ($t$-dependent) \\
HierarchyGate & Node-wise gate from priors $(C_i,D_i)$ and features & $g_i$ network & Controls stage update strength per node \\
ParentBias & Node-wise importance term in attention logits & $\gamma$ & Biases message sources within stage \\
HierAttnBlend & Attention aggregation using $B^{(t)}$, depth, ParentBias & $\lambda_D$ (depth term) & Defines message mixing in each stage \\
DendroELU & Depth-modulated activation & $\lambda$ & Controls smoothing across hierarchy \\
DeepHierarchyBlockV2 & Gated residual refinement block & depth, dropout & Core hierarchical refinement in each stage \\
LearnableShrink & Shrinkage $s_t$ applied to $\Delta^{(t)}$ & $s_t$ & Internal stage scaling inside STS \\
AuxSoftmax heads & Intermediate softmax predictions & $\lambda_t$ weights & Deep supervision inside STS \\
External reweighting & Hard-node upweighting across rounds & $\tau_h, \gamma$ & External loop inside STS \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
\FloatBarrier

\section{Theory: Hierarchy in the Proposed Model}
\label{subsec:theory_hierarchy}
In this subsection, how community and depth priors are used to create a graph-induced node hierarchy and how this hierarchy affects the model's attention and residual updates are formally explained. In addition to that, the meaning of the model hierarchy is discussed. The key idea is that each node receives continuous scores that quantify its \emph{community consistency}, \emph{core-periphery depth}, and \emph{learned reliability}, and these scores globally modulate message passing without coarsening the graph.

\subsection{Construction of hierarchy scores}
Given the preprocessed node embedding $z_i$, the model forms a
\emph{hierarchy gate} $g_i \in (0,1)$ by combining features,
centrality, and depth through a small neural network:
\begin{equation}
g_i = \sigma\!\big(
\mathbf{w}^{\top}
[\,z_i,\, C_i,\, D_i^{\text{depth}}\,]
+ b
\big),
\label{eq:hier_gate}
\end{equation}
where $\sigma(\cdot)$ is the logistic function, $\mathbf{w}$ and
$b$ are learnable parameters, and $[\,\cdot,\cdot,\cdot\,]$
denotes concatenation. Intuitively, nodes that are central,
community-consistent, and structurally reliable receive larger
$g_i$, while ambiguous or peripheral nodes receive smaller $g_i$.

\subsection{Use of hierarchy in attention and residual refinement}
The blended propagation operator $B$ (obtained by combining the
original adjacency and the feature-based KNN graph) defines a
structural prior on edges (see \eqref{eq:blend_basic}). The hierarchy-aware attention layer
uses $B$ depth and node importance to construct logits
$\ell_{ij}$ for messages from node $j$ to node $i$:
\begin{equation}
\ell_{ij}
=
\phi(z_i, z_j)
+
\log B_{ij}
+
\lambda_D
\bigl|
D_i^{\text{depth}} - D_j^{\text{depth}}
\bigr|
+
\beta_j,
\label{eq:hier_attention_logits}
\end{equation}
where $\phi$ encodes feature interactions, $\lambda_D \in \mathbb{R}$
controls depth penalization, and $\beta_j$ is the energy-based
ParentBias term for node $j$ (see \eqref{eq:parentbias}). The attention weights are then
\begin{equation}
\alpha_{ij}
=
\frac{\exp(\ell_{ij})\,\mathbf{1}\{B_{ij}>0\}}
{\sum_{m \in \mathcal{N}_B(i)} \exp(\ell_{im})},
\qquad
\mathcal{N}_B(i) = \{j \mid B_{ij}>0\},
\label{eq:attn_weights_masked}
\end{equation}
which favor edges that are (i) supported by $B_{ij}$, (ii)
hierarchically close in depth, and (iii) originating from
informative parents with high $\beta_j$.

Inside the DeepHierarchyBlock, the gate $g_i$ controls the
strength of residual feature updates. Let $h_i^{(\ell)}$ be the
representation of node $i$ at internal layer $\ell$ and
$\Delta_i^{(\ell)}$ the transformed update (after affine
mapping, nonlinearity, normalization, and dropout). The update
rule is
\begin{equation}
h_i^{(\ell+1)}= h_i^{(\ell)}+g_i\,\mathrm{ELU}\!\big(\Delta_i^{(\ell)}\big)
+(1-g_i)\,\mathrm{ELU}\!\big(\alpha\,\Delta_i^{(\ell)}\big)
\label{eq:hier_residual}
\end{equation}
where $\alpha \in (0,1)$ controls a conservative path. When
$g_i$ is large, the node receives a strong residual update;
when $g_i$ is small, the update is downscaled, stabilizing the
propagation on unreliable or noisy nodes.

\subsection{Distinct notion of hierarchy}
The notion of hierarchy in the proposed model is fundamentally
different from classical ``graph hierarchy'' in pooling or
multiscale GNNs, where the graph is recursively coarsened into
super-nodes and processed at multiple resolutions. Here, the
hierarchy is soft, node-wise, and purely structural:
each node is endowed with continuous scores
$\bigl(C_i, D_i^{\text{depth}}, g_i\bigr)$ that quantify its
community consistency, core--periphery depth, and learned
reliability, but the underlying node set and adjacency are never
coarsened. These hierarchical scores are reused throughout the
network---in attention
\eqref{eq:hier_attention_logits}--\eqref{eq:attn_weights_masked},
in residual refinement \eqref{eq:hier_residual}, and in the
boosting stages---so that ``core'' and ``peripheral'' nodes are
treated differently during the entire training process. As a
result, the architecture is hierarchy-aware without relying on
graph pooling, and is complementary to standard multiscale GNN
approaches.

\section{Trade-Off Between KNN and Structural Adjacency}
This subsection describes how training automatically steers KNN and structural adjacency balance under the learned structural hierarchy by analyzing the learned trade-off parameter $\alpha$ that balances the feature-driven cosine KNN graph and the masked structural adjacency in the propagation operator.

The propagation operator is an adaptive convex combination of the feature-driven
cosine KNN graph and the masked structural adjacency:
\begin{equation}
B(\alpha)
=
\alpha\,\widetilde{A}^{\text{mask}}
+
(1-\alpha)\,\widetilde{S},
\label{eq:blend_operator}
\end{equation}
where $\alpha \in (0,1)$ is a learnable scalar shared across nodes at each
boosting stage, and $\widetilde{A}^{\text{mask}}$ and $\widetilde{S}$ are
row-normalized (see \eqref{eq:adjrownorm} and \eqref{eq:topk_softmax}). Small $\alpha$ highlights the learned similarity structure
(i.e., the KNN graph), whereas large $\alpha$ highlights the original topology.

Let $\mathcal{L}(\alpha)$ denote the training loss and $G_B = \partial
\mathcal{L} / \partial B$ the gradient of the loss with respect to the blended
operator. By the chain rule,
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \alpha}
=
\Big\langle
G_B,\,
\widetilde{A}^{\text{mask}} - \widetilde{S}
\Big\rangle,
\label{eq:dL_dalpha}
\end{equation}
where $\langle \cdot,\cdot \rangle$ denotes the Frobenius inner product. When
this inner product is negative, gradient descent increases $\alpha$, pushing
the model towards the structural adjacency. On the other hand, optimization
shifts mass towards the KNN graph by decreasing $\alpha$ if this inner
product is positive. As a result, $\alpha$ is a data-driven knob that
automatically chooses the balance that minimizes the loss under the current
hierarchy, rather than a fixed hyperparameter tuned by hand.

\section{Deep Supervision over Internal STS Stages}
\label{subsec:deep_supervision}
In addition to the final prediction head, the model incorporates auxiliary classification heads to the intermediate boosting stages.
These stages correspond to the internal STS stages (stage-wise logit refinement within one forward pass), and should not be confused with the external boosting rounds.
Let \(F^{(t)} \in \mathbb{R}^{N \times K}\) denote the logits at boosting stage \(t \in \{1,\dots,T\}\), and
\begin{equation}
P^{(t)} = \mathrm{softmax}\!\big(F^{(t)}\big) \in [0,1]^{N \times K}
\label{eq:deep_softmax}
\end{equation}
the corresponding class-probability predictions. The final prediction uses \(P^{(T)}\), while the intermediate predictions \(\{P^{(t)}\}_{t=1}^{T-1}\) act as auxiliary outputs.

Given one-hot labels \(Y \in \{0,1\}^{N \times K}\) and a per-node classification loss \(\ell(\cdot,\cdot)\) (cross-entropy or focal), the deep supervision objective is a weighted sum of the losses at all stages:
\begin{equation}
\mathcal{L}_{\mathrm{deep}}
= \ell\big(Y, P^{(T)}\big)
+ \sum_{t=1}^{T-1} \lambda_t \, \ell\big(Y, P^{(t)}\big),
\label{eq:deep_supervision_loss}
\end{equation}
where \(\lambda_t \ge 0\) are small auxiliary weights that downweight earlier stages relative to the final head.
In STS training, this objective is applied with the train mask and external boosting weights (round-dependent), i.e., losses are summed over training nodes and scaled by \(w_i^{(r)}\) in the external reweighting loop.
More explicitly, the per-node losses are aggregated as \(\sum_{t=1}^{T}\lambda_t\sum_{i=1}^{N} m_i\, w_i^{(r)}\,\ell\!\big(y_i, P_i^{(t)}\big)\), where \(m_i\) is the train mask.

This design exhibits its strengths in two complementary and mutually reinforcing aspects. First, strong gradient signals are sent directly to the early stages via~\eqref{eq:deep_supervision_loss}, each boosting stage is encouraged to be locally predictive rather than acting as an unconstrained feature transformer. This stabilizes optimization and gives the stage hierarchy (coarse-to-fine refinement across \(t\)) genuine semantic meaning. Second, it regularizes the hierarchy: intermediate representations must already align with the graph structure and labels under the current KNN--adjacency blend, so subsequent stages focus on correcting difficult or ambiguous nodes instead of repairing arbitrary distortions. This design can improve optimization stability and encourage the intermediate stages to be predictive. We empirically evaluate its impact via ablations and stage-wise diagnostics.

\begin{figure}[!t]
\centering
\includegraphics[width=\FigW]{boosting_arc.png.jpeg}
\caption{Boosting architecture of DsHBoosting (internal stage-wise refinement and auxiliary heads as part of STS).}
\label{fig:boosting_arch}
\end{figure}
\FloatBarrier

Figure~\ref{fig:boosting_arch} visualizes the internal stage-wise refinement mechanism used in STS. The model forms an additive sequence of stage predictors, where each stage produces a residual correction to the accumulated logits rather than generating predictions from scratch. After each refinement step, an auxiliary softmax head provides deep supervision, ensuring that intermediate stages remain directly predictive and that gradients reach early parts of the network. This stage-wise design supports stable optimization and reduces reliance on a single deep message-passing stack, since refinement is distributed across stages. As a result, later stages focus on correcting the remaining errors (often concentrated on heterophilous or ambiguous nodes), while earlier stages learn coarser but reliable class separation under the current KNN--adjacency trade-off.

\section{Boosting and Loss Function}
\label{subsec:boosting_focal_leakage}
In this section, the model boosting scheme is illustrated. The loss function used for the model training is shown. More importantly, how model prevents from the information leakage is explained.

\subsection{Internal and External Boosting (STS)}
An additive ensemble of boosting stages forms the core network structure. 
Let $F^{(t)} \in \mathbb{R}^{N \times K}$ denote the logits after stage $t$, with
$F^{(0)}$ initialized to zero. 
At each stage the model predicts a residual correction $\Delta^{(t)}$ and updates
the global logits as in \eqref{eq:internal_boost}. This is the internal part of STS (stage-wise refinement with shared parameters).
\begin{equation}
F^{(t)} 
= F^{(t-1)} + s_t \,\Delta^{(t)}, 
\qquad t = 1,\dots,T,
\label{eq:internal_boost}
\end{equation}
where $s_t \in (0,1)$ is a learnable shrinkage parameter that controls the step size
of stage $t$. Early stages learn coarse, hierarchy-aware corrections, while later
stages only refine the remaining errors. This yields a graph-aware analogue of
gradient boosting: instead of learning a single monolithic transformation, the model
learns a stable additive sequence of residual experts, since each $\Delta^{(t)}$ is
gated by the hierarchy (via centrality and depth) and scaled by a learnable shrinkage factor that encourages incremental corrections.

On top of this internal stage-wise boosting, an outer boosting loop that
iteratively reweights training nodes across rounds is applied. This is the external part of STS (node reweighting across rounds).
Let $w_i^{(r)}$ be the weight of
node $i$ at boosting round $r$, and let $p_{i,y_i}^{(r)}$ be the predicted probability of
its true class. The hardness score is defined as:
\begin{equation}
h_i^{(r)} = 1 - p_{i,y_i}^{(r)},
\label{eq:hardness}
\end{equation}
and nodes are marked as ``hard'' whenever $h_i^{(r)} \ge \tau$ for a fixed threshold
$\tau \in (0,1)$. The weights are then updated as
\begin{equation}
w_i^{(r+1)} \propto 
w_i^{(r)} \big( 1 + (\gamma - 1)\,\mathbf{I}\{h_i^{(r)} \ge \tau\} \big),
\label{eq:weight_update}
\end{equation}
followed by normalization over training nodes so that 
$\sum_{i \in \mathcal{V}_{\text{train}}} w_i^{(r+1)} = 1$.
In other words, consistently misclassified or low-confidence nodes receive a 
multiplicative upweighting factor $\gamma > 1$, while "easy" nodes maintain roughly
the same influence. This external boosting mechanism forces the model to concentrate 
its capacity on structurally ambiguous, heterophilous, or otherwise challenging 
regions of the graph, while preserving a clean separation between training, 
validation, and test labels.
The external weights \(w_i^{(r)}\) multiply the deeply-supervised per-stage losses for training nodes (i.e., they apply to all auxiliary heads and the final head).
\subsection{Focal Loss}
To further highlight nodes that are unclear or confusing, the model uses a focal
classification loss instead of plain cross-entropy. For a node $i$ with one-hot
label vector $y_i \in \{0,1\}^{K}$ and predicted probabilities 
$p_i \in [0,1]^K$, the per-node focal loss is given in \eqref{eq:focal_loss}.
\begin{equation}
\ell_{\text{focal}}(y_i, p_i)
= - \sum_{c=1}^{K} y_{ic} \, (1 - p_{ic})^{\gamma} \log p_{ic},
\label{eq:focal_loss}
\end{equation}
where $\gamma \ge 0$ is the focusing parameter. 
When $p_{ic}$ is large (easy, confident predictions), the factor 
$(1 - p_{ic})^{\gamma}$ suppresses the contribution of that node. 
When $p_{ic}$ is small (hard, confused nodes), the loss is amplified.
During training, we provide the train mask and the external weights \(w_i^{(r)}\) as sample weights, and auxiliary heads are weighted across internal STS stages.

On graphs with heterophily, class imbalance, or label noise, this design is 
particularly advantageous. Combined with the external boosting weights in 
\eqref{eq:weight_update}, focal loss yields a two-level focusing mechanism: 
the outer loop reweights hard nodes in the sample space, while the focal term 
focuses gradient magnitude on hard nodes in the loss space. The resulting model 
naturally concentrates its representational strength on the most ambiguous parts 
of the hierarchy, without destabilizing training.
\label{subsec:boosting_focal_leakage}


\section{Coupling Between  Model Components}
In this section, we introduce the connection between the model components, STS, KNN-adjacency blending, and structural hierarchy. In addition, we show how the STS part works and the connections among the STS components.

\subsection{Gradient Decomposition and Coupling}
Let stage logits satisfy \eqref{eq:internal_boost} and let $P^{(t)}=\mathrm{softmax}(F^{(t)})$ as in \eqref{eq:deep_softmax}. The (deeply supervised) objective is defined in the following equation \eqref{eq:obj}, and is referenced throughout this subsection to avoid repetition:
\begin{equation}
\mathcal{L}=\sum_{t=1}^T \lambda_t \sum_{i=1}^N m_i w_i\,
\ell\!\big(y_i, P_i^{(t)}\big), 
\label{eq:obj}
\end{equation}
where $m_i$ is the train mask, $w_i$ external boosting weights, and $\lambda_t$ deep-supervision weights (with $\lambda_T=1$ for the final head and smaller $\lambda_t$ for $t<T$). During implementation, the loss is computed over nodes using the train mask value for node i, and the external boosting loop updates $w_i$ across rounds to focus training on hard nodes.

Assume
\begin{align}
\Delta^{(t)} &= \Delta^{(t)}\!\Big(X, A_{\text{blend}}^{(t)}, H, g, E^{(t)}\Big), \label{eq:delta_dep}\\
E^{(t)} &= \psi\!\big(P^{(t-1)}\big),\text{where $E^{(t)}$ is error signal derived from the previous stage prediction  } \nonumber\\ 
A_{\text{blend}}^{(t)} &= \alpha_t A^m + (1-\alpha_t)S^{(t)}, \label{eq:ablend}\\
A^m &= A\odot M,\qquad
S^{(t)}=\mathcal{K}\!\big(Z_0^{(t)};\tau\big). \nonumber
\end{align}
Then, for any parameter $\theta$ appearing in the stages,
\begin{equation}
\nabla_\theta \mathcal{L}
=\sum_{t=1}^T \lambda_t\sum_{i=1}^N m_i w_i\,
\Big(\tfrac{\partial \ell}{\partial F_i^{(t)}}\Big)
\Big(\tfrac{\partial F_i^{(t)}}{\partial \theta}\Big), 
\label{eq:grad_fact}
\end{equation}
and STS$\leftrightarrow$hierarchy$\leftrightarrow$KNN coupling holds because
$\tfrac{\partial F^{(t)}}{\partial \theta}$ contains the chained routes
\begin{align}
F^{(t-1)} &\to P^{(t-1)} \to E^{(t)} \to \Delta^{(t)} \to F^{(t)}, \label{eq:path_sts}\\
Z_0^{(t)} &\to S^{(t)}(\tau) \to A_{\text{blend}}^{(t)} \to \Delta^{(t)} \to F^{(t)}, \label{eq:path_knn}\\
H &\to \Delta^{(t)} \to F^{(t)}. \label{eq:path_hier}
\end{align}
\noindent\textbf{Interpretation.}
$w_i$ and $\lambda_t$ scale gradients in \eqref{eq:grad_fact}, while $s_t$ rescales the stage update in \eqref{eq:internal_boost}.
The hierarchy signal $H$ and KNN temperature $\tau$ affect where gradients flow by reshaping $\Delta^{(t)}$
through \eqref{eq:delta_dep}--\eqref{eq:ablend}.

\begin{table}[t]
\centering
\caption{Main ingredients used in the stage update $\Delta^{(t)}$.}
\label{tab:stage_ingredients}
\TableStd
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{p{0.23\linewidth} p{0.74\linewidth}}
\toprule
\textbf{Symbol} & \textbf{Meaning} \\
\midrule
$X$ & Node feature matrix after preprocessing (TF--IDF, normalization, PCA). \\
$A^m = A \odot M$ & Masked structural adjacency used to prevent leakage. \\
$S^{(t)}$ & Cosine Top-$k$ similarity graph built from stage embeddings $Z_0^{(t)}$. \\
$k$ & Fixed hyperparameter controlling the number of retained feature-neighbors per node. \\
$\tau$ & Temperature parameter controlling softmax sharpness in the KNN weights. \\
$A_{\text{blend}}^{(t)}$ & Stage-specific propagation operator: $\alpha_t A^m + (1-\alpha_t)S^{(t)}$. \\
$\alpha_t$ & Learnable trade-off parameter between structural adjacency and KNN similarity. \\
$H$ & Structural hierarchy signals derived from graph priors (depth and community consistency). \\
$g$ & Node-wise hierarchy gate controlling update strength. \\
$E^{(t)}$ & Stage context derived from $P^{(t-1)}$ (uncertainty/confidence signal). \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
\FloatBarrier

\subsection{Gradient w.r.t.\ Hierarchy Signal $H$ (STS $\leftrightarrow$ Hierarchy Coupling).}
Let the objective be the deeply-supervised STS objective in \eqref{eq:obj}. Since the internal stage-wise refinement follows \eqref{eq:internal_boost} and the stage increment $\Delta^{(t)}$ depends on $H$ through the hierarchy-gated blocks, the chain rule gives
\begin{equation}
\frac{\partial \mathcal{L}}{\partial H}
=
\sum_{t=1}^T
\left(\frac{\partial \mathcal{L}}{\partial F^{(t)}}\right)
\left(\frac{\partial F^{(t)}}{\partial H}\right)
=
\sum_{t=1}^T
s_t\left(\frac{\partial \mathcal{L}}{\partial F^{(t)}}\right)
\frac{\partial \Delta^{(t)}}{\partial H}.
\label{eq:dL_dH_chain}
\end{equation}
Expanding node-wise weights from \eqref{eq:obj} yields
\begin{equation}
\frac{\partial \mathcal{L}}{\partial H}
=
\sum_{t=1}^T
\lambda_t\, s_t
\sum_{i=1}^N m_i w_i
\left(\frac{\partial \ell(y_i,P_i^{(t)})}{\partial F_i^{(t)}}\right)
\frac{\partial \Delta_i^{(t)}}{\partial H},
\label{eq:dL_dH_weighted}
\end{equation}
which makes explicit how deep supervision $\lambda_t$, external boosting $w_i$, masking $m_i$,
and internal scaling $s_t$ jointly modulate hierarchy gradients.

\subsection{Gradient w.r.t.\ KNN Temperature $\tau$ under Top-$k$ Softmax Weights.}
The KNN layer computes cosine similarities from normalized features as in \eqref{eq:cosine_sim} and then applies a hard Top-$k$ selection followed by temperature-softmax weights as in \eqref{eq:topk_softmax}. On any region where the Top-$k$ set $\mathcal{N}_k(i)$ is unchanged, gradients flow through the temperature-softmax weights within the selected set, while index changes are non-smooth and yield a piecewise-defined derivative.
Since the blended adjacency satisfies \eqref{eq:ablend}, we have
\begin{equation}
\frac{\partial A_{\mathrm{blend}}^{(t)}}{\partial \tau}
=
(1-\alpha_t)\frac{\partial \widetilde{S}^{(t)}}{\partial \tau}.
\label{eq:dAblend_dTau}
\end{equation}
Combining \eqref{eq:obj}, \eqref{eq:internal_boost}, and \eqref{eq:dAblend_dTau}, the end-to-end gradient becomes
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \tau}
=
\sum_{t=1}^T
\lambda_t\, s_t\,
\left\langle
\frac{\partial \ell}{\partial F^{(t)}},
\frac{\partial \Delta^{(t)}}{\partial A_{\mathrm{blend}}^{(t)}}
(1-\alpha_t)\frac{\partial \widetilde{S}^{(t)}}{\partial \tau}
\right\rangle.
\label{eq:dL_dTau}
\end{equation}

\subsection{Interaction within STS}
For node $i$ and stage $t$, the logit-gradient is scaled by mask, external boosting, and deep supervision (from \eqref{eq:obj}):
\begin{equation}
\frac{\partial \mathcal{L}}{\partial F_i^{(t)}}
=
\lambda_t\, m_i w_i\,
\frac{\partial \ell(y_i,P_i^{(t)})}{\partial F_i^{(t)}}.
\label{eq:dL_dF_it}
\end{equation}
Since the internal refinement follows \eqref{eq:internal_boost}, the gradient to the stage increment satisfies
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \Delta_i^{(t)}}
=
s_t\,\frac{\partial \mathcal{L}}{\partial F_i^{(t)}}
=
\lambda_t\, m_i w_i\, s_t\,
\frac{\partial \ell}{\partial F_i^{(t)}}.
\label{eq:dL_dDelta_it}
\end{equation}
Finally, the sensitivity of the objective to the internal stage scale is
\begin{equation}
\frac{\partial \mathcal{L}}{\partial s_t}
=
\left\langle
\frac{\partial \mathcal{L}}{\partial F^{(t)}},\, \Delta^{(t)}
\right\rangle,
\label{eq:dL_ds}
\end{equation}
so $s_t$ increases when the current stage update aligns with the negative loss gradient.

This analysis clarifies that STS, KNN blending, and structural hierarchy are not independent modules but are tightly coupled through shared gradient pathways. As a result, removing one component alters both gradient magnitude and gradient routing, which motivates the ablation study in Section V. 

\begin{algorithm}[t]
\caption{Training DsHBoosting with STS (end-to-end), deep supervision, and external boosting}
\label{alg:dshboost_train_clean}
\begin{algorithmic}[1]
\Require Graph $G=(V,E,X,y)$ with $N=|V|$ nodes; adjacency $A$; splits $(\mathcal{V}_{\mathrm{tr}},\mathcal{V}_{\mathrm{va}},\mathcal{V}_{\mathrm{te}})$.
\Require Structural priors $(C,D)$ and leakage mask $M$; \#stages $T$; \#rounds $R$; max epochs per round $E_{\mathrm{round}}$.
\Require Hardness threshold $\tau_h$, upweight factor $\gamma>1$; deep-supervision weights $\{\lambda_t\}_{t=1}^T$; loss $\ell$ (CE or focal).
\Ensure Trained parameters $\Theta$ and test metrics.

\State \textbf{Leakage prevention:} $A^m \gets A\odot M$; $\widetilde{A^m}\gets\mathrm{AdjRowNorm}(A^m)$.
\State \textbf{Hierarchy signal:} $(H,g) \gets \mathcal{G}(X,A^m,C,D)$.
\State \textbf{Init external weights:} $w_i^{(1)}\gets \frac{1}{|\mathcal{V}_{\mathrm{tr}}|}$ if $i\in\mathcal{V}_{\mathrm{tr}}$, else $0$.

\For{$r=1,\dots,R$} \Comment{external boosting rounds}

\State \textbf{Train end-to-end (one fit call):}
\State Train model parameters $\Theta$ for up to $E_{\mathrm{round}}$ epochs with early stopping based on validation loss / ReduceLROnPlateau on $\mathcal{V}_{\mathrm{va}}$ using the masked, reweighted deep-supervision loss
\State $\displaystyle
\mathcal{L}\gets \sum_{t=1}^{T}\lambda_t \sum_{i=1}^{N} m_i\,w_i^{(r)}\,\ell\!\left(y_i,P_i^{(t)}\right),
\quad
m_i=\mathbb{I}[i\in\mathcal{V}_{\mathrm{tr}}].
$

\State \textbf{(Inside forward pass, STS stages $t=1..T$):}
\State Initialize logits by model design: $F^{(0)}\gets 0$.
\For{$t=1,\dots,T$} \Comment{internal STS stages (architectural)}
\State $Z_0^{(t)} \gets f_{\mathrm{enc}}\!\left(X, F^{(t-1)}\right)$.
\State $S^{(t)} \gets \mathrm{CosineTopKNN}\!\left(Z_0^{(t)};k,\tau,M\right)$; $\widetilde{S^{(t)}}\gets \mathrm{AdjRowNorm}(S^{(t)})$.
\State $B^{(t)} \gets \alpha_t\,\widetilde{A^m} + (1-\alpha_t)\,\widetilde{S^{(t)}}$.
\State $\Delta^{(t)} \gets \Phi\!\left(Z_0^{(t)},B^{(t)},H,g\right)$.
\State $F^{(t)} \gets F^{(t-1)} + s_t\,\Delta^{(t)}$; \quad $P^{(t)}\gets \softmax(F^{(t)})$.
\EndFor

\State \textbf{External reweighting (train nodes only):}
\State Compute $P^{(T)}$; for $i\in\mathcal{V}_{\mathrm{tr}}$ set $h_i\gets 1-P^{(T)}_{i,y_i}$.
\State $w_i^{(r+1)} \gets w_i^{(r)}\Big(1+(\gamma-1)\,\mathbb{I}[h_i\ge\tau_h]\Big)$ for $i\in\mathcal{V}_{\mathrm{tr}}$.
\State $w_i^{(r+1)}\gets 0$ for $i\notin\mathcal{V}_{\mathrm{tr}}$; normalize $\sum_i w_i^{(r+1)}=1$.

\EndFor

\State \textbf{Evaluation:} compute $P^{(T)}$ on full graph and report metrics on $\mathcal{V}_{\mathrm{te}}$.
\end{algorithmic}
\end{algorithm}


\chapter{Experiments and Results}

In this chapter, we present our experimental setups and model performance comparison with respect to benchmark models. Secondly, we show how our structural hierarchy relates to the graph priors. Then, we evaluate whether oversmoothing occurred. Moreover, we introduce how our model avoids the information leakage phenomenon of transductive semi-supervised learning. We showed the stability of the ParentBias and its relation to prediction uncertainty. Finally, we finish this chapter by stating an ablation study of our model design.

\section{Datasets}
\label{chp:Datasets}
In this work, we used nine different real-world graph benchmark datasets for transductive semi-supervised node classification. Each dataset is represented as a single attributed graph $G=(V,E,X,y)$, where nodes correspond to entities (papers, webpages, actors), edges encode relations (citations, hyperlinks, co-occurrences, co-purchases), node features are high-dimensional descriptors, and labels represent node categories.

\FloatBarrier
\begin{table}[H]
\centering
\caption{Summary of benchmark datasets used in experiments.}
\label{tab:datasets}
\TableStd
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lrrrrrl}
\toprule
\textbf{Dataset} & \textbf{\#Nodes} & \textbf{\#Edges} & \textbf{\#Features} & \textbf{\#Classes} & \textbf{Split} & \textbf{Notes} \\
\midrule
Cora & 2,708 & 5,429 & 1,433 & 7 & 60/20/20 & Homophilous citation network \\
CiteSeer & 3,327 & 4,732 & 3,703 & 6 & 60/20/20 & Homophilous citation network \\
Chameleon & 2,277 & 36,101 & 2,325 & 5 & 60/20/20 & WikipediaNetwork (heterophilous) \\
Squirrel & 5,201 & 217,073 & 2,089 & 5 & 60/20/20 & WikipediaNetwork (heterophilous) \\
Texas & 183 & 309 & 1,703 & 5 & 60/20/20 & WebKB (heterophilous) \\
Cornell & 183 & 295 & 1,703 & 5 & 60/20/20 & WebKB (heterophilous) \\
Wisconsin & 251 & 499 & 1,703 & 5 & 60/20/20 & WebKB (heterophilous) \\
Actor & 7,600 & 30,019 & 932 & 5 & 60/20/20 & Actor co-occurrence network (heterophilous) \\
WikiCS & 11,701 & 216,123 & 300 & 10 & 60/20/20 & Wikipedia CS articles \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}

\FloatBarrier


\subsection{Citation Networks}
The citation network datasets we used in this study are Cora \cite{1} and CiteSeer. These datasets represent homophilous citation networks.

In the Cora dataset, nodes represent research papers, and edges represent citations. It contains 7 classes, 2708 nodes, 1433 node features, and 5429 edges. It is considered homophilous, since linked papers tend to belong to similar research areas.

Another citation network used in this work is the CiteSeer dataset. In this dataset each node represents a scientific document, and edges represent citation relations. It is also considered homophilous since citations often connect papers from related topics. It contains 3327 nodes, 4732 edges, 3703 node features, and 6 classes. 

\subsection{Wikipedia-Based Graphs}

The Chameleon dataset, the Squirrel dataset, and the WikiCS \cite{23} dataset are used in this research as Wikipedia-based graphs. 

The Chameleon dataset is a Wikipedia page network in which nodes represent Wikipedia articles and edges correspond to hyperlink connections between pages. It contains 2277 nodes, 36101 edges, 2325 node features, and 5 classes. It shows strong heterophily, since the connected pages are mostly belong to different classes. Traditional message-passing graph neural networks, which rely on homophily assumptions, might work poorly under the heterophily datasets since neighbor aggregation mixes representations from nodes with different labels, causing oversmoothing and propagating misleading signals instead of reinforcing class-consistent information \cite{22,23}.

The Squirrel dataset is another Wikipedia page graph which constructed in a similar way to Chameleon, where nodes show articles and edges show hyperlinks. It contains 5201 nodes, 217073 edges, 2089 node features, and 5 classes. It is known as highly heterophilous graph. It gives a challenge for a benchmark to evaluate models designed to handle heterophily and noisy neighborhood aggregation due to its size and structural properties.

WikiCS is a Wikipedia-based network of computer science articles. Nodes correspond to articles related to computer science topics, and edges show hyperlink relationships. WikiCS contains lower-dimensional node features 300, and a larger number of nodes 11701 compared to the Chameleon and Squirrel datasets. It has 10 classes. It creates a diverse structural setting for evaluating graph neural networks on real-world hyperlink data.

\subsection{WebKB Page Networks}

The WebKB datasets, which are Cornell, Texas, and Wisconsin \cite{23}, are webpage networks collected from university computer science departments. The nodes in these datasets show individual web pages, and edges correspond to hyperlink connections between pages within the same university domain. These datasets are relatively small with respect to other datasets used in this work. They have (183,183,251) nodes accordingly. These datasets are known as strongly heterophilous graphs. These datasets are useful to understand the robustness and stability of graph neural networks under small-sample and structurally noisy conditions.

\subsection{Actor Co-occurrence Network}

The Actor dataset \cite{23} is a co-occurrence network built using film metadata. Nodes represent actors, and edges show co-occurrence relationships (e.g., actors cast in the same movie or production).

The co-occurrence relationships in the Actor dataset do not mean label similarity, which makes Actor a heterophilous graph. Even though it contains a large number of nodes compared to the WebKB datasets, it still exhibits challenging structural properties. Hence, in larger-scale settings, the Actor dataset serves as a benchmark for evaluating graph neural network models.

\section{Experimental Setup}
\label{subsec:exp set}
Table~\ref{tab:datasets} summarizes the benchmark datasets used in experiments, and Table~\ref{tab:perf} presents the summary of the model performance comparison. We can say that our evaluation suite spans both homophilous and heterophilous graphs with node counts ranging from 183 to 11,701 and feature dimensionalities from 300 to 3,703. 

As we discussed in Section~\ref{chp:Datasets} We used 9 real-world benchmark datasets to check our model performance comparison. We used a stratified sampling method \cite{47} for creating data splitting. The proportions of the train/validation/test are 60\%/20\%/20\%. All the datasets were downloaded from the Python \texttt{torch\_geometric.datasets} library \cite{51} then exported to our \texttt{text\_min} format (edges/features/labels) and reused identically by the R/Keras and PyTorch pipelines. The model was constructed and trained in R by using TensorFlow and Keras libraries \cite{48},\cite{49}. The performance of the proposal has been compared with established state-of-the-art methods, namely one-layer MLP \cite{27}, GCN \cite{19}, GAT \cite{20}, GCNII \cite{50}, FAGCN \cite{25}, MixHop \cite{29}, LINKX \cite{26}.

 
All competitor results reported in this thesis are produced by running the baseline models within our own training pipelines.
For each dataset, we reuse the same $20$ stratified splits (60/20/20) that we created, stored as index files
(\texttt{train\_idx.txt}, \texttt{val\_idx.txt}, \texttt{test\_idx.txt}) under \texttt{seed\_01}, \dots, \texttt{seed\_20}.
This ensures that every method is evaluated on identical train/validation/test partitions.

For each competitor method, we followed the official implementations whenever publicly available. In the cases where official repositories were not accessible, we used PyTorch Geometric \cite{51} implementations or reimplemented the models according to their original papers and report exact settings in Appendix~\ref{sec:competitors}

For competitor hyperparameters, we adopt settings recommended in the corresponding papers or official implementations when available; otherwise we use the default settings of the respective PyTorch Geometric modules or our fixed configurations (Appendix~\ref{sec:competitors}).

To avoid information leakage, TF--IDF reweighting, row-wise $\ell_2$ normalization, and PCA/SVD dimensionality reduction are fitted using training nodes only (per split), and the learned transform is then applied to validation and test nodes. We explain this process in Section~\ref{sec:leakageprev}. 

All methods are evaluated over the $20$ splits and we report mean test accuracy and standard deviation.
Model selection is performed using the validation set only.
For PyTorch baselines (MixHop, LINKX, FAGCN) we apply early stopping on validation accuracy: we keep the checkpoint with the best validation accuracy and report its corresponding test accuracy. 
R/Keras baselines are trained for a fixed number of epochs (no early stopping). Full baseline hyperparameter settings are reported in Appendix~\ref{sec:competitors}.


All \textsc{DsHBoosting} results are obtained using the same  $20$ stratified splits are employed for the competitor baselines.
For each seed, the model is trained on $\mathcal{V}_{\mathrm{train}}$, model selection is performed on $\mathcal{V}_{\mathrm{val}}$, and the final performance is computed once on $\mathcal{V}_{\mathrm{test}}$.
We report mean test accuracy and standard deviation over the 20 splits.


Feature preprocessing (TF--IDF reweighting, row-wise $\ell_2$ normalization, and PCA/SVD) is fitted exclusively on training nodes for each split and then applied to validation and test nodes.
The PCA dimension is determined per seed as
$d_{\mathrm{PCA}}=\min(d_{\max},|\mathcal{V}_{\mathrm{train}}|-1)$, where $d_{\max}$ is dataset-dependent (Table~\ref{tab:hyperparams}), which prevents degenerate PCA when the training set is small.

The model is trained for $R$ external boosting rounds, each consisting of $E_r$ epochs, resulting in a maximum of $R \cdot E_r$ epochs.
Early stopping is applied based on validation loss with dataset-specific patience.
We additionally employ a ReduceLROnPlateau schedule: when validation loss does not improve for a fixed number of epochs, the learning rate is reduced by a factor of $0.5$, down to a minimum of $10^{-5}$.

The adjacency blending coefficient is initialized as $\alpha_0$ (Table~\ref{tab:hyperparams}).
Internal stage-wise residual updates are initialized using predefined shrink factors for each stage.
Unless explicitly stated otherwise, hyperparameters are fixed per dataset and held constant across the 20 seeds.

Full dataset-specific configurations (rounds, epochs per round, patience values, learning-rate scheduling parameters, shrink factors, and focal loss settings) are reported in Appendix~\ref{app:dsh_details}.

The hyperparameters of the model are tuned by using stratified 5-fold cross-validation \cite{70,71,72} on labeled nodes, selecting the setting that minimizes mean validation loss across folds. For each fold, we trained inductively \cite{67,68,69} on the training subgraph using the same boosting-style training schedule and evaluated on the fold's validation subgraph. The details can be found in Appendix~\ref{appx:hyperparameter}


\FloatBarrier
\begin{table}[H]
\centering
\caption{Summary of model performance comparison: Mean test accuracy $\pm$ standard deviation across 20 random seeds.}
\label{tab:perf}
\TableStd
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccc}
\toprule
\textbf{Model} & \textbf{Cora} & \textbf{CiteSeer} & \textbf{WikiCS} & \textbf{Actor} & \textbf{Squirrel} & \textbf{Chameleon} & \textbf{Wisconsin} & \textbf{Cornell} & \textbf{Texas} \\
\midrule
MLP & 80.280 $\pm$ 0.015 & 80.750 $\pm$ 0.015 & 73.300 $\pm$ 0.090 & 36.150 $\pm$ 0.013 & 32.680 $\pm$ 0.010 & 45.810 $\pm$ 0.020 & 80.470 $\pm$ 0.042 & 72.120 $\pm$ 0.024 & 76.300 $\pm$ 0.045 \\
GCN-Proxy & 67.530 $\pm$ 0.017 & 78.600 $\pm$ 0.015 & 47.290 $\pm$ 0.009 & 33.970 $\pm$ 0.007 & 32.450 $\pm$ 0.001 & 39.950 $\pm$ 0.021 & 77.920 $\pm$ 0.040 & 72.000 $\pm$ 0.038 & 75.210 $\pm$ 0.041 \\
GAT-Proxy & 80.010 $\pm$ 0.015 & 80.330 $\pm$ 0.014 & 73.730 $\pm$ 0.008 & 35.900 $\pm$ 0.011 & 32.850 $\pm$ 0.001 & 45.960 $\pm$ 0.018 & 78.010 $\pm$ 0.040 & 70.000 $\pm$ 0.034 & 73.800 $\pm$ 0.048 \\
GCNII-Proxy & 60.520 $\pm$ 0.069 & 78.920 $\pm$ 0.014 & 51.840 $\pm$ 0.007 & 36.260 $\pm$ 0.008 & 20.000 $\pm$ 0.001 & 38.840 $\pm$ 0.030 & 79.710 $\pm$ 0.050 & 73.370 $\pm$ 0.030 & 79.130 $\pm$ 0.043 \\
FAGCN & 88.510 $\pm$ 0.010 & 92.640 $\pm$ 0.010 & 73.040 $\pm$ 0.010 & 35.650 $\pm$ 0.011 & 36.810 $\pm$ 0.030 & 57.320 $\pm$ 0.024 & 63.580 $\pm$ 0.061 & 52.870 $\pm$ 0.057 & 86.060 $\pm$ 0.060 \\
LINKX & 84.120 $\pm$ 0.014 & 88.010 $\pm$ 0.010 & 82.370 $\pm$ 0.007 & 29.050 $\pm$ 0.011 & 65.010 $\pm$ 0.020 & 73.250 $\pm$ 0.021 & 71.600 $\pm$ 0.078 & 67.870 $\pm$ 0.061 & 76.950 $\pm$ 0.036 \\
MixHop & 80.270 $\pm$ 0.045 & 93.720 $\pm$ 0.006 & 34.480 $\pm$ 0.074 & 32.290 $\pm$ 0.012 & 44.450 $\pm$ 0.174 & 68.480 $\pm$ 0.019 & 74.990 $\pm$ 0.051 & 64.500 $\pm$ 0.080 & 79.670 $\pm$ 0.057 \\
DsHBoosting & 93.960 $\pm$ 0.012 & 94.340 $\pm$ 0.015 & 81.120 $\pm$ 0.010 & 40.140 $\pm$ 0.076 & 86.490 $\pm$ 0.067 & 80.000 $\pm$ 0.045 & 93.210 $\pm$ 0.054 & 93.000 $\pm$ 0.054 & 93.260 $\pm$ 0.071 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}


Table~\ref{tab:perf} illustrates that DsHBoosting achieves the best overall performance, ranking first on most benchmarks and delivering especially large gains on heterophilous graphs. It reports mean test accuracy ($\pm$ standard deviation across seeds) for DsHBoosting and competing baselines. The proposed model achieves the best accuracy on homophilous citation graphs (Cora, CiteSeer), showing that the added hierarchy-aware and stage-wise boosting components do not degrade performance when local smoothing is beneficial. More importantly, the largest gains appear on heterophilous datasets (Actor, Chameleon, Squirrel, and WebKB), where standard message passing models (GCN, GAT) exhibit low accuracy due to neighbor-label mismatch and oversmoothing. In contrast, DsHBoosting maintains high accuracy, which is consistent with the model’s ability to combine feature-driven similarity with controlled, hierarchy-conditioned updates. The reported accuracy variation across seeds further indicates that these improvements are stable rather than driven by a particular split. 



\section{Hierarchy Validation}

In this section we validate the structural hierarchy components of the model. We used 5 different seeds for the study, and we present the results of the Cornell dataset. We examine the association of the $ Hierarchy Gate$ value with respect to $Depth$, and $Centrality$ inputs. Under our design we hypothesize a (positive/negative) association and verify it across 5 seeds. 


\FloatBarrier
\begin{table}[H]
\centering
\caption{Mean $\pm$ standard deviation of Spearman correlations between (i) Depth and Gate, and (ii) Centrality and Gate across seeds for the Cornell dataset.}
\label{tab:cent-gate}
\TableStd
\begin{tabular}{cc}
\toprule
$\boldsymbol{\rho(\mathrm{Depth},\, g)}$ & $\boldsymbol{\rho(\mathrm{Centrality},\, g)}$ \\
\midrule
$-0.688 \pm 0.283$ & $-0.050 \pm 0.118$ \\
\bottomrule
\end{tabular}
\end{table}



According to the Table~\ref{tab:cent-gate}, the learned gate $g_i$ shows a stable negative Spearman correlation with depth; $D_i$ this means deeper nodes get smaller gate values. The correlation with centrality $C_i$ is near zero. Therefore, we can say that the hierarchy mechanism is active and primarily implements a depth-aware modulation on Cornell, which provides interpretable structural behavior. 




\begin{figure}[H]
\centering
\includegraphics[width=\FigWbig]{heatmap_good.jpeg}
\caption{Hierarchy validation on Cornell. Nodes are ordered by hierarchical clustering of the final embeddings $Z_{\text{final}}$ using cosine distance (average linkage). Rows show the depth prior $D_i$, the centrality prior $C_i$, and the learned hierarchy gate $g_i$. The structured bands indicate that $g_i$ varies consistently across embedding-consistent node groups, supporting that the hierarchy module injects meaningful structural bias rather than acting as an arbitrary modulation.}
\label{fig:hierarchy_validation_two}
\end{figure}
\FloatBarrier



Figure ~\ref{fig:hierarchy_validation_two} illustrates how the learned hierarchy gate is related to the structural priors after the model has formed as a final representation. We used hierarchical clustering (hclust) \cite{52} by using the cosine distance of the final embeddings $Z_{final}$. The dendrogram groups together nodes that the model considers similar in representation space. By observing consistent block patterns in the gate row, we can conclude that $g_i$ is not random but instead varies systematically across groups of nodes with similar embeddings. This suggests that the hierarchy mechanism learns a structured modulation that aligns with the representation geometry and incorporates meaningful structural priors.

\section{Oversmoothing Evaluation}
\label{subsec:Oversmoothin Evaluation}
Oversmoothing occurs when the graph neural network has large propagation depth. In our case the effective depth is supervised by the boosting rounds and neighborhood information mix. The learned blend parameter $\alpha_t$ and the $k-NN$ construction controls the aggression of the information mix with respect to nodes at each stage. The val/test performance can be degraded even if training loss continues to improve, and the reason for that is repeatedly emphasized neighborhood aggregation over node-specific features. 

To determine oversmoothing, we extract the stage-wise node embeddings and average cosine similarity across all node pairs, and dispersion-to-mean $\mathbb{E}\|z_i-\bar z\|_2$ (collapse
in spread). We also report Dirichlet energy (graph smoothness),
\begin{equation}
E(Z)=\frac{1}{2|E|}\sum_{(i,j)\in E}\|z_i-z_j\|_2^2,
\end{equation}
where smaller $E(Z)$ indicates that embeddings vary less across edges. Classical oversmoothing is characterized by
cosine similarity increasing toward 1, dispersion decreasing
toward 0, and Dirichlet energy decreasing toward 0 as
depth/stage increases. We used  $DeepBlock$ embedding after (DeepHierarchyBlock) which shows whether hierarchy-gated refinement spread diversity or collapsed the representation, and $\texttt{z0\_ln}$ (early-stage embedding) is the early-stage embedding at the stage $t$ and tells how quickly the raw stage encoder collapses. All results in this analysis are reported over 5 random seeds (Squirrel). 

\FloatBarrier
\begin{figure}[H]
\centering
\begin{subfigure}[t]{0.323\textwidth}
\centering
\includegraphics[width=\linewidth]{DE_Deep.pdf}
\caption{Dirichlet E (DeepBlock)}
\label{fig:os_de_deep}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.323\textwidth}
\centering
\includegraphics[width=\linewidth]{cosine_Deep.pdf}
\caption{Mean cosine (DeepBlock)}
\label{fig:os_cos_deep}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.323\textwidth}
\centering
\includegraphics[width=\linewidth]{disp_Deep.pdf}
\caption{Dispersion (DeepBlock)}
\label{fig:os_disp_deep}
\end{subfigure}
\vspace{4pt}
\begin{subfigure}[t]{0.323\textwidth}
\centering
\includegraphics[width=\linewidth]{DE_z0.pdf}
\caption{Dirichlet E ($z0\_\mathrm{ln}$)}
\label{fig:os_de_z0}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.323\textwidth}
\centering
\includegraphics[width=\linewidth]{cosin_z0.pdf}
\caption{Mean cosine ($z0\_\mathrm{ln}$)}
\label{fig:os_cos_z0}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.323\textwidth}
\centering
\includegraphics[width=\linewidth]{disp_z0.pdf}
\caption{Dispersion ($z0\_\mathrm{ln}$)}
\label{fig:os_disp_z0}
\end{subfigure}
\caption{Oversmoothing diagnostics across stages (Squirrel).
Dirichlet energy measures graph smoothness; mean cosine similarity and dispersion quantify representation collapse across stages.}
\label{fig:oversmoothing_2x3}
\end{figure}
\FloatBarrier

By looking at Figure ~\ref{fig:oversmoothing_2x3}, we can say that none of the embeddings shows a strong sign of oversmoothing. The mean cosine similarity does not approach 1, and the Dirichlet energy does not drastically decrease toward 0. The dispersion plots show the moderate reduction in dispersion with respect to the stage index which means increased smoothing. However, this reduction does not imply that the model has a systematic cosine blow-up at the $DeepBlock$ output. Overall, we can say that the model applies controlled smoothing across stages. 


\section{Leakage Prevention}
\label{sec:leakageprev}
Because structure, features, and labels co-exist within a single graph, transductive node classification is particularly susceptible to subtle forms of information leakage. The proposed pipeline explicitly mitigates leakage at three levels:

\paragraph*{(i) Training-only feature transforms}
PCA, TF--IDF weighting, and related feature preprocessing steps are fitted 
\emph{only} on training nodes. Let $X_{\text{train}}$ denote the feature matrix
restricted to the training set. All statistics---document frequencies, IDF weights,
centering vectors, and PCA loadings---are computed from $X_{\text{train}}$ alone.
The resulting linear transform is then applied to validation and test nodes, but
their features never influence these parameters. This removes any feature-space
leakage from validation or test nodes back into the training pipeline. 
\paragraph*{(ii) Label-free structural priors}
The centrality prior $C$ and depth prior $D$ are computed from the unlabeled
graph structure $(V,E)$ only. Both $C_i$ (community-based centrality) and $D_i$
(normalized depth/eccentricity) depend solely on the topology and community 
structure, not on node labels. Consequently, these priors are fully unsupervised 
and can safely be computed on the full graph without introducing test labels into 
the model.

\paragraph*{(iii) Masked similarity and supervised objective}
The learned cosine KNN graph is always multiplied element-wise by the binary mask
$M$, which specifies which node pairs are allowed to form similarity edges in the
current training regime. This guarantees that forbidden pairs never affect the
learned propagation operator. Furthermore, all supervised losses, deep supervision
terms, and boosting weights are computed only on labeled training nodes;
validation labels are used solely for model selection (e.g., early stopping and
learning-rate scheduling), and test labels are never accessed during training.

By combining training-only feature transforms, label-free structural priors, and an explicit edge mask on the learned similarity graph, the methodology ensures that
performance improvements genuinely arise from improved hierarchical modeling rather than from any form of transductive leakage.


\section{Parent Bias Stability and Its Relation to Prediction Uncertainty}
\label{pbs_rlp}

In this section, we show the stability of the $ParentBias$ and its relation to prediction uncertainty. To analyze this, we used 5 seeds, and we present the analysis result of the Squirrel and Texas datasets. We created the node-wise \text{ParentBias (PB)}, the diagnostic PBraw signal, and the predicted probabilities over each $STS$ stage. We compute uncertainty measures: entropy, maximum class probability and prediction correctness. Then we report Spearman correlations between PB (and PBraw) and these uncertainty measures. We present results to confirm if the behavior is consistent across both heterophilous and small-graph settings.

\FloatBarrier
\begin{table}[H]
\centering
\caption{Mean $\pm$ standard deviation of Spearman correlations between ParentBias (PB) and prediction uncertainty ($H$ entropy and $\max p$, maximum class probability) across seeds.}
\label{tab:pb_uncertainty_summary_onecol}
\TableStd
\begin{tabular}{l c r r}
\toprule
\textbf{Dataset} & \textbf{Stage} & $\boldsymbol{\rho(\text{PB}, H)}$ & $\boldsymbol{\rho(\text{PB}, \max p)}$ \\
\midrule
Squirrel & 1 & \phantom{-}0.027 $\pm$ 0.019 & \phantom{-}0.002 $\pm$ 0.014 \\
Squirrel & 2 & \phantom{-}0.036 $\pm$ 0.030 & -0.021 $\pm$ 0.015 \\
Squirrel & 3 & \phantom{-}0.037 $\pm$ 0.030 & -0.021 $\pm$ 0.017 \\
\midrule
Texas & 1 & \phantom{-}0.015 $\pm$ 0.217 & -0.019 $\pm$ 0.218 \\
Texas & 2 & -0.031 $\pm$ 0.226 & \phantom{-}0.020 $\pm$ 0.226 \\
Texas & 3 & -0.057 $\pm$ 0.121 & \phantom{-}0.046 $\pm$ 0.120 \\
\bottomrule
\end{tabular}
\end{table}




By looking at Table~\ref{tab:pb_uncertainty_summary_onecol} we can say that for the Squirrel dataset, we have a small but consistent pattern for PB where higher PB is associated with higher entropy and lower maximum class probability. This indicates that PB tends to increase when predictions are more uncertain, suggesting PB captures an uncertainty-aligned signal on the Squirrel dataset. On the contrary, Texas exhibits highly variable correlations across seeds with large standard deviations and sign changes. Therefore, we can say that the the PB-uncertainty relationship is not stable. This instability likely comes from the small graph size, where split-dependent variance dominates correlation estimates. Overall PB is more interpretable and reliable as an uncertainty proxy on larger graphs than on very small graphs. 

\section{Stage-wise Diagnostics: Learned Mixing and Depth Interaction}
\label{sec:diag_alpha_lambda}

We report diagnostics extracted from trained DsHBoosting models to validate the
intended stage-wise behavior of (i) the learned adjacency mixing coefficient
$\alpha_t$ in the blended propagation operator and (ii) the depth-interaction
coefficient $\lambda_D$ in the hierarchy-aware attention logits.
Unless stated otherwise, gray lines show individual seeds and the thick black
line shows the mean across seeds.

\subsection{Learned adjacency mixing per stage ($\alpha_t$)}
\label{subsec:diag_alpha}

Recall that at each internal STS stage we use the blended operator
$B^{(t)}=\alpha_t \widetilde{A}^{\text{mask}}+(1-\alpha_t)\widetilde{S}^{(t)}$.
Thus, larger $\alpha_t$ indicates greater reliance on the masked structural
adjacency (topology), while smaller $\alpha_t$ indicates greater reliance on the
feature-driven cosine Top-$k$ similarity graph.

\FloatBarrier
\begin{figure*}[t]
\centering
\begin{subfigure}{0.49\linewidth}
 \centering
 \includegraphics[width=\linewidth]{alpha_plot.png}
 \caption{Cora}
 \label{fig:alpha_cora}
\end{subfigure}\hfill
\begin{subfigure}{0.49\linewidth}
 \centering
 \includegraphics[width=\linewidth]{alpha_plotwins.png}
 \caption{Wisconsin}
 \label{fig:alpha_wisconsin}
\end{subfigure}
\caption{Learned adjacency mixing coefficient $\alpha_t$ per internal STS stage.
Gray lines denote individual seeds and the black line denotes the mean across seeds.}
\label{fig:alpha_per_stage_both}
\end{figure*}
\FloatBarrier

On both datasets, $\alpha_t$ is relatively stable across seeds and shows a mild
stage-wise drift. On Cora, the mean $\alpha_t$ increases slightly with stage index,
suggesting that later stages place somewhat more emphasis on masked structural adjacency.
On Wisconsin, the mean $\alpha_t$ also shows a modest upward trend, but with stronger
seed-to-seed variation, indicating that the optimal topology--KNN balance is more
dataset- and split-dependent in the heterophilous WebKB setting.

\subsection{Learned depth-interaction coefficient per stage ($\lambda_D$)}
\label{subsec:diag_lambdaD}

In the hierarchy-aware attention logits, the implementation contains a depth-difference term
of the form $+\lambda_D\,|D_i-D_j|$ (see Eq.~\eqref{eq:hier_attention_logits_impl}).
Because $\lambda_D$ is \emph{unconstrained} in the code, it can be positive or negative:
$\lambda_D>0$ encourages cross-depth interactions, while $\lambda_D<0$ penalizes them.
Therefore, the correct interpretation is based on the sign and magnitude learned on each dataset.

\FloatBarrier
\begin{figure*}[t]
\centering
\begin{subfigure}{0.49\linewidth}
 \centering
 \includegraphics[width=\linewidth]{lambdaD_plot.png}
 \caption{Cora}
 \label{fig:lambdaD_cora}
\end{subfigure}\hfill
\begin{subfigure}{0.49\linewidth}
 \centering
 \includegraphics[width=\linewidth]{lambda_plotwins.png}
 \caption{Wisconsin}
 \label{fig:lambdaD_wisconsin}
\end{subfigure}
\caption{Learned depth-interaction coefficient $\lambda_D$ per internal STS stage.
Gray lines denote individual seeds and the black line denotes the mean across seeds.
The dashed horizontal line indicates $\lambda_D=0$.}
\label{fig:lambdaD_per_stage_both}
\end{figure*}
\FloatBarrier


On Cora, $\lambda_D$ is consistently positive across stages and seeds, with a clear
decay toward smaller values in later stages. This indicates that depth-difference has
its strongest influence early in STS and becomes less influential later, where attention is
increasingly governed by the blended adjacency prior $\log B_{ij}$, feature interactions,
and the ParentBias term.

On Wisconsin, $\lambda_D$ exhibits substantially larger variance across seeds and can cross
zero (some runs learn $\lambda_D<0$). This implies that in the WebKB regime the model does not
consistently prefer cross-depth interactions: depending on the split, training may either
encourage cross-depth mixing (positive $\lambda_D$) or suppress it (negative $\lambda_D$).
Accordingly, we avoid a single-direction claim for Wisconsin and interpret $\lambda_D$ as a
dataset-dependent learned knob controlling whether depth separation is promoted or reduced.
\section{Ablation Study}

In this section, we present an ablation study of the model. We split our ablation study into three parts. (i) \textit{STS ablation}, (ii) \textit{KNN-adjacency blend ablation}, and (iii) \textit{structural hierarchy ablation}. 

\FloatBarrier
\begin{table}[H]
\centering
\caption{Graph statistics and feature similarity summary (mean over nodes).}
\label{tab:graph_stats}
\TableStd
\begin{tabular}{lrrrrr}
\toprule
\textbf{Dataset} & \textbf{Mean Depth} & \textbf{Mean Centrality} & \textbf{Louvain Score} & \textbf{Heterophily} & \textbf{Mean CosSim} \\
\midrule
Cora & 0.701 & 0.0010 & 0.905 & 0.528 & 0.085 \\
CiteSeer & 0.858 & 0.0005 & 0.984 & 0.118 & 0.075 \\
WikiCS & 0.792 & 0.0030 & 0.724 & 0.859 & 0.779 \\
Actor & 0.697 & 0.0009 & 0.789 & 0.770 & 0.175 \\
Chameleon & 0.749 & 0.0120 & 0.901 & 0.799 & 0.007 \\
Squirrel & 0.743 & 0.0140 & 0.787 & 0.800 & 0.017 \\
Cornell & 0.626 & 0.0160 & 0.886 & 0.803 & 0.298 \\
Wisconsin & 0.646 & 0.0140 & 0.844 & 0.645 & 0.338 \\
Texas & 0.640 & 0.0160 & 0.828 & 0.740 & 0.339 \\
\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:graph_stats} summarizes structural and feature-based properties of the benchmark graphs.
The heterophily column shows that Cora and CiteSeer are comparatively less heterophilous (0.528 and 0.118), while most remaining datasets are strongly heterophilous (e.g., WikiCS 0.859, Cornell 0.803, Chameleon 0.799, Squirrel 0.800, Actor 0.770).
This indicates that for these graphs, edges frequently connect nodes from different classes, and purely neighbor-averaging message passing can mix conflicting label signals.

The mean cosine similarity (Mean CosSim) provides a feature-geometry view: Chameleon (0.007) and Squirrel (0.017) have extremely low average feature similarity, implying that even feature vectors of nodes in the graph are weakly aligned on average; therefore, a learned similarity neighborhood can be crucial to find informative neighbors beyond the raw adjacency.
Actor (0.175) also shows relatively low feature similarity, while WebKB graphs (Cornell/Wisconsin/Texas) have moderate cosine similarity (0.298/0.338/0.339), suggesting that features are more informative there and a KNN view may be helpful but less critical than on Wikipedia graphs like Chameleon and Squirrel.
WikiCS is a special case: despite high heterophily (0.859), it has very high cosine similarity (0.779), meaning that feature space is highly structured and can provide a strong alternative neighborhood even when structural edges are label-inconsistent.

The Louvain score is high for most datasets (often around 0.8--0.98), indicating pronounced community structure in the topology; in such cases, community-based priors (local consistency) can yield a meaningful hierarchy signal.
For example, CiteSeer has a very high Louvain score (0.984), suggesting strong community organization, which aligns with the observation that hierarchy-related components can be especially useful when community structure is pronounced.
Finally, the mean depth values are in a similar range across datasets (roughly 0.626--0.858), showing that all graphs contain a mix of core and peripheral nodes; this supports using depth-based priors to modulate propagation strength, where peripheral regions are treated more conservatively under heterophily.

\FloatBarrier
\begin{table}[H]
\centering
\caption{Ablation study of the model (mean test accuracy $\pm$ standard deviation across 20 random seeds).}
\label{tab:ablation}
\TableStd
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccc}
\toprule
\textbf{Model} & \textbf{Cora} & \textbf{CiteSeer} & \textbf{WikiCS} & \textbf{Actor} & \textbf{Squirrel} & \textbf{Chameleon} & \textbf{Wisconsin} & \textbf{Cornell} & \textbf{Texas} \\
\midrule
Full Model
& 93.960 $\pm$ 0.012 & 94.340 $\pm$ 0.015 & 81.120 $\pm$ 0.010 & 40.140 $\pm$ 0.076 & 86.490 $\pm$ 0.067 & 80.000 $\pm$ 0.045 & 93.210 $\pm$ 0.054 & 93.000 $\pm$ 0.054 & 93.260 $\pm$ 0.071 \\
STS Ablation
& 80.660 $\pm$ 0.025 & 74.120 $\pm$ 0.090 & 70.220 $\pm$ 0.030 & 32.940 $\pm$ 0.016 & 41.960 $\pm$ 0.014 & 42.430 $\pm$ 0.029 & 79.620 $\pm$ 0.061 & 86.370 $\pm$ 0.118 & 83.690 $\pm$ 0.086 \\
KNN+AB Ablation
& 89.660 $\pm$ 0.013 & 86.880 $\pm$ 0.010 & 79.110 $\pm$ 0.006 & 25.720 $\pm$ 0.004 & 73.460 $\pm$ 0.027 & 78.790 $\pm$ 0.023 & 93.490 $\pm$ 0.032 & 85.370 $\pm$ 0.071 & 94.340 $\pm$ 0.028 \\
Hierarchy Ablation
& 89.170 $\pm$ 0.015 & 69.660 $\pm$ 0.160 & 73.870 $\pm$ 0.019 & 35.080 $\pm$ 0.053 & 72.190 $\pm$ 0.048 & 47.500 $\pm$ 0.060 & 90.000 $\pm$ 0.047 & 91.450 $\pm$ 0.163 & 89.020 $\pm$ 0.047 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}



Firstly, we study the contribution of the proposed Stage-wise Training Strategy (STS); we remove STS components while keeping the base forward architecture as similar as possible. We removed (i) \textit{external reweighting (STS outer loop)} by fixing the training node weights to uniform and model trained single run, (ii) \textit{deep supervision (STS auxiliary heads)} by not using the auxiliary intermediate heads, and (iii) \textit{stage-wise refinement (STS internal stages)} via setting the number of the stages $T=1$. With this setting we remove the additive residual accumulation and eliminate stage-wise boosting. Our loss uses only the final prediction head. The hardness-based weight update across boosting rounds was not used. This ablation allow us to quantify how much of the performance gain is attributed of the STS mechanism. 

The second ablation is based on examine the contribution of the KNN-adjacency blending. In this study we removed the effect of the  $CosineTopKNN$ structure and blending operator. At every STS stage, the message-passing adjacency reduces to the leakage-mask structural adjacency, and no feature-based neighbors are used. All other components of STS remain unchanged.

The final ablation study is based on the structural hierarchy ablation. In this study we removed all hierarchy priors. The depth and centrality signals are set as constants. The node-wise hierarchy gate was transformed to the constant gate. Other layer components of the hierarchy building were removed from the model. All other components of STS remain unchanged.

\textbf{Effect of STS:} By looking at Table~\ref{tab:ablation}, we can say that the STS has the largest and most consistent impact. The accuracy of the every dataset is reduced after removing the STS. This indicates that stage-wise refinement and external-internal boosting are the main reasons for the performance gains. The performance drop is notably severe on heterophilous graphs such as Squirrel and Chameleon. These two datasets have high heterophily scores (0.8, 0.799) and very low mean cosine similarity (0.017, 0.007). In heterophily datasets, label propagation by using local neighbors is not stable. The STS algorithm is crucial for correcting hard-to-classify nodes and balance the learning under noisy neighborhood signals.

\textbf{Effect of KNN-adjacency blending (KNN-AB):} We can say that it is mostly helpful when graphs have low feature similarity. The datasets Actor,
Squirrel and partially Chameleon, which exhibit low feature similarity and high heterophily, are mostly affected. Table~\ref{tab:graph_stats} states that Actor/Squirrel/Chameleon have low mean cosine similarity (0.175, 0.017, 0.007) and high heterophily (0.77, 0.8, 0.799). The large drops of (e.g., Actor and Squirrel) can be explained as follows: blending the observed adjacency with a learned Top-$k$ cosine graph provides an alternative neighborhood signal that is more matched with feature geometry. The datasets with higher cosine similarity, e.g., Texas and Wisconsin, have about 0.339 and 0.338, respectively, and removing KNN-AB does not consistently reduce performance and can even slightly improve it. This suggests that when features are already informative, adding a learned similarity graph is not always necessary and may introduce noise, especially for the small graph datasets.


\textbf{Effect of structural hierarchy:} Table~\ref{tab:ablation} shows that it has the strongest effect when the graph has powerful community/structural organization. This ablation study shows a clear performance drop on CiteSeer. By looking at Table~\ref{tab:ablation} and Table~\ref{tab:graph_stats}, we can say that the drop in the accuracy comes from this dataset's structural organization, since Table~\ref{tab:graph_stats} shows that the CiteSeer dataset has high Louvain scores (0.984) with high mean depth (0.858). In addition, the accuracy drop of the Chameleon dataset reflects that the interaction between hierarchy gating and heterophily/low feature similarity, where hierarchy helps stabilize transformations even when local neighborhoods are noisy. The hierarchy-aware gating provides useful global structural bias beyond local message passing. Although the structural hierarchy is still beneficial for the highly heterophilous datasets, the contribution is smaller than STS and often complementary to KNN-AB. The reason is that heterophily primarily undermines the reliability of neighbors, rather than merely affecting hierarchical organization.

We notice some of the ablations yield small improvements for the datasets Texas and Wisconsin. This shows that the benefit of each component is data dependent and can vary with feature quality and graph structure.

Overall, Table~\ref{tab:ablation} and Table~\ref{tab:graph_stats} jointly support that \textbf{(i)} STS is the most essential component across all datasets. \textbf{(ii)} KNN-adjacency blending has the strongest influence when feature similarity is weak, and heterophily is high. \textbf{(iii)} Structural hierarchy contributes most on datasets with strong structural community signals, complementing STS rather than replacing it.




% =================================================
% CONCLUSIONS
% =================================================
\chapter{Conclusion and Future Work}

\section{Summary of Findings}
In this work we proposed \textsc{DsHBoosting}, a semi-supervised
node classification model that explicitly targets heterophilous
graphs. The model combines a feature-adaptive Top-k cosine similarity
graph that is learned by node embeddings at each stage,  with masked structural adjacency through a stagewise
learned blending parameter. We train the model in
a unified \textsc{Stagewise Training Scheme (STS)}. Moreover, we
introduced a node-wise structural hierarchy built from graph
priors in order to modulate attention, propagation strength, and residual updates.

The experiments confirm that \textsc{\textsc{DsHBoosting}} achieves strong
performance across both homophilous and heterophilous
regimes. The ablation study illustrates that STS is generally the most important determinant of the prediction accuracy. KNN-adjacency blending
is helpful mostly when feature similarity is weak and
heterophily is high. The structural hierarchy contributes
most when the graph shows strong structural organization.
The hierarchy validation results indicate that the learned
hierarchy gate is not arbitrary.

\section{Limitations and Future Work}
Despite its strong empirical performance, the proposed \textsc{DsHBoosting} framework presents several limitations. The \textsc{DsHBoosting} has a computational cost higher than that of standard single-stage GNNs. The STS requires multiple internal stages $T$ and external reweighting rounds, $R$, which increases both training time and memory usage. The overall complexity grows approximately linearly with respect to $T$ and $R$. Second, cosine-based KNN graph construction creates additional preprocessing overhead. Pairwise similarity construction in high-dimensional feature space can be costly for large-scale graphs. Finally, the model is constructed by several hyperparameters, and performance can be sensitive to their tuning. Improper selection of hyperparameters may lead to unstable training behavior or suboptimal results.

A promising direction for the future work is to extend \textsc{DsHBoosting}
into a multi-view learning framework by learning
stage-wise weights over multiple neighborhood views beyond
structural adjacency and cosine KNN. Another extension could be
to introduce a non-structural hierarchy learned from representation
space or prediction uncertainty.


```
















