# HBoost-2e2boosting-
The repository contains everything about the HBoost(2e2boosting) model


#### The Hierarchy in the Proposed Model
<img width="175" height="140" alt="heatmap_good3" src="https://github.com/user-attachments/assets/041c1977-3b5b-4301-9d36-0cc244364d50" />
<img width="875" height="540" alt="heatmap_good" src="https://github.com/user-attachments/assets/fbfeb46c-f41d-43b2-903c-43dbb84408e9" />
<img width="1536" height="557" alt="structure" src="https://github.com/user-attachments/assets/997e8e39-6917-47a1-ab08-ffd154b23707" />
<img width="875" height="540" alt="boosting_arc.png" src="https://github.com/user-attachments/assets/d95373c5-e555-4cc6-8444-29ad7cae246f" />


\documentclass[journal]{IEEEtran}

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{lettrine}
\usepackage{indentfirst}
\usepackage{lipsum}
\usepackage{parskip}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{libertine}
\usetikzlibrary{arrows.meta,positioning,fit}

\usepackage[
backend=biber,
style=alphabetic,
sorting=ynt
]{biblatex}
\addbibresource{sample.bib}

\title{DsHBoosting-GCN: Deep Supervised End-to-End Boosted Structurally Hierarchical GNNs}
\author{Bartu~Tamer}

\begin{document}
\maketitle
\section{Introduction}
\lettrine{G}{raph} datasets provide a natural representation for the relational problems with the function of the observation’s behavior, which
is not determined only by its own attributes but also by the network of interactions it takes part in. It encodes this relational structure through nodes, edges, and graph-level attributes. Because it can be applied to real-life data, this interactive system representation has attracted noticeable attention lately. The examples of the real-life applications of graph datasets include social and communication platforms \cite{1}\cite{2},\cite{3},  finance \cite{4},\cite{5}, biology and medicine \cite{6},\cite{7}, and transportation \cite{8}. This variety of applications creates its own problems and answers; therefore, studies related to the node classification task have become increasingly prevalent.

Graph neural networks (GNNs) \cite{9} have been used for different types of datasets for node classification tasks, but the main problem with GNNs is that they can not catch the real-life datasets' heterophilous nature, i.e., \textit{connected nodes tends to be different from each other's labels}. To overcome this problem with the heterophily different methods were proposed. Large-scale learning on non-homophilous graphs (LINKX) \cite{10} grants straightforward minibatch training and inference have performance improvement on heterophilous datasets. The node classification architecture H2GCN \cite{11} identifies a set of designs. The design does not mix a node with its neighbor immediately; it looks beyond 1-hop to 2-hop neighbors and uses the information from multiple layers. This design helps to improve performance on the non-homophilous datasets. Another approach for handling the heterophily is the Generalized PageRank Graph Neural Network (GPR-GNN) \cite{12} adaptively learns weights of trust for each hop distance by using GPR ,so it jointly optimizes node features and topological information. It determines hop distances to understand whether neighbors are helpful or misleading. Our proposed method concerns the non-homophilous graphs and estimates them by using a trade-off between k-nearest neighbors (KNN) and structural adjacency. 

 Another problem with the node classification is \textit{oversmoothing}, i.e., \textit{a common failure for the message-passing GNNs where model loses signal after stacking many layers and node embeddings become similar to each other}. Deeper Graph Convolutional Networks (GCNIIs) \cite{13} proposed to overcome \textit{oversmoothing} problem through using initial residuals and identity mapping by providing theoretical and experimental evidence that the two approaches adequately relieve the problem. DropEdge \cite{14} which randomly removes a certain amount of edges from the input at each training epoch, acting like a message-passing reducer. Our model escapes from the oversmoothing by learning a new propagation operation at each stage; therefore, it never applies the same Laplacian power consecutively.

In addition to these topics, the researchers also created algorithms by investigating the topological and hierarchical structures of the graph datasets. Structural node embeddings (struc2vec) \cite{15} learns node embeddings on structural roles such as hubs and bridges. Hierarchical graph convolutional network (H-GCN) \cite{16} proposed to handle shallow neighborhood aggregation and to have a better graph pooling structure. In our model hierarchy is created by using graph priors, and it is different from the classical "graph hierarchy" in pooling. It is node-wise and structural. 

To improve algorithm performance when working with unbalanced datasets, ensemble learning methods were implemented to graph learning methods. One of them can be seen as a boosting approach. Boosting-GNN \cite{17} is an ensemble model that uses GNN as a base classifier during boosting. Focal loss \cite{18} reweighting can also used for solving the problem of the imbalanced dataset \cite{19}. Although these methods improves the performance of the model, these methods are not connected, and the boosting model is not an end-to-end model. Our proposed method is creating end-to-end boosting by minimizing the focal loss.
\textbf{Our contributions can be summarized as follows:}
\begin{itemize}
\item We propose a semi-supervised node classification algorithm \emph{HBoosting-GCN}, an end-to-end boosted hierarchical GNN that explicitly targets heterophilous graphs by learning, at each stage, a task-adaptive blend between structural adjacency and a cosine-based KNN graph in the learned feature space.
\item We introduce a \emph{hierarchy gate} driven by graph priors (depth and local community purity), which modulates message passing in a node-wise and structure-aware fashion, providing a principled mechanism to control smoothing strength across the graph.
\item We incorporate \emph{deep supervision} by attaching auxiliary softmax heads to intermediate boosting stages. These auxiliary losses stabilize optimization of the hierarchical model, provide multi-scale node-level signals, and help maintain class-discriminative structure across layers.
\item We integrate \emph{stage-wise boosting} and \emph{focal loss} directly into the GNN, reweighting nodes according to their hardness while keeping a single, shared set of parameters. This yields an end-to-end boosting scheme that remains computationally efficient and avoids training multiple independent GNNs.
\item We design a suite of \emph{hierarchy and over-smoothing diagnostics} based on embeddings, cosine-KNN structure, and gate statistics (e.g., correlations with depth and local purity), which allow us to quantitatively and visually inspect how the model uses topology, hierarchy, and boosting throughout training.
\end{itemize}
The remaining of this paper is organized as follows. Section II revises the related work. Section III introduces the preliminaries. In Section IV, our proposed method, an end-to-end boosted hierarchical GNN, is presented. Section V illustrates the experimental results. At last, we finalize our paper in Section VI.
\section{Related Work}
With the rise of heterophilous real-life datasets, the problem of heterophily attracted many researchers. Additionally, recent algorithms also address the issue of imbalanced datasets. The hierarchy of the graph and learning by summarizing it into an aggregated super-node and extracting global information is also one important direction these days. New research about these fields is mostly aimed at the following sections.

\subsection{Related Work on Hierarchy and Structural Models
}
To have a hierarchical structure in the graph datasets, several different approaches have been conducted. DIFFPOOL \cite{20} is designed for the graph classification tasks; it creates a structural hierarchy by frequently learning a soft-cluster assignment matrix at each GNN layer that aggregates node embeddings into clusters. H-GCN \cite{16} was created for the semi-supervised node classification tasks. The hierarchy is constructed in the model by repeatedly aggregating structurally similar nodes into hyper-nodes. Then, it refines the layers to restore the original graph. Since it extends the receptive field for each node, more global information can be captured.
HU-GNN \cite{21} is an uncertainty-aware graph neural network. It jointly learns embeddings and uncertainty to obtain robustness. It constructs a structural hierarchy by using a Gumbel-Softmax pooling layer. The similarity between these models is that they create a pooling-based graph-level hierarchy. Another method for defining hierarchy is based on structural roles and distances. 
The struc2vec \cite{15} measures node similarity at different scales by using hierarchy. GraphWave \cite{22} creates a method for representing every node's network neighborhood using low-dimensional embeddings by leveraging heat wavelet diffusion patterns. 
Distance Encoding (DE) \cite{23} can use different graph distances, such as shortest path distance or generalized PageRank scores, and then, using these priors, it controls the message aggregation in GNNs.
Graphormer \cite{24} inserts shortest path distance and centrality encodings as biases into attention. Hierarchical Distance Structural Encoding (HDSE) \cite{25} models the node distances in graph by centering the multi-level hierarchical structure of the graph and applying graph-coarsening. It defines hierarchical distance, and it guides the attention mechanism directly. 

Our approach differs from these methods in that we do not build a coarsened graph; rather, we define a node-wise structural hierarchy by using graph priors such as depth and community purity ($D_i$, $C_i$). These priors are used to define the position of a node in the graph. These priors are jointly used as parameters for the scalar gate, $g_i$, which regulates propagandation strength, residual updates, and stage-wise boosting. Therefore, we can say that structural hierarchy controls information flows over the network.
\subsection{Related Work on Handling Heterophily Problem}
The problem of having non-homophilous nodes drops the accuracy score in the node classification tasks since they affect the message passing or local aggregation, which could not capture the certain label patterns \cite{26}, \cite{27}. To overcome these problems, H2CGN \cite{11} divides the ego features and neighbor features instead of mixing them. It uses higher-order neighbors and combines multiple layers. Geom-GCN \cite{27} defines the neighborhoods by using structural and positional similarity. GloGNN \cite{28} learns the coefficient matrix to capture correlations between nodes with respect to which aggregation is being performed. FAGCN \cite{29} assigns positive or negative weights to neighbors through learned attention. LINKX \cite{10} enables large-scale learning on strongly non-homophilous graphs. GPR-GNN\cite{12} model controls the contribution of the propagandization of each layer by optimizing the Generalized PageRank weights. Mixhop \cite{30} combines multi-order representations 

Our model, HBoosting-GCN, learns a stage-wise trade-off between cosine KNN and structural adjacency and uses hierarchy gate $g_i$ to understand the level of trust graph propagandation each node should consider. This trade-off balances the benefit of structural information versus the risk of harmful heterophilous edges.
\subsection{Related Work on Imbalanced Data and Deep Supervision}
The ensemble method boosting is used for having better performance with unbalanced datasets. AdaGCN \cite{31} implements Adaboost and GCN layers to get deeper network models. To improve performance on the imbalanced data Boosting-GNN \cite{17} uses the GNN as a subclassifier of the boosting algorithm. Reweighting the different samples in the loss function by using focal loss  \cite{18},\cite{19} is another approach. 

Deep supervision stabilizes the training by using auxiliary losses at intermediate layers. Deeply-Supervised Nets (DSN) \cite{32} is used for the image-classification tasks. The algorithm starts with a standard convolutional neural network (CNN) \cite{33} backbone, and connects with a local classifier (softmax) to each hidden layer, which becomes a companion output that directly predicts the class labels.

Our contribution is boosting and deep supervision fused into one end-to-end architecture. In every boosting stage, a structural hierarchy-aware GNN block triggers such that it clarifies the logits by a residual update, and it is directly supervised over its own auxiliary softmax head under focal loss. This creates end-to-end boosting, since weaker learners become internal stages of one shared network. In the proposed model, deep supervision shapes the hierarchical corrections stage by stage. 

\section{Preliminaries}

\subsection{Problem Setup and Notation}

This subsection presents the model inputs.The model considers a standard transductive semi-supervised node
classification problem on a single attributed graph. The data used in the model are shown
as
\begin{equation}
 G = (V, E, X, \mathbf{y}),
\end{equation}
where $V$ is the node set, $E$ shows the set of undirected edges,
$X$ is the raw node feature matrix, and $\mathbf{y}$ represents the
node label vector.

More precisely, assume
\begin{equation}
 X \in \mathbb{R}^{N \times F}, 
 \qquad 
 \mathbf{y} \in \{0,\ldots,K-1\}^N,
\end{equation}
where $N = |V|$ denotes the number of nodes and $F$ is the number of
features per node. The structural component $(V,E)$ is coded by the
adjacency matrix
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

In addition to these initial graph parameters, a depth vector
$\mathbf{D} \in [0,1]^N$ and a community-based centrality vector
$\mathbf{C} \in [0,1]^N$ are computed by using the full graph $A$.
These graph priors are described in Section~\ref{subsec:preproc}.
For simplicity, element-wise (Hadamard) products for matrices or
vectors are denoted by the symbol ``$\odot$''.

The model uses a binary mask $M \in \{0,1\}^{N \times N}$, which
controls that node pairs are allowed in the learned similarity graph:
\begin{equation}
 M_{ij} =
 \begin{cases}
 0, & \text{edge between } i \text{ and } j \text{ is hindered},\\[0.3ex]
 1, & \text{edge between } i \text{ and } j \text{ is allowed}.
 \end{cases}
\end{equation}
In all stages, the learned similarity operator is multiplied
element-wise by $M$ so that forbidden edges never appear.

\subsection{Preprocessing and Graph Priors}
\label{subsec:preproc}
In this subsection, feature engineering and graph-based priors that are used in the model are introduced. 
Before conducting the model with the high-dimensional, sparse raw
feature matrix (bag-of-words), a TF--IDF transform is applied. For
each word (feature) $f$, the count of how many nodes it has appeared
in is computed as its document frequency $\mathrm{df}_f$. If a feature
is seen in fewer nodes, a larger weight is assigned to that feature.
Let $N$ be the number of nodes. The TF--IDF weight for feature $f$ is
\begin{equation}
 w_f = \log\!\left( \frac{N + 1}{\mathrm{df}_f + 1} \right) + 1.
\end{equation}
Every feature value at a node is multiplied by its own weight,
resulting in a reweighted feature matrix $\widehat{X}$ with
\begin{equation}
 \widehat{X}_{i,f} = w_f \, X_{i,f}.
\end{equation}

Secondly, every node’s feature vector is normalized to have a unit
Euclidean norm:
\begin{equation}
 \big\| \widehat{X}_{i,:} \big\|_2 = 1,
 \qquad
 \forall i \in \{1,\ldots,N\},
\end{equation}
which avoids scale imbalances across nodes.
Third, dimensionality reduction is applied via Principal Component
Analysis (PCA). Crucially, the PCA transformation is conducted only on
the training nodes and then applied to all nodes to avoid any
feature-space leakage from validation or test nodes into the training
procedure. Denoting the PCA mapping fitted on training nodes by
$\mathrm{PCA}_{\mathrm{train}}(\cdot)$, we obtain reduced features
\begin{equation}
 \widetilde{X}_{i,:}
 =
 \mathrm{PCA}_{\mathrm{train}}\big(\widehat{X}_{i,:}\big),
 \quad \forall i.
\end{equation}

This process (TF--IDF, $\ell_2$-normalization, and PCA fitted only on
training nodes) is used in all benchmark datasets (Cora, Citeseer, and
Chameleon). For the datasets Chameleon and Squirrel, richer structural
augmentation is achieved by using multi-hop structural augmentation
(such as including $AX$ and $A^2X$) to show that the model fits well
in different feature settings.
\section{Methodology}
\label{sec:methodology}
This section provides a detailed illustration of the suggested methodology. Firstly, problem setup and notations are described. Secondly, feature transformations before conducting the model and graph priors used in the model are explained. Model layers are introduced, and how they interact to form a hierarchy-aware boosting architecture is explained. Moreover, the trade-off between the feature-driven cosine KNN graph and masked structural adjacency in the propagandation operator is shown. The deep supervision idea of the model is driven. Finally, the boosting architecture of the model, the loss function used for the training, and how the model prevents leakage are discussed. 
\subsection{Model Layers and Structure}
\label{subsec:model_layers}
This part shows how the main layers are designed, how they work, and
how they help each other. Together they create a hierarchical,
depth-aware architecture. Throughout this subsection, let
\begin{equation}
 Z \in \mathbb{R}^{N \times d}
\end{equation}
denote the current node embedding matrix at a given stage, and let
$Z_i \in \mathbb{R}^d$ be the embedding of node $i$.

\subsubsection{Row-normalized adjacency: AdjRowNorm}

The \emph{AdjRowNorm} layer produces a row-stochastic matrix where each
row sums to one. Given an adjacency-like operator
$A \in \mathbb{R}^{N \times N}$, the normalized matrix
$\widetilde{A}$ is
\begin{equation}
 \widetilde{A}_{ij}
 =
 \frac{A_{ij}}{\sum_{k=1}^{N} A_{ik} + \varepsilon},
 \qquad \varepsilon > 0,
\end{equation}
so that each row of $\widetilde{A}$ sums approximately to one. This
layer is crucial for stabilizing the message passing and normalizing
the learned similarity operators so that they can be interpreted as
probability kernels.

\subsubsection{Cosine KNN Graph: CosineKNNTopK}

The \emph{CosineKNNTopK} layer is a feature-based topological layer. It
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
\end{equation}
For each node $i$, the top-$k$ most similar neighbors are kept; denote
this set by $\mathcal{N}_k(i)$. A row-wise softmax is applied so that
their weights sum to one:
\begin{equation}
 \widetilde{S}_{ij}
 =
 \frac{\exp(S_{ij}/\tau)\,\mathbf{1}\{j \in \mathcal{N}_k(i)\}}
 {\sum_{m \in \mathcal{N}_k(i)} \exp(S_{im}/\tau)},
\end{equation}
where $\tau > 0$ is a temperature parameter. The KNN graph is dynamic:
as embeddings are updated during training, the similarity graph also
changes. On heterophilous datasets, it generally discovers long-range
label-consistent neighbors that are not directly connected in the
original graph.

\subsubsection{Learnable adjacency blending: BlendAdjAlpha}

The \emph{BlendAdjAlpha} layer is used for combining the original
adjacency and the KNN-based similarity graph into a single propagation
operator. Both components are row normalized. A learned scalar
parameter $\alpha \in (0,1)$ determines the relative weight:
\begin{equation}
 B = \alpha\,\widetilde{A} + (1 - \alpha)\,\widetilde{S}.
\end{equation}
If $\alpha \approx 1$, the model has confidence in the original graph
and effectively ignores the KNN graph. If $\alpha \approx 0$, the model
relies heavily on the learned similarity structure. The blending
parameter is learned end-to-end and shared across nodes at each stage
of the model. In later sections, we show how this learnable adjacency
blending and the cosine KNN graph layer create a trade-off between
original topology and feature-based similarity.

\subsubsection{Parent Bias: variance-based importance}

The \emph{ParentBias} layer determines one scalar weight per node
based on the squared norm of its current embedding. Nodes with higher
embedding variance are evaluated as informative or uncertain and
receive higher bias. A typical form is
\begin{equation}
 \beta_i
 =
 \sigma\big( \gamma \,\|Z_i\|_2^2 \big),
\end{equation}
where $\gamma>0$ is a learnable scale and $\sigma(\cdot)$ is the
logistic function. This value is then added into the attention logits.
Nodes with strong, discriminative embeddings are more likely to impact
their neighbors, while nodes with uncertain embeddings are down-
weighted in the attention softmax. The ParentBias layer therefore
acts as a data-driven importance weighting mechanism for message
sources.

\subsubsection{Hierarchy Gate: depth-aware node gate}

The \emph{Hierarchy Gate} component takes three types of inputs for
each node: preprocessed node features $X_i$, a scalar centrality score
$C_i$, and a normalized depth score $D_i$. These inputs are passed
through a small neural network to produce a gate value
\begin{equation}
 g_i = g(X_i, C_i, D_i) \in (0,1).
\end{equation}
Nodes that are central topologically, well connected, and cluster-
consistent tend to have higher $g_i$. This gate is used to modulate
the effect of the deep residual updates, so that reliable nodes
receive stronger updates and less reliable nodes are updated more
conservatively.

\subsubsection{Hierarchy-Aware Attention}

The \emph{hierarchy-aware attention} layer employs the blended operator
and topological information to decide how strongly each node should
attend to each neighbor. It first projects node features into a latent
space, then forms attention logits that combine three signals:

\begin{itemize}
 \item graph structure: log-weights from the blended adjacency
 $\log B_{ij}$ provide that edges favored by either the
 original graph or the KNN graph receive higher prior
 importance;
 \item graph hierarchy: a depth-based term penalizes pairs of nodes
 that are far from each other in the hierarchical sense;
 \item node importance: the ParentBias term $\beta_j$ boosts neighbors
 with high embedding squared norm.
\end{itemize}

A stylized form of the attention logits is
\begin{equation}
 \ell_{ij}
 =
 \phi(Z_i, Z_j)
 +
 \log B_{ij}
 +
 \psi(D_i, D_j)
 +
 \beta_j,
\end{equation}
where $\phi(\cdot,\cdot)$ encodes feature interactions and
$\psi(\cdot,\cdot)$ encodes depth differences. The attention weights
are then obtained by a softmax over neighbors:
\begin{equation}
 \alpha_{ij}
 =
 \frac{\exp(\ell_{ij})}
 {\sum_{m=1}^{N} \exp(\ell_{im})},
\end{equation}
and used to aggregate neighbor features for each node.

\subsubsection{DendroELU: depth-decayed activation}

The \emph{DendroELU} layer customizes the activations to be depth-
aware. Nodes near the structural core keep almost all activation,
while outer nodes are damped. For node $i$, the activation is
\begin{equation}
 \varphi(Z_i, D_i)
 =
 \mathrm{ELU}(Z_i)\,\exp(-\lambda D_i),
\end{equation}
where $\lambda>0$ is a learned parameter. This acts as a hierarchical
regularizer: it keeps strong responses in the core and suppresses
over-amplification in the periphery.

\subsubsection{DeepHierarchyBlockV2: refinement of gated residual features}

The \emph{DeepHierarchyBlockV2} is a multi-layer gated residual block
that uses the Hierarchy Gate scores to refine node embeddings. After
applying an affine transform, nonlinearity, normalization, and dropout
at each internal layer, it interpolates between a conservative update
(for low-gate, uncertain nodes) and a strong residual update (for
high-gate, dependable nodes). A simplified update for node $i$ at
internal layer $\ell$ is
\begin{equation}
 H^{(\ell+1)}_i
 =
 H^{(\ell)}_i
 +
 g_i\,\Delta^{(\ell)}_i
 +
 (1 - g_i)\,\alpha\,\Delta^{(\ell)}_i,
\end{equation}
where $\Delta^{(\ell)}_i$ is the transformed feature update and
$\alpha \in (0,1)$ controls the conservative path. Stacking multiple
such layers yields deep, stability-preserving refinement, allowing the
model to aggressively reshape features in structurally reliable areas
while avoiding oscillations or overfitting elsewhere.

\subsubsection{ClusterGate: channel gating based on prototypes}

In the latent space, prototype-driven channel gating is carried out by
the \emph{ClusterGate} layer. After learning a limited number of
prototypes, it calculates each node's soft assignment to these
prototypes and transforms them into per-channel gate values that
either selectively amplify or suppress feature dimensions.

Let $p_i \in \mathbb{R}^K$ be the soft assignment vector of node $i$
over $K$ prototypes. The per-channel gate vector is
\begin{equation}
 \mathbf{g}_i^{\text{chan}}
 =
 \sigma\big( p_i W_g + \mathbf{b}_g \big),
\end{equation}
and the gated embedding is
\begin{equation}
 Z'_i = \mathbf{g}_i^{\text{chan}} \odot Z_i,
\end{equation}
where $W_g$ and $\mathbf{b}_g$ are learnable parameters and
$\sigma(\cdot)$ acts element-wise. This produces sharper, more
class-discriminative node representations prior to the boosting stage
by emphasizing channels that are consistent with a node's most likely
structural/semantic pattern and downweighting irrelevant or noisy
channels.


\begin{figure*}[!t]
\centering

\begin{subfigure}[b]{0.3\textwidth}
\centering
\includegraphics[width=\linewidth]{heatmap_good.jpeg}
\caption{Image 1}
\label{fig:img1}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.3\textwidth}
\centering
\includegraphics[width=\linewidth]{heatmap_good2.jpeg}
\caption{Image 2}
\label{fig:img2}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.3\textwidth}
\centering
\includegraphics[width=\linewidth]{heatmap_good3.jpeg}
\caption{Image 3}
\label{fig:img3}
\end{subfigure}

\caption{Your overall caption here.}
\label{fig:three_in_row}
\end{figure*}



\subsection{Theory: Hierarchy in the Proposed Model}
\label{subsec:theory_hierarchy}

In this subsection, how community and depth priors are used to create a graph-induced node hierarchy and how this hierarchy affects the model's attention and residual updates are formally explained. In addition to that, the meaning of the model hierarchy is discussed. The key idea is that each node receives continuous scores that quantify its \emph{community consistency}, \emph{core--periphery depth}, and \emph{learned reliability}, and these scores globally modulate message passing without coarsening the graph.

\subsubsection{Construction of hierarchy scores}

Let $\mathcal{N}(i)$ denote the (undirected) neighbor set of
node $i$ in $A$, and let $c_i$ be the community label of node
$i$ obtained from a community detection method (e.g., Louvain).
The community-based centrality prior $C_i$ is defined as
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
where $\mathbf{1}\{\cdot\}$ is the indicator function and
$C_i \in [0,1]$ measures how well node $i$ agrees with its
local community.

Let $e_i$ be the eccentricity of node $i$, and let
$e_{\min} = \min_i e_i$, $e_{\max} = \max_i e_i$.
The normalized depth prior $D_i^{\text{depth}}$ is
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
so that nodes close to the structural core have smaller
$D_i^{\text{depth}}$, while peripheral nodes have larger values.

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

\subsubsection{Use of hierarchy in attention and residual refinement}

The blended propagation operator $B$ (obtained by combining the
original adjacency and the feature-based KNN graph) defines a
structural prior on edges. The hierarchy-aware attention layer
uses $B$, depth, and node importance to construct logits
$\ell_{ij}$ for messages from node $j$ to node $i$:
\begin{equation}
 \ell_{ij}
 =
 \phi(z_i, z_j)
 +
 \log B_{ij}
 -
 \lambda_D
 \bigl|
 D_i^{\text{depth}} - D_j^{\text{depth}}
 \bigr|
 +
 \beta_j,
 \label{eq:hier_attention_logits}
\end{equation}
where $\phi$ encodes feature interactions, $\lambda_D \!>\! 0$
controls depth penalization, and $\beta_j$ is the variance-based
ParentBias term for node $j$. The attention weights are then
\begin{equation}
 \alpha_{ij}
 =
 \frac{\exp(\ell_{ij})}{
 \sum_{k} \exp(\ell_{ik})
 },
 \qquad
 \sum_{j} \alpha_{ij} = 1,
 \label{eq:hier_attention_weights}
\end{equation}
which favor edges that are (i) supported by $B_{ij}$, (ii)
hierarchically close in depth, and (iii) originating from
informative parents with high $\beta_j$.

Inside the DeepHierarchyBlock, the gate $g_i$ controls the
strength of residual feature updates. Let $h_i^{(\ell)}$ be the
representation of node $i$ at internal layer $\ell$ and
$\Delta_i^{(\ell)}$ the transformed update (after affine
mapping, nonlinearity, normalization and dropout). The update
rule is
\begin{equation}
 h_i^{(\ell+1)}
 =
 h_i^{(\ell)}
 +
 \bigl(
 g_i + \alpha\,(1 - g_i)
 \bigr)\,
 \Delta_i^{(\ell)},
 \label{eq:hier_residual}
\end{equation}
where $\alpha \in (0,1)$ controls a conservative path. When
$g_i$ is large, the node receives a strong residual update;
when $g_i$ is small, the update is downscaled, stabilizing the
propagation on unreliable or noisy nodes.

\subsubsection{Distinct notion of hierarchy}

The notion of hierarchy in the proposed model is fundamentally
different from classical ``graph hierarchy'' in pooling or
multiscale GNNs, where the graph is recursively coarsened into
super-nodes and processed at multiple resolutions. Here, the
hierarchy is \emph{soft, node-wise, and purely structural}:
each node is endowed with continuous scores
$\bigl(C_i, D_i^{\text{depth}}, g_i\bigr)$ that quantify its
community consistency, core--periphery depth, and learned
reliability, but the underlying node set and adjacency are never
coarsened. These hierarchical scores are reused throughout the
network---in attention
\eqref{eq:hier_attention_logits}–\eqref{eq:hier_attention_weights},
in residual refinement \eqref{eq:hier_residual}, and in the
boosting stages---so that ``core'' and ``peripheral'' nodes are
treated differently during the entire training process. As a
result, the architecture is hierarchy-aware without relying on
graph pooling, and is complementary to standard multiscale GNN
approaches.

\subsection{Trade-Off Between KNN and Structural Adjacency}
This subsection describes how training automatically steers this balance under the learned hierarchy by analyzing the learned trade-off parameter $\alpha$ that balances the feature-driven cosine KNN graph and the masked structural adjacency in the propagation operator.

The propagation operator is an adaptive convex combination of the feature-driven
cosine KNN graph and the masked structural adjacency:
\begin{equation}
B(\alpha)
\;=\;
\alpha\,\widetilde{A}^{\text{mask}}
\;+\;
(1-\alpha)\,\widetilde{S},
\label{eq:blend_operator}
\end{equation}
where $\alpha \in (0,1)$ is a learnable scalar shared across nodes at each
boosting stage, and $\widetilde{A}^{\text{mask}}$ and $\widetilde{S}$ are
row-normalized. Small $\alpha$ highlights the learned similarity structure
(i.e., the KNN graph), whereas large $\alpha$ highlights the original topology.

Let $\mathcal{L}(\alpha)$ denote the training loss and $G_B = \partial
\mathcal{L} / \partial B$ the gradient of the loss with respect to the blended
operator. By the chain rule,
\begin{equation}
\frac{\partial \mathcal{L}}{\partial \alpha}
\;=\;
\Big\langle
G_B,\,
\widetilde{A}^{\text{mask}} - \widetilde{S}
\Big\rangle,
\label{eq:dL_dalpha}
\end{equation}
where $\langle \cdot,\cdot \rangle$ denotes the Frobenius inner product. When
this inner product is negative, gradient descent increases $\alpha$, pushing
the model towards the structural adjacency. On the other hand, optimization
shifts mass towards the KNN graph by \emph{decreasing} $\alpha$ if this inner
product is positive. As a result, $\alpha$ is a data-driven knob that
automatically chooses the balance that minimizes the loss under the current
hierarchy, rather than a fixed hyperparameter tuned by hand.




\subsection{Deep Supervision over Boosting Stages}
\label{subsec:deep_supervision}


In addition to the final prediction head, the model affixes auxiliary classification heads to the intermediate boosting stages. Let \(F^{(t)} \in \mathbb{R}^{N \times K}\) denote the logits at boosting stage \(t \in \{1,\dots,T\}\), and
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

This design shines in two complementary ways. First, because strong gradient signals are sent directly to the early stages via~\eqref{eq:deep_supervision_loss}, each boosting stage is encouraged to be locally predictive rather than acting as an unconstrained feature transformer. This stabilizes optimization and gives the stage hierarchy genuine semantic meaning. Second, it regularizes the hierarchy: intermediate representations must already align with the graph structure and labels under the current KNN--adjacency blend, so subsequent stages focus on correcting difficult or ambiguous nodes instead of repairing arbitrary distortions. In practice, this leads to smoother stage-wise refinement, better calibration, and improved robustness on both homophilous and heterophilous graphs.


\subsection{Boosting, Focal Loss, and Leakage Prevention}
\label{subsec:boosting_focal_leakage}
In this section, the model boosting scheme is illustrated. The loss function used for the model training is shown. More importantly, how model prevents from the information leakage is explained
\subsubsection{Internal and External Boosting}

An additive ensemble of boosting stages forms the core network structure. 
Let $F^{(t)} \in \mathbb{R}^{N \times K}$ denote the logits after stage $t$, with
$F^{(0)}$ initialized to zero. 
At each stage the model predicts a residual correction $\Delta^{(t)}$ and updates
the global logits as
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
learns a stable \emph{additive hierarchy of experts}, since each $\Delta^{(t)}$ is
gated by the hierarchy (via centrality and depth) and implicitly constrained to be small.

On top of this internal stage-wise boosting, an outer boosting loop that
iteratively reweights training nodes across rounds is applied. Let $w_i^{(r)}$ be the weight of
node $i$ at boosting round $r$, and let $p_i^{(r)}$ be the predicted probability of
its true class. The hardness score defined as:
\begin{equation}
 h_i^{(r)} = 1 - p_i^{(r)},
 \label{eq:hardness}
\end{equation}
and mark nodes as ``hard'' whenever $h_i^{(r)} \ge \tau$ for a fixed threshold
$\tau \in (0,1)$. The weights are then updated as
\begin{equation}
 w_i^{(r+1)} \propto 
 w_i^{(r)} \big( 1 + (\gamma - 1)\,\mathbf{1}\{h_i^{(r)} \ge \tau\} \big),
 \label{eq:weight_update}
\end{equation}
followed by normalization over training nodes so that 
$\sum_{i \in \mathcal{V}_{\text{train}}} w_i^{(r+1)} = 1$.
In other words, consistently misclassified or low-confidence nodes receive a 
multiplicative upweighting factor $\gamma > 1$, while easy nodes maintain roughly
the same influence. This external boosting mechanism forces the model to concentrate 
its capacity on structurally ambiguous, heterophilous, or otherwise challenging 
regions of the graph, while preserving a clean separation between training, 
validation, and test labels.

\subsubsection{Focal Loss}

To further highlight nodes that are unclear or confusing, the model uses a focal
classification loss instead of plain cross-entropy. For a node $i$ with one-hot
label vector $y_i \in \{0,1\}^{K}$ and predicted probabilities 
$p_i \in [0,1]^K$, the per-node focal loss is
\begin{equation}
 \ell_{\text{focal}}(y_i, p_i)
 = - \sum_{c=1}^{K} y_{ic} \, (1 - p_{ic})^{\gamma} \log p_{ic},
 \label{eq:focal_loss}
\end{equation}
where $\gamma \ge 0$ is the focusing parameter. 
When $p_{ic}$ is large (easy, confident predictions), the factor 
$(1 - p_{ic})^{\gamma}$ suppresses the contribution of that node. 
When $p_{ic}$ is small (hard, confused nodes), the loss is amplified.

On graphs with heterophily, class imbalance, or label noise, this design is 
particularly advantageous. Combined with the external boosting weights in 
\eqref{eq:weight_update}, focal loss yields a \emph{two-level focusing mechanism}: 
the outer loop reweights hard nodes in the sample space, while the focal term 
focuses gradient magnitude on hard nodes in the loss space. The resulting model 
naturally concentrates its representational strength on the most ambiguous parts 
of the hierarchy, without destabilizing training.

\subsubsection{Leakage Prevention}

Because structure, features, and labels all co-exist on a single graph, transductive
node classification is especially susceptible to subtle forms of information leakage. 
The proposed pipeline explicitly addresses leakage at three levels:

\paragraph*{(i) Training-only feature transforms}
PCA, TF--IDF weighting, and related feature preprocessing steps are fitted 
\emph{only} on training nodes. Let $X_{\text{train}}$ denote the feature matrix
restricted to the training set. All statistics---document frequencies, IDF weights,
centering vectors, and PCA loadings---are computed from $X_{\text{train}}$ alone.
The resulting linear transform is then applied to validation and test nodes, but
their features never influence these parameters. This removes any feature-space
leakage from validation or test nodes back into the training pipeline.

\paragraph*{(ii) Label-free structural priors}
The centrality prior $C$ and depth prior $D$ are computed from the \emph{unlabeled}
graph structure $(V,E)$ only. Both $C_i$ (community-based centrality) and $D_i$
(normalized depth / eccentricity) depend solely on the topology and community 
structure, not on node labels. Consequently, these priors are fully unsupervised 
and can safely be computed on the full graph without introducing test labels into 
the model.

\paragraph*{(iii) Masked similarity and supervised objective}
The learned cosine KNN graph is always multiplied element-wise by the binary mask
$M$, which specifies which node pairs are allowed to form similarity edges in the
current training regime. This guarantees that forbidden pairs never affect the
learned propagation operator. Furthermore, all supervised losses, deep supervision
terms, and boosting weights are computed \emph{only} on labeled training nodes;
validation labels are used solely for model selection (e.g., early stopping and
learning-rate scheduling), and test labels are never accessed during training.

By combining training-only feature transforms, label-free structural priors, and
an explicit edge mask on the learned similarity graph, the methodology ensures that
performance improvements genuinely arise from improved hierarchical modeling rather
than from any form of transductive leakage.

\section{References}

[1] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Galligher, and T. Eliassi-Rad, “Collective Classification in Network Data,” AI Magazine, vol. 29, no. 3, p. 93, Sep. 2008.

[2] X. Huang et al., “DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection,” arXiv (Cornell University), Jan. 2022,arxiv.2207.03579.

[3]


\end{document}




















