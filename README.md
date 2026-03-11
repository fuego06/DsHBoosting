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

<img width="552" height="524" alt="image" src="https://github.com/user-attachments/assets/c99bcdd6-7465-4873-bc63-5a692e1017ba" />
The figure summarizes the complete workflow of the proposed method. The model starts from the input graph structure A and node features X, and constructs a feature-based cosine Top-kKNN neighborhood that complements the observed edges. At each internal STS stage, a learnable trade-off blends the masked adjacency with the KNN operator to form the stage-specific propagation view, while hierarchy information derived from graph priors modulates how strongly nodes aggregate and how aggressively logits are refined. The stage outputs are supervised through auxiliary softmax heads to stabilize training and make intermediate stages predictive. Finally, an external reweighting loop across rounds increases the influence of hard training nodes so that later optimization focuses on ambiguous and heterophilous regions.


<img width="375" height="540" alt="boosting_arc.png" src="https://github.com/user-attachments/assets/d95373c5-e555-4cc6-8444-29ad7cae246f" />
The figure visualizes the internal stage-wise refinement mechanism used in STS. The model forms an additive sequence of stage predictors, where each stage produces a residual correction to the accumulated logits rather than generating predictions from scratch. After each refinement step, an auxiliary softmax head provides deep supervision, ensuring that intermediate stages remain directly predictive and that gradients reach early parts of the network. This stage-wise design supports stable optimization and reduces reliance on a single deep message-passing stack, since refinement is distributed across stages. As a result, later stages focus on correcting the remaining errors (often concentrated on heterophilous or ambiguous nodes), while earlier stages learn coarser but reliable class separation under the current KNN--adjacency trade-off.

<img width="548" height="372" alt="image" src="https://github.com/user-attachments/assets/fd122922-9e05-48e6-9958-0497b19f3cda" />


#### The Hierarchy in the Proposed Model
<img width="375" height="540" alt="heatmap_good" src="https://github.com/user-attachments/assets/fbfeb46c-f41d-43b2-903c-43dbb84408e9" />

The figure  illustrates how the learned hierarchy gate is related to the structural priors after the model has formed as a final representation. We used hierarchical clustering (hclust) by using the cosine distance of the final embeddings Z_final. The dendrogram groups together nodes that the model considers similar in representation space. By observing consistent block patterns in the gate row, we can conclude that g_i is not random but instead varies systematically across groups of nodes with similar embeddings. This suggests that the hierarchy mechanism learns a structured modulation that aligns with the representation geometry and incorporates meaningful structural priors.


<img width="505" height="112" alt="image" src="https://github.com/user-attachments/assets/d4654792-b3a9-4162-9dc3-9b4388baea19" />

According to the Table, the learned gate g_i shows a stable negative Spearman correlation with depth; D_i, which means deeper nodes get smaller gate values. The correlation with centrality C_i is near zero. Therefore, we can say that the hierarchy mechanism is active and primarily implements a depth-aware modulation on Cornell, which provides interpretable structural behavior. 

#### Hyperparameter Tuning:

<img width="378" height="248" alt="image" src="https://github.com/user-attachments/assets/6fa70d89-9e50-484b-b7c2-e951b5d90cf4" />

The table reports the training protocol and the main hyperparameters of DsHBoosting for each dataset.
Here, T denotes the number of internal STS stages, i.e., the number of sequential hierarchy-aware refinement blocks that iteratively correct logits through residual updates. R denotes the number of external boosting rounds in the STS outer loop, where training node weights are updated to emphasize hard or misclassified nodes. The parameter k denotes the number of neighbors in the cosine Top-$k$ KNN graph, which determines the size of the feature-based neighborhood. The temperature parameter tau controls the sharpness of cosine similarity weights when constructing the learned KNN adjacency.














