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
<img width="375" height="540" alt="heatmap_good" src="https://github.com/user-attachments/assets/fbfeb46c-f41d-43b2-903c-43dbb84408e9" />
<img width="375" height="540" alt="boosting_arc.png" src="https://github.com/user-attachments/assets/d95373c5-e555-4cc6-8444-29ad7cae246f" />




















