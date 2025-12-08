suppressPackageStartupMessages({
  library(dendextend)
  library(viridis)
  library(igraph)
  library(ggraph)
  library(ggplot2)
})

## cosine distance for rows of a matrix
cosine_dist <- function(Z) {
  ZN <- Z / sqrt(rowSums(Z^2) + 1e-8)
  S <- ZN %*% t(ZN) # cosine similarity
  as.dist(1 - S) # distance = 1 - cos
}

## numeric -> viridis colours (used for leaves)
num_to_cols <- function(x, option = "magma") {
  r <- rank(x, ties.method = "average")
  viridis(length(x), option = option)[r]
}
plot_embedding_dendrogram <- function(
    Z_final, g_vec, D_all, C_all,
    y = NULL,
    n_plot = 400L,
    method = "average"
){
  N <- nrow(Z_final)
  stopifnot(length(g_vec) == N, nrow(D_all) == N, nrow(C_all) == N)
  
  set.seed(1)
  idx_sub <- if (N <= n_plot) 1:N else sort(sample.int(N, n_plot))
  
  Z_sub <- Z_final[idx_sub, , drop = FALSE]
  g_sub <- as.numeric(g_vec[idx_sub])
  D_sub <- as.numeric(D_all[idx_sub, 1])
  C_sub <- as.numeric(C_all[idx_sub, 1])
  
  ## 1) cluster in embedding space
  d_emb <- cosine_dist(Z_sub)
  hc_emb <- hclust(d_emb, method = method)
  dend <- as.dendrogram(hc_emb)
  labels(dend) <- as.character(idx_sub)
  
  ## 2) colour leaves by gate (can swap to depth/centrality if you like)
  ord <- order.dendrogram(dend)
  gate_cols <- num_to_cols(g_sub, option = "magma")
  labels_colors(dend) <- gate_cols[ord]
  
  par(mfrow = c(1,1), mar = c(5,4,4,8))
  plot(
    dend,
    main = "Dendrogram from Z_final\nleaf colour = gate g_i",
    ylab = "merge height (1 - cos)"
  )
  legend("topright",
         inset = c(-0.12,0),
         xpd = TRUE,
         title = "gate g_i",
         fill = viridis(5, option = "magma"),
         legend = sprintf("q%.1f", seq(0.1,0.9,by=0.2)),
         cex = 0.7, bty = "n")
  
  invisible(list(hc = hc_emb, dend = dend, idx = idx_sub))
}

plot_prior_dendrograms <- function(
    D_all, C_all, g_vec,
    n_plot = 400L,
    method = "average"
){
  N <- nrow(D_all)
  stopifnot(nrow(C_all) == N, length(g_vec) == N)
  
  set.seed(1)
  idx_sub <- if (N <= n_plot) 1:N else sort(sample.int(N, n_plot))
  
  D_sub <- as.numeric(D_all[idx_sub,1])
  C_sub <- as.numeric(C_all[idx_sub,1])
  g_sub <- as.numeric(g_vec[idx_sub])
  
  ## Depth-based dendrogram
  d_D <- dist(D_sub, method = "euclidean")
  hc_D <- hclust(d_D, method = method)
  dend_D <- as.dendrogram(hc_D)
  labels(dend_D) <- as.character(idx_sub)
  
  ord_D <- order.dendrogram(dend_D)
  depth_cols <- num_to_cols(D_sub, option = "plasma")
  labels_colors(dend_D) <- depth_cols[ord_D]
  
  par(mfrow = c(1,2), mar = c(5,4,4,5))
  plot(
    dend_D,
    main = "Dendrogram from depth prior D_i\nleaf colour = D_i",
    ylab = "|D_i - D_j| linkage"
  )
  
  ## Centrality-based dendrogram
  d_C <- dist(C_sub, method = "euclidean")
  hc_C <- hclust(d_C, method = method)
  dend_C <- as.dendrogram(hc_C)
  labels(dend_C) <- as.character(idx_sub)
  
  ord_C <- order.dendrogram(dend_C)
  cent_cols <- num_to_cols(C_sub, option = "cividis")
  labels_colors(dend_C) <- cent_cols[ord_C]
  
  plot(
    dend_C,
    main = "Dendrogram from centrality prior C_i\nleaf colour = C_i",
    ylab = "|C_i - C_j| linkage"
  )
  
  invisible(list(
    hc_D = hc_D, dend_D = dend_D,
    hc_C = hc_C, dend_C = dend_C,
    idx = idx_sub
  ))
}



plot_mst_tree_from_embeddings <- function(
    Z_final, g_vec, D_all,
    n_plot = 400L
){
  N <- nrow(Z_final)
  stopifnot(length(g_vec) == N, nrow(D_all) == N)
  
  set.seed(1)
  idx_sub <- if (N <= n_plot) 1:N else sort(sample.int(N, n_plot))
  
  Z_sub <- Z_final[idx_sub, , drop = FALSE]
  g_sub <- as.numeric(g_vec[idx_sub])
  D_sub <- as.numeric(D_all[idx_sub,1])
  
  ## cosine distance matrix
  d_emb <- cosine_dist(Z_sub)
  d_mat <- as.matrix(d_emb)
  diag(d_mat) <- 0
  
  ## graph + MST
  g_mst <- graph_from_adjacency_matrix(
    d_mat, mode = "undirected",
    weighted = TRUE, diag = FALSE
  )
  T_mst <- mst(g_mst, weights = E(g_mst)$weight)
  
  ## attach attributes
  V(T_mst)$gate <- g_sub
  V(T_mst)$depth <- D_sub
  
  ## plot with gate colours
  p_gate <- ggraph(T_mst, layout = "dendrogram") +
    geom_edge_diagonal(alpha = 0.4) +
    geom_node_point(aes(colour = gate), size = 1.5) +
    scale_colour_viridis_c(option = "magma") +
    ggtitle("MST tree from Z_final (no hclust)\nnode colour = gate g_i") +
    theme_void() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  ## plot with depth colours
  p_depth <- ggraph(T_mst, layout = "dendrogram") +
    geom_edge_diagonal(alpha = 0.4) +
    geom_node_point(aes(colour = depth), size = 1.5) +
    scale_colour_viridis_c(option = "plasma") +
    ggtitle("MST tree from Z_final (no hclust)\nnode colour = depth D_i") +
    theme_void() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  print(p_gate)
  print(p_depth)
  
  invisible(list(tree = T_mst, idx = idx_sub))
}

plot_mst_tree_from_embeddings <- function(
    Z_final, g_vec, D_all,
    n_plot = 400L
){
  suppressPackageStartupMessages({
    library(igraph)
    library(ggraph)
    library(viridis)
  })
  
  N <- nrow(Z_final)
  stopifnot(length(g_vec) == N, nrow(D_all) == N)
  
  set.seed(1)
  idx_sub <- if (N <= n_plot) 1:N else sort(sample.int(N, n_plot))
  
  Z_sub <- Z_final[idx_sub, , drop = FALSE]
  g_sub <- as.numeric(g_vec[idx_sub])
  D_sub <- as.numeric(D_all[idx_sub,1])
  
  ## ---- cosine distance matrix
  ZN <- Z_sub / sqrt(rowSums(Z_sub^2) + 1e-8)
  S <- ZN %*% t(ZN) # cosine similarity
  Dm <- 1 - S # distance
  diag(Dm) <- 0
  
  g_all <- graph_from_adjacency_matrix(Dm, mode = "undirected",
                                       weighted = TRUE, diag = FALSE)
  T_mst <- mst(g_all, weights = E(g_all)$weight)
  
  ## choose a root: node with largest gate
  root_id <- which.max(g_sub)
  
  ## attach attributes
  V(T_mst)$gate <- g_sub
  V(T_mst)$depth <- D_sub
  
  # create rooted tree layout
  lay <- create_layout(T_mst, layout = "tree", root = root_id)
  
  p_gate <- ggraph(lay) +
    geom_edge_diagonal(alpha = 0.4) +
    geom_node_point(aes(colour = gate), size = 1.5) +
    scale_colour_viridis_c(option = "magma") +
    ggtitle("MST tree from Z_final\nnode colour = gate g_i") +
    theme_void() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  p_depth <- ggraph(lay) +
    geom_edge_diagonal(alpha = 0.4) +
    geom_node_point(aes(colour = depth), size = 1.5) +
    scale_colour_viridis_c(option = "plasma") +
    ggtitle("MST tree from Z_final\nnode colour = depth D_i") +
    theme_void() +
    theme(
      legend.position = "right",
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  print(p_gate)
  print(p_depth)
  
  invisible(list(tree = T_mst, layout = lay, idx = idx_sub))
}


plot_node_feature_heatmap <- function(
    Z_final, g_vec, D_all, C_all, y,
    n_plot = 400L,
    method = "average"
){
  suppressPackageStartupMessages({
    library(pheatmap)
    library(RColorBrewer)
    library(viridis)
  })
  
  N <- nrow(Z_final)
  stopifnot(length(g_vec) == N,
            nrow(D_all) == N,
            nrow(C_all) == N,
            length(y) == N)
  
  set.seed(1)
  idx_sub <- if (N <= n_plot) 1:N else sort(sample.int(N, n_plot))
  
  Z_sub <- Z_final[idx_sub, , drop = FALSE]
  g_sub <- as.numeric(g_vec[idx_sub])
  D_sub <- as.numeric(D_all[idx_sub,1])
  C_sub <- as.numeric(C_all[idx_sub,1])
  y_sub <- factor(y[idx_sub])
  
  ## ---- 1) clustering of columns using embeddings
  ZN <- Z_sub / sqrt(rowSums(Z_sub^2) + 1e-8)
  S <- ZN %*% t(ZN)
  D_cos <- as.dist(1 - S)
  hc_emb <- hclust(D_cos, method = method)
  
  ## ---- 2) feature matrix: rows = features, cols = nodes
  feat_mat <- cbind(
    depth = D_sub,
    centrality = C_sub,
    gate = g_sub
  ) # N_sub x 3
  
  ## scale each feature to [0,1] so the blocks are comparable
  feat_scaled <- apply(feat_mat, 2, function(v) {
    rng <- range(v)
    if (diff(rng) < 1e-8) return(rep(0.5, length(v)))
    (v - rng[1]) / (rng[2] - rng[1])
  })
  
  mat <- t(feat_scaled) # 3 x N_sub
  colnames(mat) <- paste0("n", idx_sub)
  rownames(mat) <- c("D_i (depth)", "C_i (centrality)", "g_i (gate)")
  
  ## ---- 3) column annotation: class labels
  annotation_col <- data.frame(
    Class = y_sub
  )
  rownames(annotation_col) <- colnames(mat)
  
  n_cls <- length(levels(y_sub))
  ann_colors <- list(
    Class = setNames(
      brewer.pal(max(3, min(8, n_cls)), "Set1")[seq_len(n_cls)],
      levels(y_sub)
    )
  )
  
  ## ---- 4) heatmap
  pheatmap(
    mat,
    cluster_rows = FALSE, # keep feature order
    cluster_cols = hc_emb, # use our hclust from Z_final
    color = colorRampPalette(c("white","black"))(100),
    annotation_col = annotation_col,
    annotation_colors = ann_colors,
    show_colnames = FALSE,
    main = "Nodes clustered by Z_final (columns)\nrows = depth, centrality, gate"
  )
}
