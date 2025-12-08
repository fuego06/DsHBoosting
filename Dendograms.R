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


