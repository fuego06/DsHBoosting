suppressPackageStartupMessages({
  library(dendextend)
  library(viridis)
  library(igraph)
  library(ggraph)
  library(ggplot2)
})


cosine_dist <- function(Z) {
  ZN <- Z / sqrt(rowSums(Z^2) + 1e-8)
  S <- ZN %*% t(ZN) 
  as.dist(1 - S) 
}


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
  
  
  d_emb <- cosine_dist(Z_sub)
  hc_emb <- hclust(d_emb, method = method)
  dend <- as.dendrogram(hc_emb)
  labels(dend) <- as.character(idx_sub)
  
  
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
  
 
  d_emb <- cosine_dist(Z_sub)
  d_mat <- as.matrix(d_emb)
  diag(d_mat) <- 0
 
  g_mst <- graph_from_adjacency_matrix(
    d_mat, mode = "undirected",
    weighted = TRUE, diag = FALSE
  )
  T_mst <- mst(g_mst, weights = E(g_mst)$weight)
  
  
  V(T_mst)$gate <- g_sub
  V(T_mst)$depth <- D_sub
  
 
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
 
  ZN <- Z_sub / sqrt(rowSums(Z_sub^2) + 1e-8)
  S <- ZN %*% t(ZN) 
  Dm <- 1 - S 
  diag(Dm) <- 0
  
  g_all <- graph_from_adjacency_matrix(Dm, mode = "undirected",
                                       weighted = TRUE, diag = FALSE)
  T_mst <- mst(g_all, weights = E(g_all)$weight)
  
  
  root_id <- which.max(g_sub)
  
  
  V(T_mst)$gate <- g_sub
  V(T_mst)$depth <- D_sub
  

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
  
  
  ZN <- Z_sub / sqrt(rowSums(Z_sub^2) + 1e-8)
  S <- ZN %*% t(ZN)
  D_cos <- as.dist(1 - S)
  hc_emb <- hclust(D_cos, method = method)
  
 
  feat_mat <- cbind(
    depth = D_sub,
    centrality = C_sub,
    gate = g_sub
  ) 
  
 
  feat_scaled <- apply(feat_mat, 2, function(v) {
    rng <- range(v)
    if (diff(rng) < 1e-8) return(rep(0.5, length(v)))
    (v - rng[1]) / (rng[2] - rng[1])
  })
  
  mat <- t(feat_scaled) 
  colnames(mat) <- paste0("n", idx_sub)
  rownames(mat) <- c("D_i (depth)", "C_i (centrality)", "g_i (gate)")
  

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
  

  pheatmap(
    mat,
    cluster_rows = FALSE,
    cluster_cols = hc_emb, 
    color = colorRampPalette(c("white","black"))(100),
    annotation_col = annotation_col,
    annotation_colors = ann_colors,
    show_colnames = FALSE,
    main = "Nodes clustered by Z_final (columns)\nrows = depth, centrality, gate"
  )
}



plot_mst_depth_tree_col_gate <- function(Z, D_all, g_vec,
                                         title = "MST tree from Z_final\nnode colour = gate g_i") {
  Z <- as.matrix(Z)
  N <- nrow(Z)
  
  
  depth <- if (is.matrix(D_all)) as.numeric(D_all[, 1]) else as.numeric(D_all)
  gate <- as.numeric(g_vec)
  stopifnot(length(depth) == N, length(gate) == N)
  
 
  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z)
  S <- Zn %*% t(Zn) # cosine similarity
  D <- 1 - S # cosine distance
  diag(D) <- 0
  
 
  g_full <- graph_from_adjacency_matrix(D, mode = "undirected",
                                        weighted = TRUE, diag = FALSE)
  mst_g <- mst(g_full, weights = E(g_full)$weight)
  
 
  root_idx <- which.max(depth)
  

  V(mst_g)$depth <- depth
  V(mst_g)$gate <- gate
  
 
  lay <- layout_as_tree(mst_g, root = root_idx)
  
 
  ggraph(mst_g, layout = "manual", x = lay[, 1], y = lay[, 2]) +
    geom_edge_link(colour = "grey80", alpha = 0.6) +
    geom_node_point(aes(colour = gate), size = 1.8) +
    scale_colour_viridis_c(option = "magma") +
    labs(colour = "gate",
         title = title) +
    theme_void(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
}

suppressPackageStartupMessages({
  library(RColorBrewer)
})

plot_heatmap <- function(
    Z, D_all, C_all, g_vec, y,
    n_nodes_max = 800,
    main_title = "Nodes clustered by Z_final\nrows = depth, centrality, gate"
){
  Z <- as.matrix(Z)
  N <- nrow(Z)
  
  
  if (N > n_nodes_max) {
    set.seed(123)
    keep_idx <- sort(sample(seq_len(N), n_nodes_max))
  } else {
    keep_idx <- seq_len(N)
  }
  
  Z <- Z[keep_idx, , drop = FALSE]
  depth <- if (is.matrix(D_all)) as.numeric(D_all[keep_idx, 1]) else as.numeric(D_all[keep_idx])
  cent <- if (is.matrix(C_all)) as.numeric(C_all[keep_idx, 1]) else as.numeric(C_all[keep_idx])
  gate <- as.numeric(g_vec[keep_idx])
  y_sub <- factor(y[keep_idx])
  Nsub <- nrow(Z)
  
 
  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z)
  S <- Zn %*% t(Zn)
  Dcos <- as.dist(1 - S)
  hc <- hclust(Dcos, method = "average")
  ord <- hc$order
  
 
  H_raw <- rbind(
    `D_i (depth)` = depth,
    `C_i (centrality)` = cent,
    `g_i (gate)` = gate
  ) 
  
  H_scaled <- H_raw
  for (i in seq_len(nrow(H_scaled))) {
    mu <- mean(H_scaled[i, ], na.rm = TRUE)
    sdv <- stats::sd(H_scaled[i, ], na.rm = TRUE)
    if (sdv > 0) {
      H_scaled[i, ] <- (H_scaled[i, ] - mu) / sdv
    } else {
      H_scaled[i, ] <- H_scaled[i, ] - mu
    }
  }
  
  
  H_ord <- H_scaled[, ord, drop = FALSE] 
  
  
  y_ord <- y_sub[ord]
  gate_ord <- gate[ord]
  
  
  qcuts <- quantile(gate_ord, probs = c(0.2, 0.4, 0.6, 0.8), na.rm = TRUE)
  qcuts <- unique(qcuts)
  if (length(qcuts) < 2L) qcuts <- sort(unique(c(min(gate_ord), max(gate_ord))))
  
  gate_bin <- cut(
    gate_ord,
    breaks = c(-Inf, qcuts, Inf),
    labels = paste0("q", c("0.2","0.4","0.6","0.8","1.0"))[seq_len(length(qcuts)+1L)],
    include.lowest = TRUE
  )
  
 
  class_levels <- levels(y_sub)
  class_cols <- brewer.pal(max(3, length(class_levels)), "Set1")[seq_along(class_levels)]
  names(class_cols) <- class_levels
  
  gate_levels <- levels(gate_bin)
  gate_cols <- brewer.pal(max(3, length(gate_levels)), "PuOr")[seq_along(gate_levels)]
  names(gate_cols) <- gate_levels
  
  
  heat_cols <- colorRampPalette(c("forestgreen", "white", "firebrick"))(200)
  
 
  op <- par(no.readonly = TRUE)
  on.exit(par(op))
  
 
  layout(
    matrix(c(1, 2, 3, 4), nrow = 4, byrow = TRUE),
    heights = c(2, 0.4, 0.4, 4)
  )
  
  
  par(mar = c(0, 4, 4, 2))
  plot(as.dendrogram(hc), main = main_title, ylab = "merge height", xlab = "", xaxt = "n")
  
 
  par(mar = c(0, 4, 0, 2))
  class_int <- as.integer(y_ord)
  class_mat <- matrix(class_int, nrow = Nsub, ncol = 1)
  image(
    z = t(class_mat), 
    col = class_cols,
    axes = FALSE,
    xlab = "",
    ylab = ""
  )
  axis(2, at = 0.5, labels = "Class", las = 1)
  legend("topright", legend = class_levels, fill = class_cols, cex = 0.6, bty = "n")
  
  
  par(mar = c(0, 4, 0, 2))
  gate_int <- as.integer(gate_bin)
  gate_mat <- matrix(gate_int, nrow = Nsub, ncol = 1)
  image(
    z = t(gate_mat),
    col = gate_cols,
    axes = FALSE,
    xlab = "",
    ylab = ""
  )
  axis(2, at = 0.5, labels = "Gate\nbin", las = 1)
  legend("topright", legend = gate_levels, fill = gate_cols, cex = 0.6, bty = "n")
  
 
  par(mar = c(5, 4, 2, 2))
  image(
    x = seq_len(Nsub),
    y = seq_len(nrow(H_ord)),
    z = t(H_ord),
    col = heat_cols,
    axes = FALSE,
    xlab = "nodes (clustered by Z)",
    ylab = ""
  )
  axis(1, at = pretty(seq_len(Nsub)))
  axis(2, at = seq_len(nrow(H_ord)),
       labels = rownames(H_ord), las = 1)
  
  
  zvals <- seq(-2, 2, length.out = 200)
  zcols <- heat_cols
  par(new = TRUE, mar = c(5, 0, 2, 4))
  image(
    x = 1, y = zvals, z = matrix(zvals, nrow = 1),
    col = zcols, axes = FALSE, xlab = "", ylab = ""
  )
  axis(4, at = c(-2, 0, 2), labels = c("Low", "Mid", "High"), las = 1)
  mtext("scaled prior / gate", side = 4, line = 2)
}

plot_mst_tree_from_embeddings <- function(
    Z,
    color_vec = NULL, 
    colour_name = "value", 
    main = "MST tree from Z",
    n_nodes_max = 1000
){
  
  Z <- as.matrix(Z)
  N <- nrow(Z)
  
  
  if (N > n_nodes_max) {
    set.seed(123)
    keep_idx <- sort(sample(seq_len(N), n_nodes_max))
    Z <- Z[keep_idx, , drop = FALSE]
    if (!is.null(color_vec)) color_vec <- color_vec[keep_idx]
  }
  
  
  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z)
  S <- Zn %*% t(Zn)
  D <- 1 - S 
  diag(D) <- 0
  
  
  g_full <- graph_from_adjacency_matrix(
    D, mode = "undirected", weighted = TRUE, diag = FALSE
  )
  mst_g <- mst(g_full, weights = E(g_full)$weight)
  
  
  if (!is.null(color_vec)) {
    V(mst_g)$color_val <- as.numeric(color_vec)
  } else {
    V(mst_g)$color_val <- NA_real_
  }
  
  
  lay <- layout_as_tree(mst_g, root = 1)
  lay_df <- data.frame(
    x = lay[,1],
    y = -lay[,2] 
  )
  

  p <- ggraph(mst_g, layout = "manual", x = lay_df$x, y = lay_df$y) +
    geom_edge_link(colour = "grey80", alpha = 0.7, width = 0.2) +
    geom_node_point(aes(colour = color_val), size = 1.8) +
    scale_colour_viridis_c(option = "magma", na.value = "grey70",
                           name = colour_name) +
    theme_void(base_size = 11) +
    ggtitle(main)
  
  return(p)
}



suppressPackageStartupMessages({
  library(pheatmap)
  library(RColorBrewer)
  library(dplyr)
})

plot_pheatmap_Zpriors <- function(
    Z, 
    D_all, 
    C_all, 
    g_vec, 
    y, 
    n_nodes_max = 800,
    main_title = "Nodes clustered by Z_final\ncolumns = depth, centrality, gate"
){
  Z <- as.matrix(Z)
  N <- nrow(Z)

  if (N > n_nodes_max) {
    set.seed(123)
    keep_idx <- sort(sample(seq_len(N), n_nodes_max))
  } else {
    keep_idx <- seq_len(N)
  }
  
  Z_sub <- Z[keep_idx, , drop = FALSE]
  depth <- if (is.matrix(D_all)) as.numeric(D_all[keep_idx, 1]) else as.numeric(D_all[keep_idx])
  centr <- if (is.matrix(C_all)) as.numeric(C_all[keep_idx, 1]) else as.numeric(C_all[keep_idx])
  gate <- as.numeric(g_vec[keep_idx])
  y_sub <- factor(y[keep_idx])
  

  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z_sub) 
  S <- Zn %*% t(Zn) 
  Dcos <- as.dist(1 - S) 
  hc_rows <- hclust(Dcos, method = "average")
  
  
  H <- cbind(
    `D_i (depth)` = depth,
    `C_i (centrality)` = centr,
    `g_i (gate)` = gate
  ) # N_sub x 3
  
  
  H_scaled <- scale(H)
  

  qcuts <- quantile(gate, probs = c(0.2, 0.4, 0.6, 0.8), na.rm = TRUE)
  gate_bin <- cut(
    gate,
    breaks = c(-Inf, qcuts, Inf),
    labels = c("q0-0.2", "q0.2-0.4", "q0.4-0.6", "q0.6-0.8", "q0.8-1.0"),
    include.lowest = TRUE
  )
  
  annot_row <- data.frame(
    Class = y_sub,
    GateBin = gate_bin
  )
  rownames(annot_row) <- rownames(H_scaled) <- paste0("node_", keep_idx)
  
 
  class_levels <- levels(y_sub)
  class_cols <- setNames(
    brewer.pal(max(3, length(class_levels)), "Set1")[seq_along(class_levels)],
    class_levels
  )
  gatebin_cols <- setNames(
    brewer.pal(5, "PuOr"),
    levels(gate_bin)
  )
  ann_colors <- list(
    Class = class_cols,
    GateBin = gatebin_cols
  )

  hm_cols <- colorRampPalette(c("forestgreen", "white", "firebrick3"))(100)
  

  pheatmap(
    mat = H_scaled,
    color = hm_cols,
    cluster_rows = hc_rows, 
    cluster_cols = TRUE, 
    annotation_row = annot_row,
    annotation_colors = ann_colors,
    show_rownames = FALSE,
    fontsize_col = 10,
    main = main_title,
    border_color = NA
  )
}


library(ggplot2)
library(dplyr)
library(tidyr)

plot_cosine_heatmap_with_gate <- function(
    Z,
    g_vec,
    n_nodes_max = 400,
    main = "Cosine similarity of Z\nordered by dendrogram; bar = gate g_i"
){
  Z <- as.matrix(Z)
  N <- nrow(Z)
  stopifnot(length(g_vec) == N)
  
  
  if (N > n_nodes_max) {
    set.seed(123)
    keep <- sort(sample(seq_len(N), n_nodes_max))
  } else {
    keep <- seq_len(N)
  }
  Z_sub <- Z[keep, , drop = FALSE]
  g_sub <- as.numeric(g_vec[keep])
  
  
  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z_sub)
  S <- Zn %*% t(Zn) 
  
 
  Dcos <- as.dist(1 - S)
  hc <- hclust(Dcos, method = "average")
  ord <- hc$order
  
  S_ord <- S[ord, ord]
  g_ord <- g_sub[ord]
  
  
  df_S <- as.data.frame(S_ord)
  colnames(df_S) <- seq_len(ncol(S_ord))
  df_S$rows <- seq_len(nrow(S_ord))
  df_S_long <- df_S %>%
    pivot_longer(
      cols = -rows,
      names_to = "cols",
      values_to = "sim"
    ) %>%
    mutate(
      rows = as.integer(rows),
      cols = as.integer(cols)
    )
  
  
  df_gate <- data.frame(
    node = seq_along(g_ord),
    gate = g_ord
  )
  
 
  pal_gate <- colorRampPalette(c("#2c105c", "#f28e2b", "#ffffb2"))
  n_col <- 100
  gate_palette <- pal_gate(n_col)
  g_rank <- rank(df_gate$gate, ties.method = "average") / nrow(df_gate)
  df_gate$gate_col_idx <- pmax(1, pmin(n_col, floor(g_rank * n_col)))
  df_gate$gate_col_hex <- gate_palette[df_gate$gate_col_idx]
  
 
  p_heat <- ggplot(df_S_long, aes(x = cols, y = rows, fill = sim)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "navy",
      mid = "white",
      high = "firebrick",
      midpoint = 0,
      name = "cosine\nsimilarity"
    ) +
    coord_fixed() +
    scale_y_reverse() +
    labs(
      title = main,
      x = "nodes (ordered by cosine dendrogram)",
      y = "nodes"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(hjust = 0.5),
      axis.text = element_blank(),
      axis.ticks = element_blank()
    )
  
 
  p_gate <- ggplot(df_gate, aes(x = node, y = 1, fill = gate)) +
    geom_tile() +
    scale_fill_gradient(
      low = "#2c105c",
      high = "#ffffb2",
      name = "gate g_i"
    ) +
    labs(
      x = "nodes (same order as heatmap)",
      y = ""
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = "right"
    )
  
 
  print(p_heat)
  print(p_gate)
}

plot_dendrogram_cosine_gate <- function(
    Z,
    g_vec,
    n_nodes_max = 800,
    main = "Cosine dendrogram from Z\nleaf colour = gate g_i"
){
  Z <- as.matrix(Z)
  N <- nrow(Z)
  stopifnot(length(g_vec) == N)
  
 
  if (N > n_nodes_max) {
    set.seed(123)
    keep <- sort(sample(seq_len(N), n_nodes_max))
  } else {
    keep <- seq_len(N)
  }
  Z_sub <- Z[keep, , drop = FALSE]
  g_sub <- as.numeric(g_vec[keep])
  

  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z_sub)
  S <- Zn %*% t(Zn)
  Dcos <- as.dist(1 - S)
  
  hc <- hclust(Dcos, method = "average")
  ord <- hc$order
  g_ord <- g_sub[ord]
  

  qcuts <- quantile(g_ord, probs = c(0.1, 0.3, 0.5, 0.7, 0.9))
  gate_bin <- cut(
    g_ord,
    breaks = c(-Inf, qcuts, Inf),
    include.lowest = TRUE,
    labels = c("q0.1","q0.3","q0.5","q0.7","q0.9","q1.0")
  )
  
  pal_gate <- c(
    "q0.1" = "#2c105c",
    "q0.3" = "#5b2ca0",
    "q0.5" = "#b63679",
    "q0.7" = "#f28e2b",
    "q0.9" = "#f5c04d",
    "q1.0" = "#ffffb2"
  )
  leaf_cols <- pal_gate[as.character(gate_bin)]
  
 
  op <- par(no.readonly = TRUE); on.exit(par(op))
  par(mfrow = c(2, 1))
  par(mar = c(1, 4, 4, 2) + 0.1) 
  

  plot(hc, labels = FALSE, hang = -1,
       main = main, xlab = "", sub = "")
  
  
  par(mar = c(4, 4, 1, 2) + 0.1) 
  plot(
    x = seq_along(g_ord),
    y = rep(0, length(g_ord)),
    xlab = "nodes (ordered by cosine dendrogram)",
    ylab = "",
    yaxt = "n",
    pch = 15,
    col = leaf_cols,
    cex = 1.1,
    xaxs = "i"
  )
  axis(1, at = pretty(seq_along(g_ord)))
  legend("topright",
         legend = names(pal_gate),
         pch = 15,
         col = pal_gate,
         title = "gate quantiles",
         cex = 0.7)
}

plot_cosine_sim_heatmap <- function(
    Z,
    g_vec,
    y = NULL,
    n_nodes_max = 300,
    main = "Cosine similarity between nodes\n(from Z_final)"
) {
  Z <- as.matrix(Z)
  N <- nrow(Z)
  stopifnot(length(g_vec) == N)
  
 
  if (N > n_nodes_max) {
    set.seed(123)
    keep_idx <- sort(sample(seq_len(N), n_nodes_max))
  } else {
    keep_idx <- seq_len(N)
  }
  
  Z_sub <- Z[keep_idx, , drop = FALSE]
  gate <- as.numeric(g_vec[keep_idx])
  if (!is.null(y)) y_sub <- y[keep_idx]
  
 
  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z_sub)
  S <- Zn %*% t(Zn) 
  
  
  node_ids <- as.character(keep_idx)
  rownames(S) <- node_ids
  colnames(S) <- node_ids
  
 
  Dcos <- as.dist(1 - S)
  hc <- hclust(Dcos, method = "average")
  
 
  qcuts <- quantile(gate, probs = c(0.2, 0.4, 0.6, 0.8))
  gate_bin <- cut(
    gate,
    breaks = c(-Inf, qcuts, Inf),
    include.lowest = TRUE,
    labels = c("q0.2","q0.4","q0.6","q0.8","q1.0")
  )
  
  anno_df <- data.frame(row.names = node_ids)
  
  if (!is.null(y)) {
    anno_df$Class <- factor(y_sub)
  }
  anno_df$GateBin <- gate_bin
  
 
  ann_colors <- list()
  
  if (!is.null(y)) {
    class_levels <- levels(anno_df$Class)
    ann_colors$Class <- setNames(
      brewer.pal(max(3, length(class_levels)), "Set1")[seq_along(class_levels)],
      class_levels
    )
  }
  
  gatebin_levels <- levels(gate_bin)
  ann_colors$GateBin <- setNames(
    brewer.pal(5, "PuOr")[seq_along(gatebin_levels)],
    gatebin_levels
  )
  
  
  sim_colors <- colorRampPalette(c("navy", "white", "firebrick"))(200)
  
 
  pheatmap(
    S,
    color = sim_colors,
    cluster_rows = hc,
    cluster_cols = hc,
    annotation_col = anno_df,
    annotation_row = NULL, 
    annotation_colors = ann_colors,
    show_rownames = FALSE,
    show_colnames = FALSE,
    border_color = NA,
    main = main,
    fontsize = 10,
    fontsize_row = 6,
    fontsize_col = 6
  )
}
