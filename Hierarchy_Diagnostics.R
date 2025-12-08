## =========================================================
## EXTENDED HIERARCHY DIAGNOSTICS
## =========================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(Rtsne)
  library(Matrix)
})

hierarchy_diagnostics <- function(
    Z_final, # [N, d] embeddings
    g_vec, # [N] gate g_i in [0,1]
    D_all, # [N] depth prior
    y, # [N] labels (0..K-1 or factor/character)
    C_all = NULL, # optional centrality prior
    A = NULL, # optional adjacency (matrix or sparse Matrix)
    pred = NULL, # optional predicted labels (same type/length as y)
    k_knn = 5L,
    use_tsne = TRUE,
    seed_tsne = 123L
){
  ## ---------------------------
  ## 0) Basic checks & label prep
  ## ---------------------------
  Z_final <- as.matrix(Z_final)
  N <- nrow(Z_final)
  stopifnot(length(g_vec) == N, length(D_all) == N, length(y) == N)
  if (!is.null(C_all)) stopifnot(length(C_all) == N)
  if (!is.null(pred)) stopifnot(length(pred) == N)
  
  if (is.numeric(y)) {
    y_factor <- factor(y)
  } else {
    y_factor <- factor(y)
  }
  if (!is.null(pred)) {
    if (is.numeric(pred)) pred_factor <- factor(pred) else pred_factor <- factor(pred)
    correct <- (pred_factor == y_factor)
  } else {
    pred_factor <- NULL
    correct <- NULL
  }
  
  ## ------------------------------------
  ## 1) Cosine KNN + label purity per node
  ## ------------------------------------
  cosine_knn_purity <- function(Z, y, k = 5L) {
    Z <- as.matrix(Z)
    N <- nrow(Z)
    norms <- sqrt(rowSums(Z^2)) + 1e-9
    Znorm <- Z / norms
    S <- Znorm %*% t(Znorm) # cosine sim
    
    knn_purity <- numeric(N)
    for (i in seq_len(N)) {
      sims <- S[i, ]
      sims[i] <- -Inf
      nn_idx <- order(sims, decreasing = TRUE)[seq_len(min(k, N - 1))]
      knn_purity[i] <- mean(y[nn_idx] == y[i])
    }
    knn_purity
  }
  
  knn_purity <- cosine_knn_purity(Z_final, y_factor, k = k_knn)
  
  ## ---------------------------
  ## 2) Global correlations
  ## ---------------------------
  cor_depth_gate <- suppressWarnings(cor(D_all, g_vec, use = "complete.obs"))
  cor_gate_knn <- suppressWarnings(cor(g_vec, knn_purity, use = "complete.obs"))
  if (!is.null(C_all)) {
    cor_central_gate <- suppressWarnings(cor(C_all, g_vec, use = "complete.obs"))
  } else {
    cor_central_gate <- NA_real_
  }
  
  cat("\n=== Hierarchy summary correlations ===\n")
  cat(sprintf("cor(depth, gate) = %.3f (expect negative: deeper ??? lower gate)\n", cor_depth_gate))
  cat(sprintf("cor(gate, KNN_purity) = %.3f (expect positive: higher gate ??? purer local neighborhood)\n", cor_gate_knn))
  if (!is.na(cor_central_gate)) {
    cat(sprintf("cor(centrality, gate) = %.3f\n", cor_central_gate))
  }
  if (!is.null(pred_factor)) {
    overall_acc <- mean(correct)
    cat(sprintf("overall accuracy(pred vs y) = %.3f\n", overall_acc))
  }
  cat("======================================\n\n")
  
  ## ---------------------------
  ## 3) Embedding coordinates
  ## ---------------------------
  if (ncol(Z_final) == 2L || !use_tsne) {
    emb_x <- Z_final[, 1]
    emb_y <- if (ncol(Z_final) >= 2L) Z_final[, 2] else rep(0, N)
    tsne_used <- FALSE
  } else {
    cat(sprintf("Running t-SNE on %d nodes (dim=%d)...\n", N, ncol(Z_final)))
    set.seed(seed_tsne)
    ts <- Rtsne(
      Z_final,
      perplexity = min(30, max(5, floor((N - 1) / 3))),
      check_duplicates = FALSE,
      verbose = TRUE
    )
    emb_x <- ts$Y[, 1]
    emb_y <- ts$Y[, 2]
    tsne_used <- TRUE
  }
  
  ## ---------------------------
  ## 4) Data frame for node-level plotting
  ## ---------------------------
  df <- data.frame(
    node = seq_len(N),
    depth = as.numeric(D_all),
    gate = as.numeric(g_vec),
    label = y_factor,
    knn_purity = as.numeric(knn_purity),
    emb_x = emb_x,
    emb_y = emb_y,
    stringsAsFactors = FALSE
  )
  if (!is.null(C_all)) {
    df$centrality <- as.numeric(C_all)
  }
  if (!is.null(pred_factor)) {
    df$pred <- pred_factor
    df$correct <- factor(correct, levels = c(FALSE, TRUE), labels = c("wrong", "correct"))
  }
  
  ## ---------------------------
  ## 5) Gate histogram + per-class density
  ## ---------------------------
  p_gate_hist <- ggplot(df, aes(x = gate)) +
    geom_histogram(bins = 30, fill = "grey70", colour = "grey20") +
    labs(x = "g_i", y = "count") +
    ggtitle("Distribution of hierarchy gate values g_i") +
    theme_minimal(base_size = 11)
  print(p_gate_hist)
  
  p_gate_density <- ggplot(df, aes(x = gate, colour = label)) +
    geom_density(adjust = 1.0) +
    labs(x = "g_i", y = "density", colour = "Class") +
    ggtitle("Gate distribution per class") +
    theme_minimal(base_size = 11)
  print(p_gate_density)
  
  ## ---------------------------
  ## 6) Gate bins summary (depth, purity, accuracy)
  ## ---------------------------
  df$gate_bin <- cut(
    df$gate,
    breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
    labels = c("(0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1.0]")
  )
  
  gate_bin_stats <- df %>%
    group_by(gate_bin) %>%
    summarise(
      n = n(),
      mean_gate = mean(gate, na.rm = TRUE),
      mean_depth = mean(depth, na.rm = TRUE),
      mean_knn = mean(knn_purity, na.rm = TRUE),
      .groups = "drop"
    )
  
  if (!is.null(pred_factor)) {
    gate_bin_acc <- df %>%
      group_by(gate_bin) %>%
      summarise(
        n = n(),
        acc = mean(correct == "correct"),
        .groups = "drop"
      )
    gate_bin_stats <- left_join(gate_bin_stats, gate_bin_acc, by = "gate_bin")
  }
  
  cat("=== Gate-bin summary (binned by g_i) ===\n")
  print(gate_bin_stats)
  cat("========================================\n\n")
  
  # Plot: gate bin vs KNN purity / depth
  p_gatebin_knn <- ggplot(gate_bin_stats, aes(x = gate_bin, y = mean_knn, group = 1)) +
    geom_line() + geom_point(size = 2) +
    labs(x = "gate bin", y = "mean KNN purity", title = "Mean KNN purity per gate bin") +
    theme_minimal(base_size = 11)
  print(p_gatebin_knn)
  
  p_gatebin_depth <- ggplot(gate_bin_stats, aes(x = gate_bin, y = mean_depth, group = 1)) +
    geom_line() + geom_point(size = 2) +
    labs(x = "gate bin", y = "mean depth D_i", title = "Mean depth per gate bin") +
    theme_minimal(base_size = 11)
  print(p_gatebin_depth)
  
  if (!is.null(pred_factor) && "acc" %in% colnames(gate_bin_stats)) {
    p_gatebin_acc <- ggplot(gate_bin_stats, aes(x = gate_bin, y = acc, group = 1)) +
      geom_line() + geom_point(size = 2) +
      labs(x = "gate bin", y = "accuracy", title = "Accuracy per gate bin") +
      theme_minimal(base_size = 11)
    print(p_gatebin_acc)
  }
  
  ## ---------------------------
  ## 7) Node-level scatter plots
  ## ---------------------------
  # Depth vs gate
  p_depth_gate <- ggplot(df, aes(x = depth, y = gate, colour = label)) +
    geom_point(alpha = 0.6, size = 1.0) +
    geom_smooth(method = "loess", se = FALSE, colour = "black", linewidth = 0.7) +
    labs(
      x = "Depth prior D_i",
      y = "Hierarchy gate g_i",
      colour = "Class"
    ) +
    ggtitle("Depth vs hierarchy gate (D_i vs g_i)") +
    theme_minimal(base_size = 11)
  print(p_depth_gate)
  
  # Embedding coloured by gate
  title_emb <- if (tsne_used) {
    "t-SNE of final embeddings coloured by gate g_i"
  } else {
    "Embeddings (first 2 dims) coloured by gate g_i"
  }
  
  p_emb_gate <- ggplot(df, aes(x = emb_x, y = emb_y, colour = gate)) +
    geom_point(alpha = 0.8, size = 1.2) +
    scale_colour_gradient(low = "darkblue", high = "yellow") +
    labs(
      x = if (tsne_used) "t-SNE 1" else "Embedding dim 1",
      y = if (tsne_used) "t-SNE 2" else "Embedding dim 2",
      colour = "g_i"
    ) +
    ggtitle(title_emb) +
    theme_minimal(base_size = 11)
  print(p_emb_gate)
  
  # Gate vs KNN purity
  p_gate_knn <- ggplot(df, aes(x = gate, y = knn_purity, colour = label)) +
    geom_point(alpha = 0.7, size = 1.2) +
    geom_smooth(method = "lm", se = FALSE, colour = "black", linewidth = 0.6) +
    labs(
      x = "Hierarchy gate g_i",
      y = sprintf("KNN label purity (cosine, k = %d)", k_knn),
      colour = "Class"
    ) +
    ggtitle("Gate vs local label consistency (KNN purity)") +
    theme_minimal(base_size = 11)
  print(p_gate_knn)
  
  # Centrality vs gate (if provided)
  if (!is.null(C_all)) {
    p_central_gate <- ggplot(df, aes(x = centrality, y = gate, colour = label)) +
      geom_point(alpha = 0.6, size = 1.0) +
      labs(
        x = "Centrality prior C_i",
        y = "Hierarchy gate g_i",
        colour = "Class"
      ) +
      ggtitle("Centrality vs hierarchy gate (C_i vs g_i)") +
      theme_minimal(base_size = 11)
    print(p_central_gate)
  }
  
  # Prediction-based plots (if pred given)
  if (!is.null(pred_factor)) {
    # Embedding coloured by correctness
    p_emb_correct <- ggplot(df, aes(x = emb_x, y = emb_y, colour = correct)) +
      geom_point(alpha = 0.9, size = 1.3) +
      labs(
        x = if (tsne_used) "t-SNE 1" else "Embedding dim 1",
        y = if (tsne_used) "t-SNE 2" else "Embedding dim 2",
        colour = "Prediction"
      ) +
      ggtitle("Embeddings coloured by prediction correctness") +
      theme_minimal(base_size = 11)
    print(p_emb_correct)
    
    # Gate vs correctness (scatter w/ jitter)
    p_gate_correct <- ggplot(df, aes(x = correct, y = gate)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.7) +
      geom_jitter(width = 0.1, alpha = 0.25, size = 0.6) +
      labs(x = "Prediction", y = "g_i") +
      ggtitle("Gate values for correct vs wrong predictions") +
      theme_minimal(base_size = 11)
    print(p_gate_correct)
  }
  
  ## ---------------------------
  ## 8) Edge-level gate smoothness (if A provided)
  ## ---------------------------
  edge_stats <- NULL
  p_edge_gate <- NULL
  
  if (!is.null(A)) {
    if (inherits(A, "Matrix")) {
      A_mat <- A
    } else {
      A_mat <- as.matrix(A)
      A_mat <- Matrix(A_mat, sparse = TRUE)
    }
    
    # undirected edges i<j
    edges <- which(A_mat != 0, arr.ind = TRUE)
    if (nrow(edges) > 0) {
      edges <- edges[edges[,1] < edges[,2], , drop = FALSE]
      n_edges <- nrow(edges)
      
      g_i <- g_vec[edges[,1]]
      g_j <- g_vec[edges[,2]]
      d_i <- D_all[edges[,1]]
      d_j <- D_all[edges[,2]]
      same_label_edge <- (y_factor[edges[,1]] == y_factor[edges[,2]])
      
      gate_diff <- abs(g_i - g_j)
      depth_diff <- abs(d_i - d_j)
      
      mean_gate_diff <- mean(gate_diff)
      mean_gate_same <- mean(gate_diff[same_label_edge])
      mean_gate_diff_lbl <- mean(gate_diff[!same_label_edge])
      mean_depth_diff <- mean(depth_diff)
      
      edge_stats <- list(
        n_edges = n_edges,
        mean_gate_diff = mean_gate_diff,
        mean_gate_diff_same_label = mean_gate_same,
        mean_gate_diff_diff_label = mean_gate_diff_lbl,
        mean_depth_diff = mean_depth_diff
      )
      
      cat("=== Edge-level gate smoothness ===\n")
      cat(sprintf("number of edges : %d\n", n_edges))
      cat(sprintf("mean |g_i - g_j| over edges : %.3f\n", mean_gate_diff))
      cat(sprintf("mean |g_i - g_j| (same label) : %.3f\n", mean_gate_same))
      cat(sprintf("mean |g_i - g_j| (different label): %.3f\n", mean_gate_diff_lbl))
      cat(sprintf("mean |D_i - D_j| over edges : %.3f\n", mean_depth_diff))
      cat("==================================\n\n")
      
      # scatter of gate_i vs gate_j on a subset of edges
      if (n_edges > 0) {
        max_plot_edges <- min(4000L, n_edges)
        idx_sample <- sample(seq_len(n_edges), max_plot_edges)
        df_edge <- data.frame(
          g_i = g_i[idx_sample],
          g_j = g_j[idx_sample],
          same_label = factor(
            same_label_edge[idx_sample],
            levels = c(FALSE, TRUE),
            labels = c("diff_label", "same_label")
          )
        )
        
        p_edge_gate <- ggplot(df_edge, aes(x = g_i, y = g_j, colour = same_label)) +
          geom_point(alpha = 0.4, size = 0.7) +
          labs(
            x = "g_i (source)",
            y = "g_j (neighbor)",
            colour = "Edge type"
          ) +
          ggtitle("Gate smoothness along edges (g_i vs g_j)") +
          theme_minimal(base_size = 11)
        print(p_edge_gate)
      }
    }
  }
  
  invisible(list(
    df = df,
    gate_bin_stats = gate_bin_stats,
    cor_depth_gate = cor_depth_gate,
    cor_gate_knn = cor_gate_knn,
    cor_central_gate = cor_central_gate,
    edge_stats = edge_stats
  ))
}


############################################################
## 1) Build full-graph inputs for your trained model
############################################################

build_full_inputs <- function(model, X_features, A, C_vec, D_vec, M_mat = NULL) {
  n_nodes <- nrow(X_features)
  n_features <- ncol(X_features)
  
  X_in <- array(X_features, dim = c(1, n_nodes, n_features))
  A_in <- array(as.matrix(A), dim = c(1, n_nodes, n_nodes))
  C_in <- array(C_vec, dim = c(1, n_nodes, 1))
  D_in <- array(D_vec, dim = c(1, n_nodes, 1))
  
  # If your model has a 5th input (M mask), pass full ones
  if (length(model$inputs) == 4L) {
    list(X_in, A_in, C_in, D_in)
  } else if (length(model$inputs) == 5L) {
    if (is.null(M_mat)) {
      M_mat <- matrix(1, n_nodes, n_nodes)
      diag(M_mat) <- 0
    }
    M_in <- array(M_mat, dim = c(1, n_nodes, n_nodes))
    list(X_in, A_in, C_in, D_in, M_in)
  } else {
    stop("Unexpected number of model inputs: ", length(model$inputs))
  }
}

############################################################
## 2) Get g_vec: outputs of your Hierarchy Gate layer
############################################################

get_hierarchy_gate_outputs <- function(model,
                                       inputs,
                                       layer_pattern = "hierarchy_gate|hierarch_gate") {
  layer_names <- sapply(model$layers, function(l) l$name)
  idx <- which(grepl(layer_pattern, layer_names, ignore.case = TRUE))[1]
  if (is.na(idx)) stop("Could not find HierarchyGate layer in model.")
  
  hg_layer <- model$layers[[idx]]
  hg_model <- keras::keras_model(inputs = model$inputs, outputs = hg_layer$output)
  H_pred <- predict(hg_model, inputs)
  
  # Expected shapes: (1, N, 1) or (N, 1)
  if (length(dim(H_pred)) == 3L) {
    g_vec <- as.numeric(H_pred[1, , 1])
  } else if (length(dim(H_pred)) == 2L) {
    g_vec <- as.numeric(H_pred[, 1])
  } else {
    stop("Unexpected HierarchyGate tensor shape: ", paste(dim(H_pred), collapse = " x "))
  }
  g_vec
}

############################################################
## 3) Get Z_final: final node embeddings from the model
############################################################

get_final_embeddings <- function(model, inputs, layer_name = NULL) {
  layer_names <- sapply(model$layers, function(l) l$name)
  
  # Option 1: you know the exact embedding layer name
  if (!is.null(layer_name)) {
    if (!layer_name %in% layer_names) {
      stop("Embedding layer name '", layer_name, "' not found in model.")
    }
    emb_layer_name <- layer_name
  } else {
    # Option 2: use penultimate layer by default
    L <- length(model$layers)
    emb_layer_name <- model$layers[[L - 1L]]$name
  }
  
  cat("Using embedding layer:", emb_layer_name, "\n")
  
  emb_model <- keras::keras_model(
    inputs = model$inputs,
    outputs = keras::get_layer(model, emb_layer_name)$output
  )
  
  Z_pred <- predict(emb_model, inputs)
  
  # Expected shapes: (1, N, d) or (N, d)
  if (length(dim(Z_pred)) == 3L) {
    Z_final <- Z_pred[1, , ]
  } else if (length(dim(Z_pred)) == 2L) {
    Z_final <- Z_pred
  } else {
    stop("Unexpected embedding tensor shape: ", paste(dim(Z_pred), collapse = " x "))
  }
  as.matrix(Z_final) # N x d
}
