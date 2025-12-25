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


get_stage_embeddings <- function(model, inputs_all, t) {
  layer_Zt <- get_layer(model, sprintf("DeepBlock_t%d", t))
  sub_Zt <- keras_model(inputs = model$inputs,
                        outputs = layer_Zt$output)
  Z_pred <- predict(sub_Zt, inputs_all)
  Z_t <- Z_pred[1,,] # shape N x d_t
  Z_t
}

## Optional: infer T from layer names (or just pass stages = 1:4 etc.)
infer_num_stages <- function(model) {
  lname <- vapply(model$layers, function(l) l$name, "")
  max(as.integer(gsub("DeepBlock_t", "", grep("^DeepBlock_t[0-9]+$", lname, value = TRUE))))
}


############################################################
## FIXED: stage-wise hierarchy diagnostics
## - no longer assumes diag_t$cor_summary exists
## - computes correlations directly
############################################################

run_stage_hierarchy_diagnostics <- function(
    model,
    inputs_all,
    g_vec,
    D_all,
    C_all,
    y,
    A_raw,
    stages = NULL,
    k_knn = 32L,
    do_tsne = FALSE
){
  if (is.null(stages)) {
    stages <- seq_len(infer_num_stages(model))
  }
  
  N <- length(y)
  stopifnot(length(g_vec) == N,
            nrow(D_all) == N,
            nrow(C_all) == N)
  
  diag_list <- list()
  summary_tbl <- list()
  
  depth_vec <- as.numeric(D_all[,1])
  cent_vec <- as.numeric(C_all[,1])
  
  for (t in stages) {
    cat(sprintf("\n========== Stage t = %d ==========\n", t))
    
    Z_t <- get_stage_embeddings(model, inputs_all, t)
    
    diag_t <- hierarchy_diagnostics(
      Z_final = Z_t,
      g_vec = g_vec,
      D_all = D_all,
      y = y,
      C_all = C_all,
      A = A_raw,
      pred = NULL,
      k_knn = as.integer(k_knn),
      use_tsne = do_tsne
    )
    
    diag_list[[paste0("t", t)]] <- diag_t
    
    ## --- correlations we care about ---
    ## depth vs gate and centrality vs gate don't depend on stage
    cor_depth_gate <- suppressWarnings(cor(depth_vec, g_vec))
    cor_central_gate <- suppressWarnings(cor(cent_vec, g_vec))
    
    ## gate vs KNN-purity DOES depend on Z_t
    if (!is.null(diag_t$knn_purity)) {
      cor_gate_knn <- suppressWarnings(cor(g_vec, diag_t$knn_purity))
    } else {
      cor_gate_knn <- NA_real_
    }
    
    ## edge-level smoothness (if available)
    if (!is.null(diag_t$edge_stats)) {
      mean_edge_gate_diff <- diag_t$edge_stats$mean_gate_diff
      mean_edge_gate_diff_same <- diag_t$edge_stats$mean_gate_diff_same_label
      mean_edge_gate_diff_diff <- diag_t$edge_stats$mean_gate_diff_diff_label
    } else {
      mean_edge_gate_diff <- NA_real_
      mean_edge_gate_diff_same <- NA_real_
      mean_edge_gate_diff_diff <- NA_real_
    }
    
    summary_tbl[[length(summary_tbl) + 1]] <- data.frame(
      stage = t,
      cor_depth_gate = cor_depth_gate,
      cor_gate_knn = cor_gate_knn,
      cor_central_gate = cor_central_gate,
      mean_edge_gate_diff = mean_edge_gate_diff,
      mean_edge_gate_diff_same = mean_edge_gate_diff_same,
      mean_edge_gate_diff_diff = mean_edge_gate_diff_diff
    )
  }
  
  summary_tbl <- dplyr::bind_rows(summary_tbl)
  list(
    diag_per_stage = diag_list,
    summary_per_stage = summary_tbl
  )
}


plot_dendrogram_from_Z_and_gate <- function(
    Z,
    g_vec,
    y = NULL,
    n_subsample = 800L,
    main = "Dendrogram from Z, leaf colour = gate g_i"
){
  set.seed(1)
  N <- nrow(Z)
  idx <- if (N > n_subsample) sample.int(N, n_subsample) else seq_len(N)
  
  Z_sub <- Z[idx, , drop = FALSE]
  g_sub <- g_vec[idx]
  
  # cosine distance
  row_l2 <- function(M) { M / pmax(sqrt(rowSums(M^2)), 1e-8) }
  Zn <- row_l2(Z_sub)
  cos_sim <- tcrossprod(Zn)
  D_cos <- as.dist(1 - cos_sim)
  
  hc <- hclust(D_cos, method = "average")
  
  # ----- SAFE BINNING OF g_sub -----
  qprobs <- c(0.1, 0.3, 0.5, 0.7, 0.9)
  qcuts <- as.numeric(quantile(g_sub, probs = qprobs, na.rm = TRUE))
  
  brks <- unique(c(-Inf, qcuts, Inf))
  n_int <- length(brks) - 1L
  
  base_labels <- paste0("q", qprobs)
  lab <- base_labels[seq_len(n_int)]
  
  g_disc <- cut(
    g_sub,
    breaks = brks,
    labels = lab,
    include.lowest = TRUE
  )
  
  # order leaves according to hclust
  ord <- hc$order
  g_ord <- g_disc[ord]
  
  # small colour palette for the bins
  pal <- c(
    "q0.1" = "black",
    "q0.3" = "purple",
    "q0.5" = "red",
    "q0.7" = "orange",
    "q0.9" = "yellow"
  )
  pal <- pal[names(pal) %in% levels(g_ord)]
  
  # plot
  op <- par(no.readonly = TRUE)
  on.exit(par(op))
  
  par(mar = c(6, 4, 4, 2) + 0.1, xpd = TRUE)
  plot(hc, labels = FALSE, main = main, xlab = "subsampled nodes")
  
  # coloured points under leaves
  y0 <- par("usr")[3]
  points(
    x = seq_along(ord),
    y = rep(y0 - 0.02 * diff(par("usr")[3:4]), length(ord)),
    pch = 16,
    col = pal[as.character(g_ord)],
    cex = 0.7
  )
  
  legend("topright",
         legend = names(pal),
         col = pal,
         pch = 16,
         title = "gate g_i")
}
############################################################
## 3) Heatmap of priors ordered by Z^{(t)} clustering
############################################################

plot_priors_heatmap_for_Z <- function(
    Z,
    D_all,
    C_all,
    g_vec,
    y = NULL,
    main = "Nodes clustered by Z (priors + gate heatmap)"
){
  # Z: N x d
  N <- nrow(Z)
  stopifnot(
    length(g_vec) == N,
    (is.vector(D_all) || nrow(D_all) == N),
    (is.vector(C_all) || nrow(C_all) == N)
  )
  
  # --- 1) Vectorize priors -------------------------------------------
  depth <- if (is.matrix(D_all)) as.numeric(D_all[, 1]) else as.numeric(D_all)
  centrality <- if (is.matrix(C_all)) as.numeric(C_all[, 1]) else as.numeric(C_all)
  gate <- as.numeric(g_vec)
  
  # --- 2) Cluster nodes by embeddings Z (cosine distance) ------------
  row_l2 <- function(M) M / pmax(sqrt(rowSums(M^2)), 1e-8)
  Zn <- row_l2(Z) # N x d, L2-normalized rows
  cosSim <- tcrossprod(Zn) # N x N
  D_cos <- as.dist(1 - cosSim) # dissimilarity
  
  hc <- hclust(D_cos, method = "average")
  ord <- hc$order # permutation of nodes
  
  # --- 3) Build priors matrix H: rows = {depth, centrality, gate} ----
  H <- rbind(
    depth = depth,
    centrality = centrality,
    gate = gate
  ) # 3 x N
  
  # z-score rows so colour scales are comparable
  H_scaled <- H
  for (i in seq_len(nrow(H_scaled))) {
    mu <- mean(H_scaled[i, ], na.rm = TRUE)
    sdv <- stats::sd(H_scaled[i, ], na.rm = TRUE)
    if (sdv > 0) {
      H_scaled[i, ] <- (H_scaled[i, ] - mu) / sdv
    } else {
      H_scaled[i, ] <- H_scaled[i, ] - mu
    }
  }
  
  # reorder columns by embedding-based cluster order
  H_ord <- H_scaled[, ord, drop = FALSE] # 3 x N
  
  # --- 4) Set up plotting layout & margins safely --------------------
  op <- par(no.readonly = TRUE)
  on.exit(par(op))
  
  if (!is.null(y)) {
    layout(
      matrix(c(1, 2), nrow = 2, byrow = TRUE),
      heights = c(4, 0.8) # top panel taller, bottom stripe short
    )
  } else {
    layout(matrix(1, nrow = 1))
  }
  
  # --- Panel 1: priors + gate heatmap -------------------------------
  par(mar = c(3, 5, 3, 2)) # bottom, left, top, right
  
  image(
    z = H_ord[nrow(H_ord):1, , drop = FALSE], # flip rows so gate on top
    col = gray.colors(200),
    axes = FALSE,
    xlab = "nodes (clustered by Z)",
    ylab = ""
  )
  title(main, line = 1)
  
  # x-axis: index ticks (not too many)
  axis(1,
       at = seq(0, 1, length.out = 6),
       labels = round(seq(1, N, length.out = 6)))
  
  # y-axis: priors names (depth, centrality, gate)
  axis(2,
       at = seq(0, 1, length.out = nrow(H_ord)),
       labels = rev(rownames(H_ord)),
       las = 1)
  
  # --- Panel 2: class stripe (if y provided) -------------------------
  if (!is.null(y)) {
    par(mar = c(2, 5, 1, 2)) # smaller margins for stripe
    
    y_fac <- as.factor(y[ord])
    y_int <- as.integer(y_fac)
    pal <- rainbow(length(levels(y_fac)))
    
    # wrap in tryCatch so tiny devices don't crash the whole analysis
    tryCatch({
      image(
        z = matrix(y_int, nrow = 1),
        col = pal,
        axes = FALSE,
        xlab = "nodes (same ordering as heatmap above)",
        ylab = ""
      )
      
      axis(1,
           at = seq(0, 1, length.out = 6),
           labels = round(seq(1, N, length.out = 6)))
      axis(2, at = 0.5, labels = "class", las = 1)
      
      legend("topright",
             legend = levels(y_fac),
             fill = pal,
             cex = 0.6,
             bty = "n")
    }, error = function(e) {
      message("Skipping class stripe panel (device too small): ", e$message)
      # do nothing else; heatmap above is already drawn
    })
  }
}

############################################################
## 4) MST tree from Z^{(t)} with colouring
############################################################

library(igraph)
library(ggraph)

plot_mst_tree_from_embeddings <- function(
    Z,
    color_vec,
    colour_name = "gate",
    main = "MST tree from Z"
){
  N <- nrow(Z)
  Zn <- Z / pmax(sqrt(rowSums(Z^2)), 1e-8)
  S <- Zn %*% t(Zn)
  D <- 1 - S
  diag(D) <- 0
  
  g_full <- graph_from_adjacency_matrix(D, mode = "undirected", weighted = TRUE)
  mst_g <- mst(g_full, weights = E(g_full)$weight)
  
  V(mst_g)$color_val <- color_vec
  
  ggraph(mst_g, layout = "tree") +
    geom_edge_link(alpha = 0.4, colour = "grey70") +
    geom_node_point(aes(colour = color_val), size = 2) +
    scale_colour_viridis_c(option = "magma") +
    ggtitle(main) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
}
############################################################
## FIXED: driver that calls the diagnostics + plots
############################################################

analyze_all_stages <- function(
    model,
    inputs_all,
    g_vec,
    D_all,
    C_all,
    y,
    A_raw,
    stages = NULL,
    k_knn = 32L
){
  if (is.null(stages)) stages <- seq_len(infer_num_stages(model))
  
  res <- run_stage_hierarchy_diagnostics(
    model = model,
    inputs_all = inputs_all,
    g_vec = g_vec,
    D_all = D_all,
    C_all = C_all,
    y = y,
    A_raw = A_raw,
    stages = stages,
    k_knn = k_knn,
    do_tsne = FALSE
  )
  
  cat("\n=== Stage-wise summary (numeric) ===\n")
  print(res$summary_per_stage)
  
  ## ---- visualisations per stage ----
  for (t in stages) {
    cat(sprintf("\n\n########## VISUALS FOR STAGE t = %d ##########\n", t))
    Z_t <- get_stage_embeddings(model, inputs_all, t)
    
    ## 1) dendrogram + gate colours
    plot_dendrogram_from_Z_and_gate(
      Z = Z_t,
      g_vec = g_vec,
      y = y,
      main = sprintf("Dendrogram from Z_t (t = %d), leaf colour = gate g_i", t)
    )
    
    ## 2) priors heatmap ordered by Z_t clustering
    plot_priors_heatmap_for_Z(
      Z = Z_t,
      D_all = D_all,
      C_all = C_all,
      g_vec = g_vec,
      y = y,
      main = sprintf("Nodes clustered by Z_t (t = %d)", t)
    )
    
    ## 3) MST coloured by gate
    print(
      plot_mst_tree_from_embeddings(
        Z = Z_t,
        color_vec = g_vec,
        colour_name = "gate",
        main = sprintf("MST from Z_t (t = %d), node colour = gate g_i", t)
      )
    )
    
    ## 4) MST coloured by depth
    print(
      plot_mst_tree_from_embeddings(
        Z = Z_t,
        color_vec = as.numeric(D_all[,1]),
        colour_name = "depth",
        main = sprintf("MST from Z_t (t = %d), node colour = depth D_i", t)
      )
    )
  }
  
  invisible(res)
}