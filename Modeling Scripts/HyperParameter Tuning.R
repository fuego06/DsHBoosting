
make_stratified_kfolds <- function(y, k_folds = 5L, seed = 7L) {
  set.seed(seed)
  N <- length(y)
  k_folds <- max(1L, min(as.integer(k_folds), N))
  fold_id <- integer(N)
  for (cls in sort(unique(y))) {
    ids <- which(y == cls)
    if (!length(ids)) next
    ids <- sample(ids)
    assign <- rep_len(seq_len(k_folds), length(ids))
    fold_id[ids] <- assign
  }
  folds <- split(seq_len(N), factor(fold_id, levels = seq_len(k_folds)))
  lapply(folds, function(v) if (is.null(v)) integer(0) else as.integer(v))
}

one_hot_batch <- function(labels_raw, Kc, N) {
  yNK <- keras::to_categorical(as.integer(labels_raw), num_classes = Kc)
  array_reshape(yNK, c(1, N, Kc))
}

graph_feats_subgraph <- function(A_masked) {
  N <- nrow(A_masked)
  g <- igraph::graph_from_adjacency_matrix(A_masked, mode = "undirected", diag = FALSE)
  deg <- igraph::degree(g)
  C_vec <- if (max(deg) > 0) as.numeric(deg)/max(deg) else rep(0, N)
  hub <- if (any(deg > 0)) which.max(deg) else 1
  D <- igraph::distances(g, v = hub, to = igraph::V(g))
  finiteMax <- if (any(is.finite(D))) max(D[is.finite(D)]) else 0
  D[!is.finite(D)] <- finiteMax
  D_vec <- if (finiteMax > 0) as.numeric(D)/finiteMax else rep(0, N)
  list(C = C_vec, D = D_vec)
}

build_inputs_subgraph <- function(X, A_phase, C_vec, D_vec, M_mat) {
  N <- nrow(X); Fdim <- ncol(X)
  stopifnot(all(dim(A_phase) == c(N,N)), all(dim(M_mat) == c(N,N)))
  if (length(C_vec) == 1L) C_vec <- rep(C_vec, N)
  if (length(D_vec) == 1L) D_vec <- rep(D_vec, N)
  stopifnot(length(C_vec) == N, length(D_vec) == N)
  list(
    array_reshape(X, c(1, N, Fdim)),
    array_reshape(A_phase, c(1, N, N)),
    array_reshape(C_vec, c(1, N, 1)),
    array_reshape(D_vec, c(1, N, 1)),
    array_reshape(M_mat, c(1, N, N))
  )
}

predict_probs_only <- function(model, inputs_base) {
  p <- predict(model, inputs_base)
  if (is.list(p)) p <- p[[1]]
  if (length(dim(p)) == 2L) p <- array_reshape(p, c(1, nrow(p), ncol(p)))
  p[1,,]
}

masked_acc <- function(pNK, y_1NK_or_NK) {
  # pNK: [N,K]; y can be [1,N,K] or [N,K]
  if (length(dim(y_1NK_or_NK)) == 3L) yNK <- y_1NK_or_NK[1,,]
  else yNK <- y_1NK_or_NK
  pred <- max.col(pNK) - 1L
  true <- max.col(yNK) - 1L
  mean(pred == true)
}

# robust cartesian product for param grid (keeps vector/list entries)
expand_param_grid <- function(grid) {
  if (is.data.frame(grid)) return(grid)
  keys <- names(grid)
  combos <- list(list())
  for (k in keys) {
    vals <- grid[[k]]
    # keep list values as-is (e.g., init_shrink vectors)
    if (!is.list(vals)) vals <- as.list(vals)
    new_combos <- list()
    for (cst in combos) {
      for (v in vals) new_combos[[length(new_combos)+1]] <- c(cst, setNames(list(v), k))
    }
    combos <- new_combos
  }
  # turn into data.frame with list-cols preserved
  as.data.frame(do.call(rbind, lapply(combos, function(x) {
    # store list entries in I(list(...)) to preserve them
    out <- lapply(x, function(v) if (is.list(v)) I(list(v)) else v)
    out
  })), stringsAsFactors = FALSE, check.names = FALSE)
}

normalize_init_shrink <- function(s, T_stages) {
  if (is.list(s)) s <- s[[1]]
  s <- as.numeric(s)
  if (!length(s)) s <- 0.8
  if (length(s) == 1L) {
    rep(s, T_stages)
  } else if (length(s) < T_stages) {
    c(s, rep(tail(s, 1L), T_stages - length(s))) # pad with last
  } else {
    s[seq_len(T_stages)] # truncate
  }
}

# simple boosting-style trainer that uses ALL train subgraph nodes uniformly (no inner val)
trainer_boost_on_all <- function(
    model, inputs_train, y_batch,
    rounds = 3L, epochs_per_round = 60L,
    hard_thresh = 0.6, up_factor = 3, verbose = 1L
){
  N <- dim(y_batch)[2]; n_out <- length(model$outputs)
  to_w <- function(v) array(as.numeric(v), dim = c(1, N, 1))
  w <- rep(1/N, N)
  for (r in seq_len(rounds)) {
    if (verbose) cat(sprintf("\n[trainer] round %d/%d\n", r, rounds))
    sw <- rep(list(to_w(w)), n_out)
    model %>% fit(
      x = inputs_train, y = rep(list(y_batch), n_out),
      sample_weight = sw,
      batch_size = 1, epochs = epochs_per_round,
      shuffle = FALSE, verbose = verbose
    )
    p <- predict_probs_only(model, inputs_train)
    true_idx <- max.col(y_batch[1,,])
    pt <- p[cbind(seq_len(N), true_idx)]
    hardness <- 1 - pt
    hard <- as.numeric(hardness >= hard_thresh)
    w <- w * (1 + (up_factor - 1) * hard)
    w <- w / sum(w)
    if (verbose) cat(sprintf(" upweighted %.1f%%\n", 100*mean(hard > 0)))
  }
  model
}

# ===============================
# 1) Main CV function (INDUCTIVE)
# ===============================
cv_inductive_hparam <- function(
    X, A, y_raw,
    param_grid,
    out_csv,
    k_folds = 5L,
    seed = 7L,
    rounds = 5L,
    epochs_per_round = 120L,
    hard_thresh = 0.65,
    up_factor = 4.0,
    verbose = 1L,
    resume = TRUE # NEW: resume from existing CSV
){
  if (!exists("build_e2e_boost_noleak"))
    stop("build_e2e_boost_noleak() must be defined in your session.")
  
  stopifnot(is.matrix(X), is.matrix(A), length(y_raw) == nrow(X), nrow(A) == nrow(X))
  N <- nrow(X); Fdim <- ncol(X); Kc <- length(unique(y_raw))
  A <- (A > 0) | t(A > 0); diag(A) <- 0; A <- +A
  
  folds <- make_stratified_kfolds(y_raw, k_folds = k_folds, seed = seed)
  grid_df <- expand_param_grid(param_grid)
  
  # ---- read completed work if resuming ----
  done <- data.frame()
  if (resume && file.exists(out_csv) && file.info(out_csv)$size > 0) {
    # header could be arbitrary???just read generically
    done <- tryCatch(read.csv(out_csv, stringsAsFactors = FALSE, check.names = FALSE),
                     error = function(e) data.frame())
  }
  
  # prepare CSV header once if file doesn't exist
  if (!file.exists(out_csv) || (is.data.frame(done) && nrow(done) == 0)) {
    hdr <- c("trial_id","fold","N_train","N_val","acc_val",
             "rounds","epochs_per_round","hard_thresh","up_factor","seed",
             colnames(grid_df), "init_shrink_used")
    write.table(t(hdr), out_csv, sep = ",", row.names = FALSE, col.names = FALSE, quote = TRUE)
  }
  
  # helper (already in your env from earlier message; included here for self-containment)
  normalize_init_shrink <- function(s, T_stages) {
    if (is.list(s)) s <- s[[1]]
    s <- as.numeric(s)
    if (!length(s)) s <- 0.8
    if (length(s) == 1L) rep(s, T_stages)
    else if (length(s) < T_stages) c(s, rep(tail(s, 1L), T_stages - length(s)))
    else s[seq_len(T_stages)]
  }
  
  for (tidx in seq_len(nrow(grid_df))) {
    conf <- grid_df[tidx, , drop = FALSE]
    
    T_stages <- if ("T" %in% names(conf)) as.integer(conf$T) else 3L
    k_top <- if ("k_top" %in% names(conf)) as.integer(conf$k_top) else 8L
    tau_cos <- if ("tau_cos" %in% names(conf)) as.numeric(conf$tau_cos) else 0.2
    use_cluster_gate <- if ("use_cluster_gate" %in% names(conf)) as.logical(conf$use_cluster_gate) else TRUE
    residual_depth_blend <- if ("residual_depth_blend" %in% names(conf)) as.logical(conf$residual_depth_blend) else TRUE
    bdim <- if ("block_feature_dim" %in% names(conf)) as.integer(conf$block_feature_dim) else 64L
    bdepth <- if ("block_depth" %in% names(conf)) as.integer(conf$block_depth) else 4L
    bdrop <- if ("block_dropout" %in% names(conf)) as.numeric(conf$block_dropout) else 0.10
    lr <- if ("lr" %in% names(conf)) as.numeric(conf$lr) else 1e-3
    l2 <- if ("l2" %in% names(conf)) as.numeric(conf$l2) else 1e-8
    clipnorm <- if ("clipnorm" %in% names(conf)) as.numeric(conf$clipnorm) else 1.0
    use_focal <- if ("use_focal" %in% names(conf)) as.logical(conf$use_focal) else TRUE
    focal_gamma <- if ("focal_gamma" %in% names(conf)) as.numeric(conf$focal_gamma) else 2.0
    
    if ("init_shrink" %in% names(conf)) {
      init_shrink <- normalize_init_shrink(conf$init_shrink, T_stages)
    } else if (any(grepl("^init_shrink_", names(conf)))) {
      v <- as.numeric(conf[, grep("^init_shrink_", names(conf)), drop = TRUE])
      init_shrink <- normalize_init_shrink(v, T_stages)
    } else {
      init_shrink <- rep(0.8, T_stages)
    }
    init_shrink_str <- paste(sprintf("%.4g", init_shrink), collapse = ";")
    
    if (verbose) cat(sprintf("\n[CV] Trial %d/%d | T=%d k_top=%d tau=%.2f dim=%d depth=%d drop=%.2f lr=%.3g l2=%.3g clip=%.2f focal=%s gamma=%.2f | shrink=[%s]\n",
                             tidx, nrow(grid_df), T_stages, k_top, tau_cos, bdim, bdepth, bdrop, lr, l2, clipnorm, use_focal, focal_gamma, init_shrink_str))
    
    for (fold in seq_len(length(folds))) {
      # ---- SKIP if already done (resume) ----
      if (nrow(done)) {
        # tolerate column names as char; coerce for comparison
        already <- tryCatch(any(done$trial_id == tidx & done$fold == fold), error = function(e) FALSE)
        if (already) {
          if (verbose) cat(sprintf(" - skip trial %d fold %d (already in %s)\n", tidx, fold, out_csv))
          next
        }
      }
      
      val_idx <- sort(folds[[fold]]); if (!length(val_idx)) next
      train_idx <- sort(setdiff(seq_len(N), val_idx)); if (!length(train_idx)) next
      
      # TRAIN subgraph
      X_tr <- X[train_idx,, drop=FALSE]
      A_tr <- A[train_idx, train_idx, drop=FALSE]
      n_tr <- length(train_idx)
      gf_tr <- graph_feats_subgraph(A_tr); C_tr <- gf_tr$C; D_tr <- gf_tr$D
      M_tr <- matrix(1, n_tr, n_tr)
      y_tr_raw <- y_raw[train_idx]
      y_tr <- one_hot_batch(y_tr_raw, Kc, n_tr)
      inputs_tr <- build_inputs_subgraph(X_tr, A_tr, C_tr, D_tr, M_tr)
      
      alpha_counts <- tabulate(y_tr_raw + 1L, nbins = Kc)
      alpha_vec <- as.numeric((1 / pmax(alpha_counts, 1))); alpha_vec <- alpha_vec / mean(alpha_vec)
      
      k_clear_session()
      model_tr <- build_e2e_boost_noleak(
        n_nodes = n_tr, n_features = ncol(X), num_classes = Kc,
        T = T_stages, init_shrink = init_shrink,
        k_top = k_top, tau_cos = tau_cos,
        use_cluster_gate = use_cluster_gate,
        residual_depth_blend = residual_depth_blend,
        deep_supervision = TRUE,
        block_feature_dim = bdim,
        block_depth = bdepth,
        block_dropout = bdrop,
        use_focal = use_focal, focal_gamma = focal_gamma, focal_alpha = alpha_vec,
        lr = lr, l2 = l2, clipnorm = clipnorm
      )
      
      model_tr <- trainer_boost_on_all(
        model_tr, inputs_tr, y_tr,
        rounds = rounds, epochs_per_round = epochs_per_round,
        hard_thresh = hard_thresh, up_factor = up_factor, verbose = verbose
      )
      
      # VAL subgraph
      X_va <- X[val_idx,, drop=FALSE]
      A_va <- A[val_idx, val_idx, drop=FALSE]
      n_va <- length(val_idx)
      gf_va <- graph_feats_subgraph(A_va); C_va <- gf_va$C; D_va <- gf_va$D
      M_va <- matrix(1, n_va, n_va)
      y_va <- one_hot_batch(y_raw[val_idx], Kc, n_va)
      inputs_va <- build_inputs_subgraph(X_va, A_va, C_va, D_va, M_va)
      
      model_va <- build_e2e_boost_noleak(
        n_nodes = n_va, n_features = ncol(X), num_classes = Kc,
        T = T_stages, init_shrink = init_shrink,
        k_top = k_top, tau_cos = tau_cos,
        use_cluster_gate = use_cluster_gate,
        residual_depth_blend = residual_depth_blend,
        deep_supervision = TRUE,
        block_feature_dim = bdim,
        block_depth = bdepth,
        block_dropout = bdrop,
        use_focal = use_focal, focal_gamma = focal_gamma, focal_alpha = alpha_vec,
        lr = lr, l2 = l2, clipnorm = clipnorm
      )
      model_va$set_weights(model_tr$get_weights())
      
      p_va <- predict_probs_only(model_va, inputs_va)
      acc_va <- masked_acc(p_va, y_va)
      
      # write a single row right away (so resumes are exact)
      row_fixed <- data.frame(
        trial_id = tidx, fold = fold,
        N_train = n_tr, N_val = n_va,
        acc_val = sprintf("%.6f", acc_va),
        rounds = rounds, epochs_per_round = epochs_per_round,
        hard_thresh = hard_thresh, up_factor = up_factor, seed = seed,
        stringsAsFactors = FALSE, check.names = FALSE
      )
      row_conf <- as.data.frame(lapply(conf, function(x){
        if (is.list(x)) paste(x[[1]], collapse = ";") else as.character(x)
      }), stringsAsFactors = FALSE, check.names = FALSE)
      row_conf$init_shrink_used <- init_shrink_str
      
      out_row <- cbind(row_fixed, row_conf)
      write.table(out_row, out_csv, sep = ",", row.names = FALSE, col.names = FALSE, quote = TRUE, append = TRUE)
      
      if (verbose) cat(sprintf(" -> trial %d fold %d acc_val=%.4f (logged)\n", tidx, fold, acc_va))
    }
  }
  
  invisible(TRUE)
}
