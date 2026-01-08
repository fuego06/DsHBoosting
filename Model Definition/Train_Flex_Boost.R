
k_sum <- keras::k_sum; k_log <- keras::k_log; k_clip <- keras::k_clip

ClusterGate <- keras::new_layer_class(
  classname = "ClusterGate",
  initialize = function(num_clusters,
                        dim = NULL, # NULL ??? use input dim; else project to this width
                        beta = 10.0, # softmax temperature
                        normalize_inputs = TRUE,
                        l2_reg = 0.0, # L2 regularization on gate params (optional)
                        ...) {
    super()$`__init__`(...)
    self$num_clusters <- as.integer(num_clusters)
    self$dim_target <- if (is.null(dim)) NULL else as.integer(dim)
    self$beta <- beta
    self$normalize_inputs <- normalize_inputs
    self$l2_reg <- l2_reg
  },
  build = function(input_shape) {
    # input: (B, N, D_in)
    D_in <- as.integer(input_shape[[3]])
    self$D_in <- D_in
    self$D_eff <- if (is.null(self$dim_target)) D_in else self$dim_target
    
    # Optional projection from D_in -> D_eff when requested
    if (!is.null(self$dim_target) && self$dim_target != D_in) {
      self$P <- self$add_weight(
        name = "proj", shape = shape(D_in, self$D_eff),
        initializer = "glorot_uniform",
        regularizer = if (self$l2_reg > 0) regularizer_l2(self$l2_reg) else NULL,
        trainable = TRUE
      )
    } else {
      self$P <- NULL
    }
    
    # Cluster prototypes U: (D_eff, K)
    self$U <- self$add_weight(
      name = "prototypes", shape = shape(self$D_eff, self$num_clusters),
      initializer = "glorot_uniform",
      regularizer = if (self$l2_reg > 0) regularizer_l2(self$l2_reg) else NULL,
      trainable = TRUE
    )
    
    # Map cluster probs (K) -> per-channel gate (D_eff)
    self$Gw <- self$add_weight(
      name = "gate_map", shape = shape(self$num_clusters, self$D_eff),
      initializer = "glorot_uniform",
      regularizer = if (self$l2_reg > 0) regularizer_l2(self$l2_reg) else NULL,
      trainable = TRUE
    )
    self$gb <- self$add_weight(
      name = "gate_bias", shape = shape(self$D_eff),
      initializer = "zeros",
      trainable = TRUE
    )
  },
  call = function(inputs, mask = NULL) {
    Z <- inputs # (B, N, D_in)
    
    # Optional projection to D_eff
    if (!is.null(self$P)) {
      Z <- tf$matmul(Z, self$P) # (B, N, D_eff)
    }
    # Normalize node features if requested
    if (self$normalize_inputs) {
      Zn <- tf$linalg$l2_normalize(Z, axis = as.integer(-1L))
    } else {
      Zn <- Z
    }
    
    # Normalize prototypes column-wise (per cluster)
    U_n <- tf$linalg$l2_normalize(self$U, axis = as.integer(0L)) # (D_eff, K)
    
    # Soft cluster assignment: (B, N, K)
    logits <- self$beta * tf$matmul(Zn, U_n) # (B, N, K)
    Pk <- tf$nn$softmax(logits, axis = as.integer(-1L))
    
    # Channel gate from cluster probs: (B, N, D_eff)
    gate <- tf$matmul(Pk, self$Gw) + self$gb
    gate <- tf$nn$sigmoid(gate)
    
    # Apply gate (elementwise)
    Z * gate
  }
)

layer_cluster_gate_flexible <- function(object,
                                        num_clusters,
                                        dim = NULL, # NULL = use input width; else any integer
                                        beta = 10.0,
                                        normalize_inputs = TRUE,
                                        l2_reg = 0.0,
                                        ...) {
  keras::create_layer(
    ClusterGate, object,
    list(num_clusters = as.integer(num_clusters),
         dim = dim,
         beta = beta,
         normalize_inputs = normalize_inputs,
         l2_reg = l2_reg,
         ...)
  )
}


# ======================================================================
# 2) Helper: fused width after concatenating [Z_attn, prev_emb]
# ======================================================================
fused_width <- function(block_feature_dim, prev_emb_dim) {
  as.integer(block_feature_dim + prev_emb_dim)
}


lambda2 <- function(f, name = NULL) keras::layer_lambda(f = f, name = name)

layer_mul2 <- function(t1, t2, name = NULL) {
  lambda2(function(tt) tt[[1]] * tt[[2]], name = name)(list(t1, t2))
}
layer_add2 <- function(t1, t2, name = NULL) {
  lambda2(function(tt) tt[[1]] + tt[[2]], name = name)(list(t1, t2))
}
layer_concat_last <- function(tensors, name = NULL) {
  lambda2(function(tt) tf$concat(tt, axis = as.integer(-1L)), name = name)(tensors)
}

focal_loss <- function(gamma = 2.0, alpha = NULL, eps = 1e-7) {
  function(y_true, y_pred) {
    y_pred <- k_clip(y_pred, eps, 1 - eps)
    ce <- -k_sum(y_true * k_log(y_pred), axis = -1L) # [B,N]
    pt <- k_sum(y_true * y_pred, axis = -1L) # [B,N]
    fl <- (1 - pt)^gamma * ce
    if (!is.null(alpha)) {
      a <- if (length(alpha) == 1L) k_constant(alpha)
      else k_sum(y_true * k_constant(alpha), axis = -1L)
      fl <- a * fl
    }
    k_mean(fl)
  }
}





layer_cluster_gate_learnable <- function(
    object,
    num_clusters,
    dim = NULL,
    beta = 10,
    normalize_inputs = TRUE,
    l2_reg = 0,
    name = NULL,
    ...
){
  keras::create_layer(
    ClusterGate, object,
    list(
      num_clusters = as.integer(num_clusters),
      dim = if (is.null(dim)) NULL else as.integer(dim),
      beta = beta,
      normalize_inputs = normalize_inputs,
      l2_reg = l2_reg,
      name = name
    )
  )
}




build_e2e_boost_flexible <- function(
    n_nodes, n_features, num_classes,
    T = 3L,
    init_shrink = c(0.85, 0.80, 0.75),
    # feature sizes
    block_feature_dim = 32L,
    prev_emb_dim = 8L,
    gate_dim_target = NULL, # if NULL -> block_feature_dim + prev_emb_dim
    # graph mix
    k_top = 8L, tau_cos = 0.5, init_alpha_blend = 0.7,
    # block depth / regularization
    block_depth = 3L, block_dropout = 0.4,
    # loss / opt
    use_focal = TRUE, focal_gamma = 1.6, focal_alpha = NULL,
    lr = 1e-3, l2 = 1e-4, clipnorm = 0.7,
    # gate
    use_cluster_gate = TRUE, gate_beta = 10, gate_normalize_inputs = TRUE, gate_l2 = 0
){
  if (length(init_shrink) == 1L) init_shrink <- rep(init_shrink, T)
  stopifnot(length(init_shrink) == T)
  
  if (is.null(gate_dim_target)) gate_dim_target <- as.integer(block_feature_dim + prev_emb_dim)
  gate_dim_target <- as.integer(gate_dim_target)
  
  # ---- Inputs
  X_in <- layer_input(shape = c(n_nodes, n_features), name = "X_in")
  A_in <- layer_input(shape = c(n_nodes, n_nodes), name = "A_in")
  C_in <- layer_input(shape = c(n_nodes, 1), name = "C_in")
  D_in <- layer_input(shape = c(n_nodes, 1), name = "D_in")
  M_in <- layer_input(shape = c(n_nodes, n_nodes), name = "M_in")
  
  # Mask A by M
  A_masked <- layer_lambda(f = function(tt) tt[[1]] * tt[[2]], name = "MaskA")(list(A_in, M_in))
  
  # Hierarchy gate
  H_hat <- list(X_in, A_masked, C_in, D_in) %>% 
    layer_hierarchy_gate(hidden = 32L, use_edge_smooth = TRUE, lambda_smooth = 1e-3)
  
  # Init logits
  InitLogits <- layer_lambda(
    f = function(X){
      b <- tf$shape(X)[0L]; n <- tf$shape(X)[1L]
      k <- tf$constant(as.integer(num_classes), dtype = tf$int32)
      tf$zeros(tf$stack(list(b, n, k)), dtype = tf$float32)
    }, name = "InitLogits"
  )
  F <- InitLogits(X_in)
  
  aux_outputs <- list()
  
  for (t in seq_len(T)) {
    # Encode features
    Z0 <- X_in %>%
      layer_dense(units = as.integer(block_feature_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2), name = sprintf("z0_dense_t%d", t)) %>%
      layer_layer_normalization(name = sprintf("z0_ln_t%d", t))
    
    # KNN (masked) + blend with A
    S_cos <- list(Z0, M_in) %>%
      layer_cosine_knn_topk_masked(k_top = as.integer(k_top), tau = tau_cos, symmetrize = TRUE,
                                   name = sprintf("knn_t%d", t))
    A_blend <- list(Z0, A_masked, S_cos) %>%
      layer_blend_adj_alpha3(init_alpha = init_alpha_blend, name = sprintf("blend_t%d", t))
    
    # Parent bias + previous prob embedding
    pb <- Z0 %>% layer_parent_bias(beta_init = 1.0)
    PrevProb <- F %>% layer_lambda(f = function(x) keras::k_softmax(x), name = sprintf("PrevProb_t%d", t))
    prev_emb <- PrevProb %>%
      layer_dense(as.integer(prev_emb_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2), name = sprintf("PrevEmb_t%d", t))
    
    # Attention over blended A
    Z_attn <- list(Z0, A_blend, D_in, pb) %>%
      layer_hierattn_blend(out_dim = as.integer(block_feature_dim), depth_dim = 1L)
    
    # Fuse with prev_emb
    Z_aug <- layer_lambda(
      f = function(tt) tf$concat(tt, axis = as.integer(-1L)),
      name = sprintf("FusePrev_t%d", t)
    )(list(Z_attn, prev_emb)) # shape (B,N, block_feature_dim + prev_emb_dim)
    
    # Pre-project to known gate width -> makes ClusterGate shape-robust
    Z_gate_in <- Z_aug %>%
      layer_dense(units = as.integer(gate_dim_target), activation = "linear",
                  kernel_regularizer = regularizer_l2(l2), name = sprintf("GatePrep_t%d", t))
    
    # Optional ClusterGate (now with correct kwargs only)
    Z_gate <- if (isTRUE(use_cluster_gate)) {
      Z_gate_in %>% layer_cluster_gate_learnable(
        num_clusters = as.integer(num_classes),
        dim = as.integer(gate_dim_target),
        beta = gate_beta,
        normalize_inputs = gate_normalize_inputs,
        l2_reg = gate_l2,
        name = sprintf("ClusterGate_t%d", t)
      )
    } else Z_gate_in
    
    # Project to block_feature_dim, damp with depth, residual
    Z_proj <- Z_gate %>%
      layer_dense(units = as.integer(block_feature_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2), name = sprintf("Proj_t%d", t))
    Z_damp <- list(Z_proj, D_in) %>% layer_dendro_elu(alpha_init = 0.01, name = sprintf("Dendro_t%d", t))
    Z_stage <- layer_lambda(f = function(tt) tt[[1]] + tt[[2]], name = sprintf("DepthResidual_t%d", t))(list(Z_proj, Z_damp))
    
    # Deep hierarchy residual block
    Z_h <- list(Z_stage, H_hat) %>%
      layer_deep_hierarchy_block_v2(feature_dim = as.integer(block_feature_dim),
                                    depth = as.integer(block_depth),
                                    dropout_rate = block_dropout,
                                    alpha = 0.3,
                                    name = sprintf("DeepBlock_t%d", t))
    
    # Stage update to logits
    delta_logits <- Z_h %>%
      layer_dense(units = as.integer(num_classes), activation = "linear",
                  kernel_initializer = initializer_zeros(),
                  bias_initializer = initializer_zeros(),
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("DeltaLogits_t%d", t))
    scaled <- delta_logits %>%
      layer_learnable_shrink(init_shrink = init_shrink[t], name = sprintf("ScaleDelta_t%d", t))
    F <- layer_lambda(f = function(tt) tt[[1]] + tt[[2]], name = sprintf("Accumulate_t%d", t))(list(F, scaled))
    
    # Optional deep-supervision head
    if (t < T) {
      Ft <- layer_lambda(f = function(x) x, name = sprintf("AuxLogits_t%d", t))(F)
      aux_outputs[[length(aux_outputs) + 1]] <- layer_activation(Ft, activation = "softmax",
                                                                 name = sprintf("AuxSoftmax_t%d", t))
    }
  }
  
  prob <- layer_activation(F, activation = "softmax", name = "FinalSoftmax")
  outputs <- if (length(aux_outputs)) c(list(prob), aux_outputs) else list(prob)
  
  model <- keras_model(inputs = list(X_in, A_in, C_in, D_in, M_in), outputs = outputs)
  
  # Compile
  n_out <- length(outputs)
  loss_list <- if (use_focal) {
    fl <- focal_loss(gamma = focal_gamma, alpha = focal_alpha)
    replicate(n_out, fl, simplify = FALSE)
  } else {
    rep(list("categorical_crossentropy"), n_out)
  }
  
  if (length(aux_outputs)) {
    lw <- c(1.0, rep(0.3 / max(1, n_out - 1L), n_out - 1L))
    metrics_list <- vector("list", n_out); names(metrics_list) <- vapply(outputs, function(o) o$name, "")
    metrics_list[[1]] <- "accuracy"
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = lr, clipnorm = clipnorm),
      loss = loss_list, loss_weights = lw, metrics = metrics_list,
      sample_weight_mode = "temporal"
    )
  } else {
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = lr, clipnorm = clipnorm),
      loss = loss_list, metrics = "accuracy",
      sample_weight_mode = "temporal"
    )
  }
  
  model
}

train_e2e_boost <- function(
    model,
    inputs_trainval,
    y_batch,
    train_mask,
    val_mask = NULL,
    rounds = 5L,
    epochs_per_round = 80L,
    hard_thresh = 0.60,
    up_factor = 3.0,
    verbose = 1L,
    es_patience = 12L,
    es_min_delta = 0.001,
    rlrop = TRUE,
    rlrop_factor = 0.5,
    rlrop_patience = 6L,
    rlrop_min_lr = 1e-5
){
  N <- dim(y_batch)[2]
  n_out <- length(model$outputs)
  to_w <- function(v) array(as.numeric(v), dim = c(1, N, 1))
  
  w <- rep(0, N); w[as.logical(train_mask)] <- 1; w <- w / sum(w)
  
  for (r in seq_len(rounds)) {
    if (verbose) cat(sprintf("\n[e2e reweight] round %d/%d\n", r, rounds))
    
    sw_train <- rep(list(to_w(w)), n_out)
    sw_val <- if (is.null(val_mask)) rep(list(array(0, dim = c(1, N, 1))), n_out)
    else rep(list(to_w(as.numeric(val_mask))), n_out)
    
    cbs <- list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = as.integer(es_patience),
        min_delta = as.numeric(es_min_delta),
        restore_best_weights = TRUE
      )
    )
    if (isTRUE(rlrop)) {
      cbs <- c(cbs, list(
        callback_reduce_lr_on_plateau(
          monitor = "val_loss",
          factor = as.numeric(rlrop_factor),
          patience = as.integer(rlrop_patience),
          min_lr = as.numeric(rlrop_min_lr),
          verbose = as.integer(verbose > 0)
        )
      ))
    }
    
    model %>% fit(
      x = inputs_trainval,
      y = rep(list(y_batch), n_out),
      sample_weight = sw_train,
      validation_data = list(inputs_trainval, rep(list(y_batch), n_out), sw_val),
      batch_size = 1, epochs = epochs_per_round, shuffle = FALSE, verbose = verbose,
      callbacks = cbs
    )
    
    # ??? your reweighting code (unchanged) ???
    p <- predict(model, inputs_trainval); if (is.list(p)) p <- p[[1]]
    if (length(dim(p)) == 2L) p <- array_reshape(p, c(1, nrow(p), ncol(p)))
    true_idx <- max.col(y_batch[1,,])
    pt <- p[1,,][cbind(seq_len(N), true_idx)]
    hardness <- 1 - pt
    hard <- as.numeric(hardness >= hard_thresh)
    hard[!as.logical(train_mask)] <- 0
    w <- w * (1 + (up_factor - 1) * hard); w[!as.logical(train_mask)] <- 0; w <- w / sum(w)
    if (verbose) cat(sprintf(" upweighted %.1f%% of train nodes\n", 100*mean(hard[as.logical(train_mask)]>0)))
  }
  model
}



library(keras)
library(dplyr)
library(tidyr)
library(ggplot2)

train_e2e_boost_with_history <- function(
    model,
    inputs_trainval,
    y_batch,
    train_mask,
    val_mask = NULL,
    rounds = 5L,
    epochs_per_round = 80L,
    hard_thresh = 0.60,
    up_factor = 3.0,
    verbose = 1L,
    es_patience = 12L,
    es_min_delta = 0.001,
    rlrop = TRUE,
    rlrop_factor = 0.5,
    rlrop_patience = 6L,
    rlrop_min_lr = 1e-5
){
  N <- dim(y_batch)[2]
  n_out <- length(model$outputs)
  to_w <- function(v) array(as.numeric(v), dim = c(1, N, 1))
  
  w <- rep(0, N); w[as.logical(train_mask)] <- 1; w <- w / sum(w)
  
  all_hist <- NULL
  
  for (r in seq_len(rounds)) {
    if (verbose) cat(sprintf("\n[e2e reweight] round %d/%d\n", r, rounds))
    
    sw_train <- rep(list(to_w(w)), n_out)
    sw_val <- if (is.null(val_mask)) rep(list(array(0, dim = c(1, N, 1))), n_out)
    else rep(list(to_w(as.numeric(val_mask))), n_out)
    
    cbs <- list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = as.integer(es_patience),
        min_delta = as.numeric(es_min_delta),
        restore_best_weights = TRUE
      )
    )
    if (isTRUE(rlrop)) {
      cbs <- c(cbs, list(
        callback_reduce_lr_on_plateau(
          monitor = "val_loss",
          factor = as.numeric(rlrop_factor),
          patience = as.integer(rlrop_patience),
          min_lr = as.numeric(rlrop_min_lr),
          verbose = as.integer(verbose > 0)
        )
      ))
    }
    
    # ---- train one boosting round ----
    hist <- model %>% fit(
      x = inputs_trainval,
      y = rep(list(y_batch), n_out),
      sample_weight = sw_train,
      validation_data = list(inputs_trainval, rep(list(y_batch), n_out), sw_val),
      batch_size = 1,
      epochs = epochs_per_round,
      shuffle = FALSE,
      verbose = verbose,
      callbacks = cbs
    )
    
    # ---- collect history ----
    hdf <- as.data.frame(hist)
    hdf$round <- r
    hdf$epoch_in_round <- seq_len(nrow(hdf))
    if (is.null(all_hist)) all_hist <- hdf else all_hist <- bind_rows(all_hist, hdf)
    
    # ---- reweighting (unchanged) ----
    p <- predict(model, inputs_trainval)
    if (is.list(p)) p <- p[[1]]
    if (length(dim(p)) == 2L) p <- array_reshape(p, c(1, nrow(p), ncol(p)))
    true_idx <- max.col(y_batch[1,,])
    pt <- p[1,,][cbind(seq_len(N), true_idx)]
    hardness <- 1 - pt
    hard <- as.numeric(hardness >= hard_thresh)
    hard[!as.logical(train_mask)] <- 0
    w <- w * (1 + (up_factor - 1) * hard); w[!as.logical(train_mask)] <- 0; w <- w / sum(w)
    if (verbose) cat(sprintf(" upweighted %.1f%% of train nodes\n", 
                             100 * mean(hard[as.logical(train_mask)] > 0)))
  }
  
  list(
    model = model,
    history = all_hist
  )
}




