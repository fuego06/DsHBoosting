
build_inputs <- function(X_features, A, C_mat, depth_vec, n_nodes, n_features) {
  list(
    array_reshape(X_features, c(1, n_nodes, n_features)),
    array_reshape(A, c(1, n_nodes, n_nodes)),
    array_reshape(C_mat, c(1, n_nodes, 1)),
    array_reshape(depth_vec, c(1, n_nodes, 1))
  )
}

one_hot_batch <- function(labels_raw, num_classes, n_nodes) {
  yNK <- keras::to_categorical(as.integer(labels_raw), num_classes = num_classes)
  array_reshape(yNK, c(1, n_nodes, num_classes))
}

softmax_rows <- function(M) { mx <- apply(M, 1, max); e <- exp(M - mx); e / rowSums(e) }
true_prob <- function(pNK, yNK) rowSums(pNK * yNK)
node_accuracy <- function(pNK, yNK) {
  pred <- max.col(pNK) - 1L; true <- max.col(yNK) - 1L; mean(pred == true)
}

make_stratified_80_20 <- function(labels_raw, val_frac_within_train = 0.10, seed = 7) {
  set.seed(seed)
  N <- length(labels_raw); cls <- sort(unique(labels_raw))
  n_train_target <- floor(0.80 * N)
  n_per_class <- table(labels_raw); prop_c <- as.numeric(n_per_class) / N
  
  train_c <- floor(prop_c * n_train_target)
  frac <- (prop_c * n_train_target) - train_c
  remainder <- n_train_target - sum(train_c)
  if (remainder > 0) {
    give <- order(frac, decreasing = TRUE)[seq_len(remainder)]
    train_c[give] <- train_c[give] + 1L
  }
  
  train_mask <- rep(FALSE, N); test_mask <- rep(FALSE, N)
  for (i in seq_along(cls)) {
    c <- cls[i]
    idx <- which(labels_raw == c); idx <- sample(idx)
    tr_take <- min(length(idx), train_c[i])
    tr <- idx[seq_len(tr_take)]; te <- setdiff(idx, tr)
    train_mask[tr] <- TRUE; test_mask[te] <- TRUE
  }
  
  val_mask <- rep(FALSE, N)
  for (c in cls) {
    tr_c <- which(train_mask & labels_raw == c)
    if (length(tr_c) >= 2L && val_frac_within_train > 0) {
      take <- max(1L, floor(val_frac_within_train * length(tr_c)))
      val_ids <- sample(tr_c, take)
      val_mask[val_ids] <- TRUE
    }
  }
  list(train = train_mask & !val_mask, val = val_mask, test = test_mask)
}

build_boosting_e2e_model <- function(
    n_nodes, n_features, num_classes,
    T = 4L,
    init_shrink = c(0.5, 0.4, 0.3, 0.2),
    k_top = 20L, tau_cos = 0.2,
    use_cluster_gate = TRUE,
    residual_depth_blend = TRUE, # TRUE: Z_proj + DendroELU(Z_proj, D)
    deep_supervision = TRUE, # aux heads at stages 1..T-1
    init_alpha_blend = 0.7,
    block_feature_dim = 64L,
    block_depth = 4L,
    block_dropout = 0.35,
    lr = 3e-3, l2 = 1e-8, clipnorm = 1.0
){
  if (length(init_shrink) == 1L) init_shrink <- rep(init_shrink, T)
  
  X_in <- layer_input(shape = c(n_nodes, n_features), name = "X_in")
  A_in <- layer_input(shape = c(n_nodes, n_nodes), name = "A_in")
  C_in <- layer_input(shape = c(n_nodes, 1), name = "C_in")
  D_in <- layer_input(shape = c(n_nodes, 1), name = "D_in")
  
  # Hierarchy gate once, reused every stage
  H_hat <- list(X_in, A_in, C_in, D_in) %>%
    layer_hierarchy_gate(hidden = 32L, use_edge_smooth = TRUE, lambda_smooth = 1e-3)
  
  # Running logits F ??? 0
  InitLogits <- layer_lambda(
    f = function(X){
      b <- tf$shape(X)[0L]; n <- tf$shape(X)[1L]
      k <- tf$constant(as.integer(num_classes), dtype = tf$int32)
      tf$zeros(tf$stack(list(b, n, k)), dtype = tf$float32)
    },
    name = "InitLogits"
  )
  F <- InitLogits(X_in)
  aux_outputs <- list()
  
  for (t in seq_len(T)) {
    # 1) Light stem
    Z0 <- X_in %>%
      layer_dense(units = as.integer(block_feature_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("z0_dense_t%d", t)) %>%
      layer_layer_normalization(name = sprintf("z0_ln_t%d", t))
    
    # 2) Cosine KNN + Blend(A,S)
    S_cos <- Z0 %>% layer_cosine_knn_topk(
      k_top = as.integer(k_top), tau = tau_cos, symmetrize = TRUE, name = sprintf("knn_t%d", t)
    )
    A_blend <- list(Z0, A_in, S_cos) %>%
      layer_blend_adj_alpha3(init_alpha = init_alpha_blend, name = sprintf("blend_t%d", t))
    
    # 3) Parent bias + previous prob embedding
    pb <- Z0 %>% layer_parent_bias(beta_init = 1.0)
    PrevProb <- F %>% layer_lambda(f = function(x) k_softmax(x), name = sprintf("PrevProb_t%d", t))
    prev_emb <- PrevProb %>%
      layer_dense(16, activation = "swish", kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("PrevEmb_t%d", t))
    
    # 4) Hierarchy-aware attention, then fuse (concat via Lambda)
    Z_attn <- list(Z0, A_blend, D_in, pb) %>%
      layer_hierattn_blend(out_dim = as.integer(block_feature_dim), depth_dim = 1L)
    Z_aug <- layer_safe_concat_last(list(Z_attn, prev_emb), name = sprintf("FusePrev_t%d", t))
    
    # 5) Optional cluster gate
    Z_gate <- if (use_cluster_gate) {
      Z_aug %>% layer_cluster_gate_learnable(
        num_clusters = as.integer(num_classes), dim = 80, beta = 10, gamma = 1.0,
        name = sprintf("ClusterGate_t%d", t)
      )
    } else Z_aug
    
    # 6) Project ??? DendroELU ??? (optional) residual-depth blend
    Z_proj <- Z_gate %>%
      layer_dense(units = as.integer(block_feature_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("Proj_t%d", t))
    Z_damp <- list(Z_proj, D_in) %>%
      layer_dendro_elu(alpha_init = 0.01, name = sprintf("Dendro_t%d", t))
    Z_stage <- if (residual_depth_blend) {
      layer_safe_add2(Z_proj, Z_damp, name = sprintf("DepthResidual_t%d", t))
    } else Z_damp
    
    # 7) Deep hierarchy block (gated by H_hat)
    Z_h <- list(Z_stage, H_hat) %>%
      layer_deep_hierarchy_block_v2(
        feature_dim = as.integer(block_feature_dim),
        depth = as.integer(block_depth),
        dropout_rate = block_dropout,
        alpha = 0.3,
        name = sprintf("DeepBlock_t%d", t)
      )
    
    # 8) ??logits ??? learnable shrink ??? accumulate
    delta_logits <- Z_h %>%
      layer_dense(units = as.integer(num_classes), activation = "linear",
                  kernel_initializer = initializer_zeros(),
                  bias_initializer = initializer_zeros(),
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("DeltaLogits_t%d", t))
    scaled <- delta_logits %>%
      layer_learnable_shrink(init_shrink = init_shrink[t], name = sprintf("ScaleDelta_t%d", t))
    F <- layer_lambda(f = function(tt) tt[[1]] + tt[[2]], name = sprintf("Accumulate_t%d", t))(list(F, scaled))
    
    # 9) Deep supervision (aux heads on earlier stages)
    if (deep_supervision && t < T) {
      Ft <- F %>% layer_lambda(f = function(x) x, name = sprintf("AuxLogits_t%d", t))
      aux_prob <- Ft %>% layer_activation(activation = "softmax", name = sprintf("AuxSoftmax_t%d", t))
      aux_outputs[[length(aux_outputs) + 1]] <- aux_prob
    }
  }
  
  prob <- F %>% layer_activation(activation = "softmax", name = "FinalSoftmax")
  outputs <- if (deep_supervision) c(list(prob), aux_outputs) else list(prob)
  model <- keras_model(inputs = list(X_in, A_in, C_in, D_in), outputs = outputs)
  
  if (deep_supervision) {
    loss_list <- c("categorical_crossentropy", rep("categorical_crossentropy", length(aux_outputs)))
    lw <- c(1.0, rep(0.3 / max(1, length(aux_outputs)), length(aux_outputs)))
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = lr, clipnorm = clipnorm),
      loss = loss_list, loss_weights = lw,
      metrics = list(FinalSoftmax = "accuracy"), # report metric only on final head
      sample_weight_mode = "temporal"
    )
  } else {
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = lr, clipnorm = clipnorm),
      loss = "categorical_crossentropy", metrics = "accuracy",
      sample_weight_mode = "temporal"
    )
  }
  model
}


build_e2e_boost_noleak <- function(
    n_nodes, n_features, num_classes,
    T = 3L,
    init_shrink = c(0.85,0.8,0.75),
    k_top = 8L, tau_cos = 0.2,
    use_cluster_gate = TRUE,
    residual_depth_blend = TRUE,
    deep_supervision = TRUE,
    init_alpha_blend = 0.7,
    block_feature_dim = 64L,
    block_depth = 4L,
    block_dropout = 0.1,
    # focal:
    use_focal = TRUE, focal_gamma = 2.0, focal_alpha = NULL,
    lr = 1e-3, l2 = 1e-8, clipnorm = 1.0
){
  if (length(init_shrink) == 1L) init_shrink <- rep(init_shrink, T)
  
  X_in <- layer_input(shape = c(n_nodes, n_features), name = "X_in")
  A_in <- layer_input(shape = c(n_nodes, n_nodes), name = "A_in")
  C_in <- layer_input(shape = c(n_nodes, 1), name = "C_in")
  D_in <- layer_input(shape = c(n_nodes, 1), name = "D_in")
  M_in <- layer_input(shape = c(n_nodes, n_nodes), name = "M_in") # NxN mask
  
  # Apply mask to A
  A_masked <- layer_lambda(f = function(tt) tt[[1]] * tt[[2]], name = "MaskA")(list(A_in, M_in))
  
  # Gate
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
    Z0 <- X_in %>%
      layer_dense(units = as.integer(block_feature_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("z0_dense_t%d", t)) %>%
      layer_layer_normalization(name = sprintf("z0_ln_t%d", t))
    
    # Cosine KNN with mask M, then blend with A_masked (both renormalized inside)
    S_cos <- list(Z0, M_in) %>% layer_cosine_knn_topk_masked(
      k_top = as.integer(k_top), tau = tau_cos, symmetrize = TRUE, name = sprintf("knn_t%d", t))
    A_blend <- list(Z0, A_masked, S_cos) %>% layer_blend_adj_alpha3(
      init_alpha = init_alpha_blend, name = sprintf("blend_t%d", t))
    
    pb <- Z0 %>% layer_parent_bias(beta_init = 1.0)
    PrevProb <- F %>% layer_lambda(f = function(x) k_softmax(x), name = sprintf("PrevProb_t%d", t))
    prev_emb <- PrevProb %>%
      layer_dense(16, activation = "swish", kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("PrevEmb_t%d", t))
    
    Z_attn <- list(Z0, A_blend, D_in, pb) %>%
      layer_hierattn_blend(out_dim = as.integer(block_feature_dim), depth_dim = 1L)
    Z_aug <- layer_safe_concat_last(list(Z_attn, prev_emb), name = sprintf("FusePrev_t%d", t))
    
    Z_gate <- if (use_cluster_gate) {
      Z_aug %>% layer_cluster_gate_learnable(
        num_clusters = as.integer(num_classes), dim = 80, beta = 10, gamma = 1.0,
        name = sprintf("ClusterGate_t%d", t))
    } else Z_aug
    
    Z_proj <- Z_gate %>%
      layer_dense(units = as.integer(block_feature_dim), activation = "swish",
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("Proj_t%d", t))
    Z_damp <- list(Z_proj, D_in) %>%
      layer_dendro_elu(alpha_init = 0.01, name = sprintf("Dendro_t%d", t))
    Z_stage <- if (residual_depth_blend) {
      layer_safe_add2(Z_proj, Z_damp, name = sprintf("DepthResidual_t%d", t))
    } else Z_damp
    
    Z_h <- list(Z_stage, H_hat) %>%
      layer_deep_hierarchy_block_v2(
        feature_dim = as.integer(block_feature_dim),
        depth = as.integer(block_depth),
        dropout_rate = block_dropout,
        alpha = 0.3,
        name = sprintf("DeepBlock_t%d", t))
    
    delta_logits <- Z_h %>%
      layer_dense(units = as.integer(num_classes), activation = "linear",
                  kernel_initializer = initializer_zeros(),
                  bias_initializer = initializer_zeros(),
                  kernel_regularizer = regularizer_l2(l2),
                  name = sprintf("DeltaLogits_t%d", t))
    scaled <- delta_logits %>% layer_learnable_shrink(
      init_shrink = init_shrink[t], name = sprintf("ScaleDelta_t%d", t))
    F <- layer_lambda(f = function(tt) tt[[1]] + tt[[2]],
                      name = sprintf("Accumulate_t%d", t))(list(F, scaled))
    
    if (deep_supervision && t < T) {
      Ft <- F %>% layer_lambda(f = function(x) x, name = sprintf("AuxLogits_t%d", t))
      aux_prob <- Ft %>% layer_activation(activation = "softmax",
                                          name = sprintf("AuxSoftmax_t%d", t))
      aux_outputs[[length(aux_outputs) + 1]] <- aux_prob
    }
  }
  
  prob <- F %>% layer_activation(activation = "softmax", name = "FinalSoftmax")
  outputs <- if (deep_supervision) c(list(prob), aux_outputs) else list(prob)
  model <- keras_model(inputs = list(X_in, A_in, C_in, D_in, M_in), outputs = outputs)
  
  # Compile (focal on all heads)
  n_out <- length(outputs)
  loss_list <- if (use_focal) {
    fl <- focal_loss(gamma = focal_gamma, alpha = focal_alpha)
    replicate(n_out, fl, simplify = FALSE)
  } else {
    if (deep_supervision) rep(list("categorical_crossentropy"), n_out) else "categorical_crossentropy"
  }
  if (deep_supervision) {
    lw <- c(1.0, rep(0.3 / max(1, n_out - 1L), n_out - 1L))
    metrics_list <- vector("list", n_out)
    names(metrics_list) <- lapply(outputs, function(o) o$name)
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
    inputs_trainval, # inputs built with M_trainval (train ??? val)
    y_batch, # shape [1, N, K]
    train_mask, # logical/0-1 length N
    val_mask = NULL, # logical/0-1 length N (optional)
    rounds = 5L,
    epochs_per_round = 50L,
    hard_thresh = 0.65,
    up_factor = 4.0,
    verbose = 1L
){
  N <- dim(y_batch)[2]
  n_out <- length(model$outputs)
  to_w <- function(v) array(as.numeric(v), dim = c(1, N, 1))
  
  # start uniform on TRAIN nodes only
  w <- rep(0, N); w[as.logical(train_mask)] <- 1
  w <- w / sum(w)
  
  for (r in seq_len(rounds)) {
    if (verbose) cat(sprintf("\n[e2e reweight] round %d/%d\n", r, rounds))
    
    sw_train <- rep(list(to_w(w)), n_out)
    sw_val <- if (is.null(val_mask)) {
      rep(list(array(0, dim = c(1, N, 1))), n_out)
    } else {
      rep(list(to_w(as.numeric(val_mask))), n_out)
    }
    
    model %>% fit(
      x = inputs_trainval,
      y = rep(list(y_batch), n_out),
      sample_weight = sw_train,
      validation_data = list(inputs_trainval, rep(list(y_batch), n_out), sw_val),
      batch_size = 1, epochs = epochs_per_round, shuffle = FALSE, verbose = verbose,
      callbacks = list(
        callback_early_stopping(monitor = "val_loss", patience = 15, restore_best_weights = TRUE)
      )
    )
    
    # predict on the same (train ??? val) masked graph
    p <- predict(model, inputs_trainval)
    if (is.list(p)) p <- p[[1]]
    if (length(dim(p)) == 2L) p <- array_reshape(p, c(1, nrow(p), ncol(p)))
    
    # ???hardness??? (1 - prob(true)) on train nodes
    true_idx <- max.col(y_batch[1,,]) # 1..K
    pt <- p[1,,][cbind(seq_len(N), true_idx)] # prob(true)
    hardness <- 1 - pt
    hard <- as.numeric(hardness >= hard_thresh)
    hard[!as.logical(train_mask)] <- 0
    
    # reweight: only hard train nodes get upweighted
    w <- w * (1 + (up_factor - 1) * hard)
    w[!as.logical(train_mask)] <- 0
    w <- w / sum(w)
    
    if (verbose) {
      frac_hard <- mean(hard[as.logical(train_mask)] > 0)
      cat(sprintf(" upweighted %.1f%% of train nodes (thresh=%.2f)\n", 100 * frac_hard, hard_thresh))
    }
  }
  
  model
}



