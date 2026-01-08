tf <- import("tensorflow", delay_load = TRUE)
tf$random$set_seed(42); set.seed(42)

one_hot_batch <- function(labels_raw, num_classes, n_nodes) {
  yNK <- keras::to_categorical(as.integer(labels_raw), num_classes = num_classes)
  array_reshape(yNK, c(1, n_nodes, num_classes))
}
node_accuracy <- function(pNK, yNK) {
  pred <- max.col(pNK) - 1L
  true <- max.col(yNK) - 1L
  mean(pred == true)
}
build_inputs <- function(X_features, A, C_mat, depth_vec, n_nodes, n_features) {
  list(
    array_reshape(X_features, c(1, n_nodes, n_features)),
    array_reshape(A, c(1, n_nodes, n_nodes)),
    array_reshape(C_mat, c(1, n_nodes, 1)),
    array_reshape(depth_vec, c(1, n_nodes, 1))
  )
}


library(keras)
library(reticulate)
tf <- import("tensorflow", delay_load = TRUE)
tf$random$set_seed(42); set.seed(42)

one_hot_batch <- function(labels_raw, num_classes, n_nodes) {
  yNK <- keras::to_categorical(as.integer(labels_raw), num_classes = num_classes)
  array_reshape(yNK, c(1, n_nodes, num_classes))
}
node_accuracy <- function(pNK, yNK) {
  pred <- max.col(pNK) - 1L
  true <- max.col(yNK) - 1L
  mean(pred == true)
}
build_inputs <- function(X_features, A, C_mat, depth_vec, n_nodes, n_features) {
  list(
    array_reshape(X_features, c(1, n_nodes, n_features)),
    array_reshape(A, c(1, n_nodes, n_nodes)),
    array_reshape(C_mat, c(1, n_nodes, 1)),
    array_reshape(depth_vec, c(1, n_nodes, 1))
  )
}

# ============================================================
# 1) Minimal helper layers
# ============================================================
AdjRowNorm <- keras::new_layer_class(
  classname = "AdjRowNorm",
  call = function(A, mask=NULL) A / (k_sum(A, axis = -1L, keepdims = TRUE) + 1e-8)
)
layer_adj_row_norm <- function(object=NULL) keras::create_layer(AdjRowNorm, object, list())

layer_cosine_knn_topk <- keras::new_layer_class(
  classname = "CosineKNNTopK",
  initialize = function(k_top = 20L, tau = 0.2, symmetrize = TRUE, ...) {
    super()$`__init__`(...); self$k_top <- as.integer(k_top); self$tau <- tau; self$symm <- symmetrize
  },
  call = function(inputs, mask = NULL){
    Z <- tf$linalg$l2_normalize(inputs, axis = 2L) # (B,N,D)
    S <- tf$matmul(Z, Z, transpose_b = TRUE) # (B,N,N)
    bs <- tf$shape(S)[0L]; n <- tf$shape(S)[1L]
    very_neg <- tf$constant(-1e9, tf$float32)
    I <- tf$eye(n, batch_shape = list(bs))
    S_no_self <- S + I * very_neg
    tk <- tf$nn$top_k(S_no_self, k = self$k_top)
    thr <- tf$expand_dims(tk$values[,, self$k_top - 1L], 2L)
    mask_topk <- tf$cast(S_no_self >= thr, tf$float32)
    logits <- (S / self$tau) + very_neg * (1 - mask_topk)
    P <- tf$nn$softmax(logits, axis = -1L)
    if (self$symm) {
      Pt <- tf$transpose(P, perm = c(0L, 2L, 1L))
      P <- 0.5 * (P + Pt)
      P <- P / (k_sum(P, axis = -1L, keepdims = TRUE) + 1e-8)
    }
    P
  }
)

BlendAdjAlpha3 <- keras::new_layer_class(
  classname = "BlendAdjAlpha3",
  initialize = function(init_alpha = 0.7, symmetrize = TRUE, row_normalize = TRUE, ...) {
    super()$`__init__`(...); self$init_alpha <- init_alpha; self$symmetrize <- symmetrize; self$row_normalize <- row_normalize
  },
  build = function(input_shape) {
    self$alpha_logit <- self$add_weight(
      name = "alpha_logit", shape = shape(),
      initializer = initializer_constant(stats::qlogis(self$init_alpha)),
      trainable = TRUE
    )
  },
  call = function(inputs, mask=NULL) {
    A <- inputs[[2]]; S <- inputs[[3]]
    if (self$row_normalize) {
      A <- A / (k_sum(A, axis = -1L, keepdims = TRUE) + 1e-8)
      S <- S / (k_sum(S, axis = -1L, keepdims = TRUE) + 1e-8)
    }
    alpha <- tf$nn$sigmoid(self$alpha_logit)
    B <- alpha * A + (1 - alpha) * S
    if (self$symmetrize) {
      Bt <- tf$transpose(B, perm = c(0L,2L,1L))
      B <- 0.5 * (B + Bt)
      B <- B / (k_sum(B, axis = -1L, keepdims = TRUE) + 1e-8)
    }
    B
  }
)
layer_blend_adj_alpha3 <- function(object, init_alpha = 0.7, symmetrize = TRUE, row_normalize = TRUE, ...) {
  keras::create_layer(BlendAdjAlpha3, object, list(init_alpha = init_alpha, symmetrize = symmetrize, row_normalize = row_normalize, ...))
}

ParentBias <- keras::new_layer_class(
  classname = "ParentBias",
  initialize = function(beta_init = 1.0, ...){ super()$`__init__`(...); self$beta_init <- beta_init },
  build = function(input_shape){
    self$beta <- self$add_weight("beta", shape = shape(),
                                 initializer = initializer_constant(self$beta_init), trainable = TRUE)
  },
  call = function(Z, mask=NULL){
    s <- tf$reduce_mean(tf$square(Z), axis = as.integer(-1L), keepdims = TRUE)
    tf$nn$sigmoid(self$beta * s) # (B,N,1)
  }
)
layer_parent_bias <- function(object, beta_init = 1.0, ...) {
  keras::create_layer(ParentBias, object, list(beta_init = beta_init, ...))
}

DendroELU <- keras::new_layer_class(
  classname = "DendroELU",
  initialize = function(alpha_init = 1e-2, ...){ super()$`__init__`(...); self$alpha_init <- alpha_init },
  build = function(input_shape){
    self$alpha <- self$add_weight("alpha", shape = shape(),
                                  initializer = initializer_constant(self$alpha_init), trainable = TRUE)
  },
  call = function(inputs, mask=NULL){
    Z <- inputs[[1]]; D <- inputs[[2]]
    tf$nn$elu(Z) * tf$exp(- self$alpha * D)
  }
)
layer_dendro_elu <- function(object, alpha_init = 1e-2, ...) {
  keras::create_layer(DendroELU, object, list(alpha_init = alpha_init, ...))
}

layer_hierarchy_gate <- keras::new_layer_class(
  classname = "HierarchyGate",
  initialize = function(hidden = 32L, use_edge_smooth = TRUE, lambda_smooth = 1e-3){
    super()$`__init__`()
    self$hidden <- as.integer(hidden); self$use_edge_smooth <- use_edge_smooth; self$lambda_smooth <- lambda_smooth
  },
  build = function(input_shape){
    self$d1 <- layer_dense(units = self$hidden, activation = "elu")
    self$bn <- layer_batch_normalization()
    self$d2 <- layer_dense(units = 1L, activation = "sigmoid")
    self$normA <- layer_adj_row_norm()
  },
  call = function(inputs, mask=NULL){
    X <- inputs[[1]]; A <- inputs[[2]]; C <- inputs[[3]]; D <- inputs[[4]]
    H <- k_concatenate(list(X, C, D), axis = -1L) %>% self$d1() %>% self$bn() %>% self$d2()
    if (self$use_edge_smooth) {
      A_norm <- self$normA(A)
      Hn <- tf$matmul(A_norm, H)
      smooth <- k_mean(k_square(H - Hn))
      self$add_loss(self$lambda_smooth * smooth)
      H <- 0.5 * (H + Hn)
    }
    H # (B,N,1)
  }
)

layer_hierattn_blend <- keras::new_layer_class(
  classname = "HierAttnBlend",
  initialize = function(out_dim, depth_dim = 1L){ super()$`__init__`(); self$out_dim <- as.integer(out_dim) },
  build = function(input_shape){
    fdim <- as.integer(input_shape[[1]][[3]])
    self$W <- self$add_weight("W", shape = shape(fdim, self$out_dim), initializer = "glorot_uniform", trainable = TRUE)
    self$q <- self$add_weight("q", shape = shape(self$out_dim, 1L), initializer = "glorot_uniform", trainable = TRUE)
    self$k <- self$add_weight("k", shape = shape(self$out_dim, 1L), initializer = "glorot_uniform", trainable = TRUE)
    self$wd <- self$add_weight("wd", shape = shape(1L), initializer = "zeros", trainable = TRUE)
  },
  call = function(inputs, mask = NULL){
    Z <- inputs[[1]]; A_ <- inputs[[2]]; D <- inputs[[3]]; pb <- inputs[[4]]
    Wh <- tf$matmul(Z, self$W) # (B,N,F')
    qi <- tf$squeeze(tf$matmul(Wh, self$q), -1L) # (B,N)
    kj <- tf$squeeze(tf$matmul(Wh, self$k), -1L) # (B,N)
    dD <- tf$abs(tf$expand_dims(D, 2L) - tf$expand_dims(D, 1L)); dD <- tf$squeeze(dD, -1L)
    e <- tf$expand_dims(qi, 2L) + tf$expand_dims(kj, 1L) + self$wd * dD
    e <- tf$nn$leaky_relu(e, alpha = 0.2)
    logA <- tf$math$log(1e-6 + A_)
    pb_j <- tf$transpose(pb, perm = c(0L, 2L, 1L))
    logits <- e + logA + pb_j
    alpha <- tf$nn$softmax(logits, axis = -1L) # (B,N,N)
    tf$einsum("bij,bjf->bif", alpha, Wh) # (B,N,F')
  }
)

DeepHierarchyBlockV2 <- keras::new_layer_class(
  classname = "DeepHierarchyBlockV2",
  initialize = function(feature_dim, depth = 6L, dropout_rate = 0.2, alpha = 0.3, ...){
    super()$`__init__`(...); self$feature_dim <- as.integer(feature_dim)
    self$depth <- as.integer(depth); self$dropout_rate <- dropout_rate; self$alpha <- alpha
  },
  build = function(input_shape){
    d <- self$feature_dim
    self$layers <- vector("list", self$depth)
    for(i in seq_len(self$depth)){
      self$layers[[i]] <- list(
        dense = layer_dense(units = d, activation = "linear"),
        norm = layer_layer_normalization(),
        drop = layer_dropout(rate = self$dropout_rate)
      )
    }
  },
  call = function(inputs, mask=NULL){
    Z <- inputs[[1]]; H <- inputs[[2]] # (B,N,D), (B,N,1)
    for(i in seq_len(self$depth)){
      Z_prev <- Z
      Zt <- self$layers[[i]]$dense(Z)
      Zt <- H * k_elu(Zt) + (1 - H) * k_elu(self$alpha * Zt)
      Zt <- self$layers[[i]]$norm(Zt)
      Zt <- self$layers[[i]]$drop(Zt)
      Z <- Z_prev + Zt
    }
    Z
  }
)
layer_deep_hierarchy_block_v2 <- function(object, feature_dim, depth = 6L, dropout_rate = 0.2, alpha = 0.3, ...) {
  keras::create_layer(DeepHierarchyBlockV2, object, list(feature_dim = as.integer(feature_dim), depth = as.integer(depth), dropout_rate = dropout_rate, alpha = alpha, ...))
}

# Optional Cluster Gate
ClusterGate <- keras::new_layer_class(
  classname = "ClusterGate",
  initialize = function(num_clusters, dim, beta = 10.0, gamma = 1.0, ...) {
    super()$`__init__`(...); self$num_clusters <- as.integer(num_clusters); self$dim <- as.integer(dim)
    self$beta <- beta; self$gamma <- gamma
  },
  build = function(input_shape){
    self$C <- self$add_weight("prototypes", shape = shape(self$num_clusters, self$dim),
                              initializer = "glorot_uniform", trainable = TRUE)
  },
  call = function(inputs, mask = NULL){
    Z <- inputs
    ZN <- tf$linalg$l2_normalize(Z, axis = -1L)
    CN <- tf$linalg$l2_normalize(self$C, axis = -1L)
    sim <- tf$matmul(ZN, tf$transpose(CN)) # (B,N,K)
    gate <- tf$nn$sigmoid(self$beta * tf$reduce_max(sim, axis = -1L, keepdims = TRUE) - self$gamma)
    Z * gate
  },
  compute_output_shape = function(input_shape) input_shape
)
layer_cluster_gate_learnable <- function(object, num_clusters, dim, beta = 10.0, gamma = 1.0, ...) {
  keras::create_layer(ClusterGate, object, list(num_clusters = as.integer(num_clusters), dim = as.integer(dim), beta = beta, gamma = gamma, ...))
}

mask_list_from_vec <- function(model, mask_vec, n_nodes) {
  # numeric 0/1, shape (1, n_nodes)
  m <- array(as.numeric(mask_vec), dim = c(1, n_nodes))
  # replicate per-output in the same order as as_y_list()
  outs <- model$outputs
  rep(list(m), length(outs))
}


layer_cluster_gate_learnable2 <- function(object, k_top = 8L, name = NULL) {
  keras::create_layer(
    keras::new_layer_class(
      classname = "ClusterGate",
      initialize = function(self, k_top) {
        self$k_top <- as.integer(k_top)
      },
      build = function(self, input_shape) {
        # input_shape: (batch, N, C)
        ch <- as.integer(input_shape[[3]])
        # Centroids shaped (C, k_top) so we can do: (B,N,C) @ (C,k) -> (B,N,k)
        self$C <- self$add_weight(
          name = "centroids",
          shape = c(ch, self$k_top),
          initializer = "glorot_uniform",
          trainable = TRUE
        )
      },
      call = function(self, inputs, mask = NULL) {
        Z <- inputs # (B,N,C)
        ZN <- tf$linalg$l2_normalize(Z, axis = -1L)
        CN <- tf$linalg$l2_normalize(self$C, axis = 0L) # (C,k)
        S <- tf$matmul(ZN, CN) # (B,N,k_top) cosine-style scores
        S
      },
      compute_output_shape = function(self, input_shape) {
        list(NULL, as.integer(input_shape[[2]]), self$k_top)
      }
    ),
    object,
    k_top = as.integer(k_top),
    name = name
  )
}


layer_safe_add2 <- function(t1, t2, name = NULL) {
  layer_lambda(f = function(tt) tt[[1]] + tt[[2]], name = name)(list(t1, t2))
}
layer_safe_concat_last <- function(tensors, name = NULL) {
  layer_lambda(f = function(tt) tf$concat(tt, axis = as.integer(-1L)), name = name)(tensors)
}

# Learnable shrink: sigmoid(w) * z, with w initialized at qlogis(init_shrink)
LearnableShrink <- keras::new_layer_class(
  classname = "LearnableShrink",
  initialize = function(init_shrink = 0.5, ...) {
    super()$`__init__`(...); self$init_shrink <- init_shrink
  },
  build = function(input_shape) {
    self$w <- self$add_weight(
      name = "shrink", shape = shape(1L),
      initializer = initializer_constant(stats::qlogis(self$init_shrink)), # logit init
      trainable = TRUE
    )
  },
  call = function(z, mask = NULL) tf$nn$sigmoid(self$w) * z
)
layer_learnable_shrink <- function(object = NULL, init_shrink = 0.5, ...) {
  keras::create_layer(LearnableShrink, object, list(init_shrink = init_shrink, ...))
}


# ---- Label-smoothed categorical cross-entropy (uses tf.keras built-in) ----
ls_ce <- function(label_smoothing = 0.0) {
  loss_categorical_crossentropy(from_logits = FALSE, label_smoothing = label_smoothing)
}

# ---- Focal loss (multiclass, expects probs) ----
focal_loss <- function(gamma = 2.0, alpha = NULL, eps = 1e-7) {
  function(y_true, y_pred) {
    y_pred <- k_clip(y_pred, eps, 1 - eps)
    ce <- -k_sum(y_true * k_log(y_pred), axis = -1L) # per-node CE
    pt <- k_sum(y_true * y_pred, axis = -1L) # prob of the true class
    fl <- (1 - pt)^gamma * ce
    if (!is.null(alpha)) {
      # alpha can be a length-K vector (class weights)
      a <- k_sum(y_true * k_constant(alpha), axis = -1L)
      fl <- a * fl
    }
    k_mean(fl)
  }
}


dice_loss <- function(smooth = 1e-6) {
  function(y_true, y_pred) {
    # y_true/y_pred shape: [B, N, K]
    num <- 2 * k_sum(y_true * y_pred, axis = c(1L,2L)) + smooth
    den <- k_sum(y_true + y_pred, axis = c(1L,2L)) + smooth
    dice <- num / den
    1 - k_mean(dice)
  }
}


make_loss_list <- function(n_outputs, loss_name = c("ce","lsce","focal","dice"),
                           ls = 0.0, gamma = 2.0) {
  loss_name <- match.arg(loss_name)
  base <- switch(loss_name,
                 ce = "categorical_crossentropy",
                 lsce = ls_ce(label_smoothing = ls),
                 focal = focal_loss(gamma = gamma),
                 dice = dice_loss())
  rep(list(base), n_outputs)
}


make_loss_weights <- function(n_outputs) {
  if (n_outputs == 1L) return(1.0)
  c(1.0, rep(0.3 / max(1, n_outputs - 1L), n_outputs - 1L))
}
node_degree <- function(A) rowSums((A + t(A)) > 0)
degree_bins <- function(deg) cut(deg, breaks = c(-Inf,0,2,5,Inf),
                                 labels = c("deg=0","deg 1???2","deg 3???5",">5"))
