
build_full_inputs <- function(model, X_features, A, C_vec, D_vec) {
  n_nodes <- nrow(X_features)
  n_features <- ncol(X_features)
  
  X_in <- array(X_features, dim = c(1, n_nodes, n_features))
  A_in <- array(as.matrix(A), dim = c(1, n_nodes, n_nodes))
  C_in <- array(C_vec, dim = c(1, n_nodes, 1))
  D_in <- array(D_vec, dim = c(1, n_nodes, 1))
  
  if (length(model$inputs) == 4L) {
    # (X, A, C, D)
    list(X_in, A_in, C_in, D_in)
  } else if (length(model$inputs) == 5L) {
    # (X, A, C, D, M) ??? allow all edges for analysis
    M_in <- array(1, dim = c(1, n_nodes, n_nodes))
    list(X_in, A_in, C_in, D_in, M_in)
  } else {
    stop("Unexpected number of model inputs.")
  }
}



get_hierarchy_gate_outputs <- function(model,
                                       inputs,
                                       layer_pattern = "hierarchy_gate|hierarch_gate") {
  layer_names <- sapply(model$layers, function(l) l$name)
  idx <- which(grepl(layer_pattern, layer_names, ignore.case = TRUE))[1]
  if (is.na(idx)) stop("Could not find HierarchyGate layer in model.")
  
  hg_layer <- model$layers[[idx]]
  hg_model <- keras_model(inputs = model$inputs, outputs = hg_layer$output)
  H_pred <- predict(hg_model, inputs) # (1, N, 1) or (N,1)
  
  if (length(dim(H_pred)) == 3L) {
    g_vec <- as.numeric(H_pred[1, , 1])
  } else if (length(dim(H_pred)) == 2L) {
    g_vec <- as.numeric(H_pred[, 1])
  } else {
    stop("Unexpected HierarchyGate tensor shape.")
  }
  g_vec
}


get_final_embeddings <- function(model, inputs) {
  layer_names <- sapply(model$layers, function(l) l$name)
  L <- length(model$layers)
  emb_layer_name <- model$layers[[L - 1L]]$name
  cat("Using penultimate layer as embedding layer:", emb_layer_name, "\n")
  
  emb_model <- keras_model(
    inputs = model$inputs,
    outputs = get_layer(model, emb_layer_name)$output
  )
  
  Z_pred <- predict(emb_model, inputs)
  if (length(dim(Z_pred)) == 3L) {
    Z_final <- Z_pred[1, , ]
  } else if (length(dim(Z_pred)) == 2L) {
    Z_final <- Z_pred
  } else {
    stop("Unexpected Z_pred shape:", paste(dim(Z_pred), collapse = " x "))
  }
  as.matrix(Z_final)
}

Z_final <- get_final_embeddings(model, inputs_all) 

Z_mat <- Z_final

good_rows <- apply(Z_mat, 1, function(r) all(is.finite(r)))
if (!all(good_rows)) {
  warning("Some nodes had NA/Inf in embeddings and were dropped for t-SNE.")
}
Z_tsne <- Z_mat[good_rows, , drop = FALSE]

N_tsne <- nrow(Z_tsne)
if (N_tsne < 10) {
  stop("Too few valid nodes for t-SNE after filtering: N_tsne = ", N_tsne)
}

perplexity_val <- min(30, max(5, floor((N_tsne - 1) / 3)))
cat("Running t-SNE on", N_tsne, "nodes with perplexity =", perplexity_val, "\n")

set.seed(123)
ts <- Rtsne(
  Z_tsne,
  perplexity = perplexity_val,
  check_duplicates = FALSE,
  verbose = TRUE
)

idx_kept <- which(good_rows)

df_tsne <- data.frame(
  x = ts$Y[, 1],
  y = ts$Y[, 2],
  true_label = factor(y[idx_kept]),
  pred_label = factor(pred_label[idx_kept]),
  correct = factor(correct[idx_kept],
                   levels = c(FALSE, TRUE),
                   labels = c("wrong", "correct")),
  depth = as.numeric(D_all[idx_kept]),
  centrality = as.numeric(C_all[idx_kept]),
  gate = as.numeric(g_vec[idx_kept])
)


p_tsne_true <- ggplot(df_tsne, aes(x = x, y = y, colour = true_label)) +
  geom_point(alpha = 0.85, size = 1.2) +
  labs(x = "t-SNE 1", y = "t-SNE 2", colour = "True class") +
  ggtitle("t-SNE of final embeddings (true labels)") +
  theme_minimal(base_size = 11)


p_tsne_pred <- ggplot(df_tsne,
                      aes(x = x, y = y,
                          colour = pred_label,
                          shape = correct)) +
  geom_point(alpha = 0.9, size = 1.4) +
  labs(x = "t-SNE 1", y = "t-SNE 2",
       colour = "Predicted class",
       shape = "Prediction") +
  ggtitle("t-SNE of final embeddings (predicted class, correct vs wrong)") +
  theme_minimal(base_size = 11)


p_tsne_depth <- ggplot(df_tsne, aes(x = x, y = y, colour = depth)) +
  geom_point(alpha = 0.9, size = 1.3) +
  scale_colour_viridis_c(option = "plasma") +
  labs(x = "t-SNE 1", y = "t-SNE 2", colour = "Depth D_i") +
  ggtitle("t-SNE of final embeddings (depth prior D_i)") +
  theme_minimal(base_size = 11)


p_tsne_gate <- ggplot(df_tsne, aes(x = x, y = y, colour = gate)) +
  geom_point(alpha = 0.9, size = 1.3) +
  scale_colour_viridis_c(option = "magma") +
  labs(x = "t-SNE 1", y = "t-SNE 2", colour = "Gate g_i") +
  ggtitle("t-SNE of final embeddings (hierarchy gate g_i)") +
  theme_minimal(base_size = 11)


deg_vec <- as.numeric(Matrix::rowSums(A_raw))

df_hier <- data.frame(
  node = seq_len(N),
  depth = D_all,
  centrality = C_all,
  degree = deg_vec,
  label = factor(y),
  gate = g_vec
)


p_depth_gate <- ggplot(df_hier, aes(x = depth, y = gate, colour = label)) +
  geom_point(alpha = 0.6, size = 0.9) +
  labs(x = "Depth prior D_i", y = "Hierarchy gate g_i", colour = "Class") +
  ggtitle("Depth vs hierarchy gate on Squirrel") +
  theme_minimal(base_size = 11)


p_central_gate <- ggplot(df_hier, aes(x = centrality, y = gate, colour = label)) +
  geom_point(alpha = 0.6, size = 0.9) +
  labs(x = "Centrality prior C_i", y = "Hierarchy gate g_i", colour = "Class") +
  ggtitle("Centrality vs hierarchy gate on Squirrel") +
  theme_minimal(base_size = 11)

p_depth_central_gate <- ggplot(df_hier, aes(x = depth, y = centrality, colour = gate)) +
  geom_point(alpha = 0.8, size = 0.9) +
  scale_colour_viridis_c(option = "plasma") +
  labs(x = "Depth prior D_i", y = "Centrality prior C_i", colour = "Gate g_i") +
  ggtitle("Structural hierarchy and learned gate") +
  theme_minimal(base_size = 11)

