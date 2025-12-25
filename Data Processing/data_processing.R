
auto_load_libraries <- function() {
  pkgs <- c(
    "Matrix", "igraph", "keras", "tensorflow", "reticulate",
    "aricode", "ggforce", "ggraph", "tidygraph", "Rtsne", "umap", "mclust","MLmetrics","car","caret","RANN",
    "torch","MASS","cluster","GGally","shiny","DT","data.table","moments","ROCR","jsonlite","RColorBrewer","dplyr","readr"
  )
  
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message(sprintf("Installing missing package: %s", pkg))
      install.packages(pkg)
    }
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}



load_graph_data <- function(
    edge_file,
    community_file = NULL,
    directed = FALSE,
    min_community_size = 1
) {
  library(igraph)
  
  
  edges <- read.table(edge_file)
  g <- graph_from_data_frame(edges, directed = directed)
  
 
  if (!is.null(community_file)) {
    circles <- readLines(community_file)
    community_list <- strsplit(circles, "\t")
    comm_membership <- rep(NA, vcount(g))
    
    for (i in seq_along(community_list)) {
      members <- as.character(community_list[[i]][-1])
      comm_membership[V(g)$name %in% members] <- i
    }
    
    V(g)$label <- comm_membership
    g <- induced_subgraph(g, !is.na(V(g)$label))
    
    
    tab <- table(V(g)$label)
    keep_labels <- names(tab[tab >= min_community_size])
    g <- induced_subgraph(g, V(g)$label %in% keep_labels)
    V(g)$label <- as.integer(factor(V(g)$label))  # clean factor
  } else {
    V(g)$label <- NA  
  }
  
 
  A <- as_adjacency_matrix(g, sparse = FALSE)
  n_nodes <- vcount(g)
  
 
  X <- scale(cbind(
    degree = degree(g),
    closeness = closeness(g),
    betweenness = betweenness(g),
    pagerank = page_rank(g)$vector,
    clustering = transitivity(g, type = "local")
  ))
  X[is.na(X)] <- 0
  
 
  C_train <- scale(degree(g))
  
  list(
    graph = g,
    A = A,
    X = X,
    C_train = C_train,
    y = if (!all(is.na(V(g)$label))) V(g)$label else NULL,
    n_nodes = n_nodes
  )
}



generate_clique_tensor <- function(graph, min_size = 3) {
  cliques <- max_cliques(graph, min = min_size)
  
  pairs <- do.call(rbind, lapply(cliques, function(cl) {
    if (length(cl) < 2) return(NULL)
    cl_names <- V(graph)$name[cl]
    t(combn(cl_names, 2))
  }))
  
  pairs <- unique(pairs)
  name_to_index <- setNames(seq_len(vcount(graph)) - 1L, V(graph)$name)
  mapped <- matrix(
    c(name_to_index[pairs[, 1]], name_to_index[pairs[, 2]]),
    ncol = 2
  )
  
  tf$constant(mapped, dtype = tf$int32)
}


compute_triangle_matrix <- function(adj_matrix) {
  triangle_matrix <- adj_matrix %*% adj_matrix %*% adj_matrix / 2
  as.matrix(triangle_matrix)
}

training_parameters <- function(X, C_train, A, triangle_matrix) {
  n_nodes <- nrow(X)
  list(
    X_batch = array(X, dim = c(1, n_nodes, ncol(X))),
    C_batch = array(C_train, dim = c(1, n_nodes, 1)),
    A_batch = array(A, dim = c(1, n_nodes, n_nodes)),
    T_batch = array(triangle_matrix, dim = c(1, n_nodes, n_nodes))
  )
}


generated_clique_tensor <- function(graph, min_size = 3) {
  cliques <- igraph::max_cliques(graph, min = min_size)
  
 
  valid_cliques <- Filter(function(cl) length(cl) >= 2, cliques)
  
  if (length(valid_cliques) == 0) {
    warning("No valid pairs extracted from cliques")
    return(NULL)
  }
  
  pairs_list <- lapply(valid_cliques, function(cl) {
    cl_names <- V(graph)$name[cl]
    if (length(cl_names) >= 2) {
      t(combn(cl_names, 2))
    } else {
      NULL
    }
  })
  
  pairs <- do.call(rbind, pairs_list)
  pairs <- unique(pairs)
  
  name_to_index <- setNames(seq_len(vcount(graph)) - 1L, V(graph)$name)
  mapped <- matrix(
    c(name_to_index[pairs[, 1]], name_to_index[pairs[, 2]]),
    ncol = 2
  )
  
  tf$constant(mapped, dtype = tf$int32)
}


generate_clique_matrix <- function(graph, min_size = 3) {
  cliques <- igraph::max_cliques(graph, min = min_size)
  mat <- matrix(0, nrow = vcount(graph), ncol = vcount(graph))
  
  for (cl in cliques) {
    nodes <- as.integer(cl)
    mat[nodes, nodes] <- 1
  }
  diag(mat) <- 0
  mat
}

generated_clique_matrix <- function(clique_tensor, n_nodes) {
  mat <- matrix(0, nrow = n_nodes, ncol = n_nodes)
  pairs <- as.array(clique_tensor)
  for (i in seq_len(nrow(pairs))) {
    mat[pairs[i, 1] + 1, pairs[i, 2] + 1] <- 1
    mat[pairs[i, 2] + 1, pairs[i, 1] + 1] <- 1  
  }
  mat
}


compute_motif_features <- function(graph) {
  c_vec <- scale(degree(graph))
  t_mat <- graph %>% as_adjacency_matrix(sparse = FALSE) %*% 
    as_adjacency_matrix(sparse = FALSE) %*% 
    as_adjacency_matrix(sparse = FALSE)
  t_vec <- scale(diag(t_mat))
  clique_counts <- rep(0, vcount(graph))
  for (cl in max_cliques(graph, min = 3)) {
    clique_counts[cl] <- clique_counts[cl] + 1
  }
  q_vec <- scale(clique_counts)
  motif_mat <- cbind(c_vec, t_vec, q_vec)
  motif_mat[is.na(motif_mat)] <- 0
  motif_mat
}


compute_motif_features2 <- function(graph) {
  adj <- igraph::as_adjacency_matrix(graph = graph, sparse = FALSE)
  triangle_counts <- diag(adj %*% adj %*% adj) / 2
  centrality <- igraph::degree(graph)
  clique_scores <- sapply(1:igraph::vcount(graph), function(v) {
    sum(sapply(igraph::max_cliques(graph, min = 3), function(cl) v %in% cl))
  })
  
  motif_features <- cbind(centrality, triangle_counts, clique_scores)
  return(motif_features)
}


prepare_graph_batch_inputs <- function(graph_list, num_clusters) {
  n_graphs <- length(graph_list)
  n_nodes <- nrow(graph_list[[1]]$X)
  n_features <- ncol(graph_list[[1]]$X)
  
  # Prepare input arrays
  X_batch <- array(0, dim = c(n_graphs, n_nodes, n_features))
  A_batch <- array(0, dim = c(n_graphs, n_nodes, n_nodes))
  C_batch <- array(0, dim = c(n_graphs, n_nodes, 1))
  T_batch <- array(0, dim = c(n_graphs, n_nodes, n_nodes))
  Q_batch <- array(0, dim = c(n_graphs, n_nodes, n_nodes))
  M_batch <- array(0, dim = c(n_graphs, n_nodes, 3))
  
  dummy_y <- array(0, dim = c(n_graphs, n_nodes, num_clusters))  
  
  for (i in 1:n_graphs) {
    g <- graph_list[[i]]
    X_batch[i, , ] <- g$X
    A_batch[i, , ] <- g$A
    C_batch[i, , 1] <- scale(igraph::degree(g$graph))
    
    
    adj <- igraph::as_adjacency_matrix(graph = g$graph, sparse = FALSE)
    T_mat <- adj %*% adj %*% adj
    T_batch[i, , ] <- T_mat
    
   
    Q_mat <- matrix(0, nrow = nrow(g$A), ncol = ncol(g$A))
    cliques <- igraph::max_cliques(graph = g$graph, min = 3)
    for (cl in cliques) {
      Q_mat[cl, cl] <- 1
    }
    Q_batch[i, , ] <- Q_mat
    
    
    M_batch[i, , ] <- compute_motif_features2(g$graph)
  }
  
  inputs <- list(
    X_input = X_batch,
    A_input = A_batch,
    Centrality_input = C_batch,
    Triangle_input = T_batch,
    Clique_input = Q_batch,
    Motif_input = M_batch
  )
  
  return(list(inputs = inputs, dummy_y = dummy_y))
}



sample_graph_data <- function(X, A, C, T, Q, sample_size = 1000, seed = 42) {
  set.seed(seed)
  
  
  n_nodes <- nrow(X)
  sample_size <- min(sample_size, n_nodes)
  
  
  sampled_indices <- sort(sample(1:n_nodes, sample_size))
  
 
  X_sampled <- X[sampled_indices, , drop = FALSE]
  A_sampled <- A[sampled_indices, sampled_indices, drop = FALSE]
  C_sampled <- C[sampled_indices]
  T_sampled <- T[sampled_indices, sampled_indices, drop = FALSE]
  Q_sampled <- Q[sampled_indices, sampled_indices, drop = FALSE]
  
  list(
    X = X_sampled,
    A = A_sampled,
    C = C_sampled,
    T = T_sampled,
    Q = Q_sampled,
    indices = sampled_indices
  )
}


run_graph_eda <- function(graph_data) {
  library(igraph)
  library(ggplot2)
  library(reshape2)
  library(viridis)
  
  g <- graph_data$graph
  A <- graph_data$A
  X <- graph_data$X
  cluster_labels <- graph_data$cluster_labels
  class_labels <- graph_data$class_labels
  
  cat("Graph Summary:\n")
  print(summary(g))
  
  cat("\nBasic Graph Stats:\n")
  cat(sprintf("Nodes: %d | Edges: %d | Components: %d\n",
              vcount(g), ecount(g), components(g)$no))
  
  cat("\nCluster Distribution:\n")
  print(table(cluster_labels))
  
  cat("\nClass Distribution:\n")
  print(table(class_labels))
  
 
  deg <- degree(g)
  deg_df <- data.frame(degree = deg)
  print(
    ggplot(deg_df, aes(x = degree)) +
      geom_histogram(bins = 30, fill = "steelblue", color = "white") +
      theme_minimal() +
      labs(title = "Degree Distribution", x = "Degree", y = "Count")
  )
  
 
  adj_df <- as.data.frame(as.table(A))
  colnames(adj_df) <- c("Row", "Col", "Value")
  
  print(ggplot(adj_df, aes(x = Col, y = Row, fill = Value)) +
    geom_tile() +
    scale_fill_viridis_c(option = "D") +
    coord_fixed() +
    theme_minimal() +
    labs(title = "Adjacency Matrix (Heatmap)", x = "Column", y = "Row"))
  
  
  
  X_df <- as.data.frame(X[, 1:min(ncol(X), 5)])
  X_df$cluster <- factor(cluster_labels)
  X_melt <- melt(X_df, id.vars = "cluster")
  
  print(
    ggplot(X_melt, aes(x = value, fill = cluster)) +
      geom_density(alpha = 0.6) +
      facet_wrap(~variable, scales = "free") +
      theme_minimal() +
      ggtitle("Feature Distributions (first 5)") +
      theme(legend.position = "bottom")
  )
  

  plot(
    g,
    vertex.color = rainbow(length(unique(cluster_labels)))[cluster_labels],
    vertex.label = NA,
    vertex.size = 6,
    edge.arrow.size = 0.3,
    layout = layout_with_fr(g),
    main = "Graph Colored by Cluster Labels"
  )
  
  
  motifs <- compute_motif_features(g)
  colnames(motifs) <- c("centrality", "triangles", "cliques")
  layout <- layout_with_fr(g)
  
  par(mfrow = c(1, 3), mar = c(1, 1, 2, 1))
  for (i in 1:3) {
    V(g)$color <- viridis(100)[rank(motifs[, i])]
    plot(g, layout = layout, vertex.label = NA, vertex.size = 6,
         edge.arrow.size = 0.3, main = colnames(motifs)[i])
  }
  par(mfrow = c(1, 1))
}


library(igraph)
library(FNN)

create_adjacency_matrices <- function(X,
                                         method = c("binary", "cosine", "gaussian", "inv_dist"),
                                         k = 5,
                                         sigma = 1,
                                         epsilon = 1e-5) {
  method <- match.arg(method)
  n <- nrow(X)
  
  A_binary <- matrix(0, n, n)
  A_weighted <- matrix(0, n, n)
  
  if (method == "binary") {
    knn <- get.knn(X, k = k)
    for (i in 1:n) {
      A_binary[i, knn$nn.index[i, ]] <- 1
    }
    A_binary <- A_binary + t(A_binary)
    A_binary[A_binary > 1] <- 1
    A_weighted <- A_binary 
    
  } else if (method == "cosine") {
    normalize <- function(x) x / sqrt(sum(x^2))
    X_norm <- t(apply(X, 1, normalize))
    sim <- X_norm %*% t(X_norm)
    diag(sim) <- 0
    A_weighted <- sim
    A_binary <- (sim > 0) * 1
    
  } else if (method == "gaussian") {
    for (i in 1:n) {
      for (j in 1:n) {
        if (i != j) {
          dist2 <- sum((X[i, ] - X[j, ])^2)
          A_weighted[i, j] <- exp(-dist2 / (2 * sigma^2))
        }
      }
    }
    A_binary <- (A_weighted > 0) * 1
    
  } else if (method == "inv_dist") {
    for (i in 1:n) {
      for (j in 1:n) {
        if (i != j) {
          dist <- sqrt(sum((X[i, ] - X[j, ])^2))
          A_weighted[i, j] <- 1 / (dist + epsilon)
        }
      }
    }
    A_binary <- (A_weighted > 0) * 1
  }
  
  g <- graph_from_adjacency_matrix(A_weighted, mode = "undirected", weighted = TRUE)
  
  return(list(
    graph = g,
    A_binary = A_binary,
    A_weighted = A_weighted
  ))
}


geda <- function(graph_data) {
  library(igraph)
  library(ggplot2)
  library(reshape2)
  library(viridis)
  
  g <- graph_data$graph
  A_list <- graph_data$A_list
  X <- graph_data$X
  cluster_labels <- graph_data$cluster_labels
  class_labels <- graph_data$class_labels
  
  cat("Graph Summary:\n")
  print(summary(g))
  
  cat("\nBasic Graph Stats:\n")
  cat(sprintf("Nodes: %d | Edges: %d | Components: %d\n",
              vcount(g), ecount(g), components(g)$no))
  
  cat("\nCluster Distribution:\n")
  print(table(cluster_labels))
  
  cat("\nClass Distribution:\n")
  print(table(class_labels))
  
 
  deg <- degree(g)
  deg_df <- data.frame(degree = deg)
  print(
    ggplot(deg_df, aes(x = degree)) +
      geom_histogram(bins = 30, fill = "steelblue", color = "white") +
      theme_minimal() +
      labs(title = "Degree Distribution", x = "Degree", y = "Count")
  )
  
 
  for (method in names(A_list)) {
    cat(sprintf("\nAdjacency Matrix: %s\n", method))
    A <- A_list[[method]]
    adj_df <- as.data.frame(as.table(A))
    colnames(adj_df) <- c("Row", "Col", "Value")
    
    print(
      ggplot(adj_df, aes(x = Col, y = Row, fill = Value)) +
        geom_tile() +
        scale_fill_viridis_c(option = "D") +
        coord_fixed() +
        theme_minimal() +
        labs(title = paste("Adjacency Matrix:", method), x = "Column", y = "Row")
    )
  }
  

  X_df <- as.data.frame(X[, 1:min(ncol(X), 5)])
  X_df$cluster <- factor(cluster_labels)
  X_melt <- melt(X_df, id.vars = "cluster")
  
  print(
    ggplot(X_melt, aes(x = value, fill = cluster)) +
      geom_density(alpha = 0.6) +
      facet_wrap(~variable, scales = "free") +
      theme_minimal() +
      ggtitle("Feature Distributions (first 5)") +
      theme(legend.position = "bottom")
  )
  

  plot(
    g,
    vertex.color = rainbow(length(unique(cluster_labels)))[cluster_labels],
    vertex.label = NA,
    vertex.size = 6,
    edge.arrow.size = 0.3,
    layout = layout_with_fr(g),
    main = "Graph Colored by Cluster Labels"
  )
  
 
  motifs <- compute_motif_features(g)
  colnames(motifs) <- c("centrality", "triangles", "cliques")
  layout <- layout_with_fr(g)
  
  par(mfrow = c(1, 3), mar = c(1, 1, 2, 1))
  for (i in 1:3) {
    V(g)$color <- viridis(100)[rank(motifs[, i])]
    plot(g, layout = layout, vertex.label = NA, vertex.size = 6,
         edge.arrow.size = 0.3, main = colnames(motifs)[i])
  }
  par(mfrow = c(1, 1))
}



plot_adj_mat <- function(A, X = NULL, matrix_name = "Adjacency Matrix") {
  library(igraph)
  library(ggplot2)
  library(reshape2)
  library(viridis)
  
  
  if (nrow(A) != ncol(A)) stop("Matrix A must be square.")
  
  cat(sprintf("\n===== %s Summary =====\n", matrix_name))
  cat(sprintf("Size: %d x %d\n", nrow(A), ncol(A)))
  cat(sprintf("Density: %.4f\n", sum(A > 0) / (nrow(A)^2)))
  cat(sprintf("Symmetric: %s\n", all(A == t(A))))
  
 
  g <- graph_from_adjacency_matrix(A, mode = "undirected", weighted = TRUE)
  
 
  A_df <- as.data.frame(as.table(A))
  colnames(A_df) <- c("Row", "Col", "Value")
  
  print(
    ggplot(A_df, aes(x = Col, y = Row, fill = Value)) +
      geom_tile() +
      scale_fill_viridis_c(option = "C") +
      coord_fixed() +
      theme_minimal() +
      labs(title = paste(matrix_name, "- Heatmap"), x = "Node j", y = "Node i")
  )
  

  deg <- rowSums(A > 0)
  deg_df <- data.frame(degree = deg)
  
  print(
    ggplot(deg_df, aes(x = degree)) +
      geom_histogram(bins = 30, fill = "steelblue", color = "white") +
      theme_minimal() +
      labs(title = paste(matrix_name, "- Degree Distribution"), x = "Degree", y = "Count")
  )
  
  
  weights <- A[upper.tri(A)]
  weights <- weights[weights > 0]
  
  if (length(weights) > 0) {
    weights_df <- data.frame(weight = weights)
    print(
      ggplot(weights_df, aes(x = weight)) +
        geom_histogram(bins = 30, fill = "forestgreen", color = "white") +
        theme_minimal() +
        labs(title = paste(matrix_name, "- Edge Weight Distribution"), x = "Weight", y = "Count")
    )
  } else {
    cat("No non-zero edge weights to plot.\n")
  }
  
 
  
  layout_safe <- layout_with_fr(g, weights = pmax(E(g)$weight, 1e-5))
  plot(
    g,
    layout = layout_with_fr(g),
    vertex.label = NA,
    vertex.size = 5,
    edge.width = E(g)$weight,
    main = paste(matrix_name, "- Graph View")
  )
  
 
  if (!is.null(X)) {
    D <- dist(X)
    hist(as.vector(D), breaks = 30,
         main = paste(matrix_name, "- Pairwise Distances in Feature Space"),
         xlab = "Euclidean Distance", col = "darkorange")
  }
}


eda_graph_features <- function(X_features, feature_names = NULL) {
  library(ggplot2)
  library(reshape2)
  library(GGally)
  library(viridis)
  
  if (is.null(feature_names)) {
    feature_names <- colnames(X_features)
    if (is.null(feature_names)) {
      feature_names <- paste0("V", 1:ncol(X_features))
      colnames(X_features) <- feature_names
    }
  }
  
  cat("===== Basic Summary =====\n")
  print(summary(X_features))
  
  cat("\n===== Feature Correlation Matrix =====\n")
  corr_matrix <- cor(X_features)
  print(round(corr_matrix, 2))
  
  
  corr_df <- melt(corr_matrix)
  gg_corr <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    scale_fill_viridis_c(option = "C", limits = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Feature Correlation Heatmap", x = "", y = "", fill = "Correlation")
  print(gg_corr)
  
  
  feature_df <- as.data.frame(X_features)
  feature_melt <- melt(feature_df, variable.name = "Feature")
  
  gg_density <- ggplot(feature_melt, aes(x = value, fill = Feature)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~Feature, scales = "free", ncol = 3) +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(title = "Density Plots of Graph Features", x = "Scaled Value", y = "Density")
  print(gg_density)
  
  
  if (ncol(X_features) > 1) {
    max_plot_dim <- min(ncol(X_features), 5)
    plot_data <- as.data.frame(X_features[, 1:max_plot_dim, drop = FALSE])
    colnames(plot_data) <- feature_names[1:max_plot_dim]
    
    cat("\n===== Pairwise Scatterplots (first", max_plot_dim, "features) =====\n")
    print(GGally::ggpairs(plot_data))
  }
}

graph_feature_analysis <- function(X_features, 
                                   labels = NULL, 
                                   output_dir = "graph_feature_plots",
                                   prefix = "graph") {
  library(ggplot2)
  library(reshape2)
  library(GGally)
  library(viridis)
  library(Rtsne)
  library(randomForest)
  library(gridExtra)
  
  dir.create(output_dir, showWarnings = FALSE)
  
  if (is.null(colnames(X_features))) {
    colnames(X_features) <- paste0("V", 1:ncol(X_features))
  }
  
 
  corr_matrix <- cor(X_features)
  corr_df <- melt(corr_matrix)
  
  p_corr <- ggplot(corr_df, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_viridis_c(limits = c(-1, 1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Feature Correlation Heatmap", x = "", y = "", fill = "Correlation")
  ggsave(file.path(output_dir, paste0(prefix, "_correlation_heatmap.png")), p_corr)
  
  
  melted <- melt(as.data.frame(X_features))
  p_density <- ggplot(melted, aes(x = value, fill = variable)) +
    geom_density(alpha = 0.6) +
    facet_wrap(~variable, scales = "free", ncol = 3) +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(title = "Density Distributions", x = "Scaled Value", y = "Density")
  ggsave(file.path(output_dir, paste0(prefix, "_density_plots.png")), p_density)
  
  
  pca <- prcomp(X_features, scale. = FALSE)
  pca_df <- data.frame(pca$x[, 1:2], label = if (!is.null(labels)) as.factor(labels) else NULL)
  p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = label)) +
    geom_point(size = 2, alpha = 0.8) +
    theme_minimal() +
    labs(title = "PCA of Graph Features")
  ggsave(file.path(output_dir, paste0(prefix, "_pca.png")), p_pca)
  
 
  tsne_out <- Rtsne(X_features, dims = 2, perplexity = 30, verbose = FALSE)
  tsne_df <- data.frame(tsne_out$Y, label = if (!is.null(labels)) as.factor(labels) else NULL)
  colnames(tsne_df) <- c("Dim1", "Dim2", "label")
  p_tsne <- ggplot(tsne_df, aes(x = Dim1, y = Dim2, color = label)) +
    geom_point(size = 2, alpha = 0.8) +
    theme_minimal() +
    labs(title = "t-SNE of Graph Features")
  ggsave(file.path(output_dir, paste0(prefix, "_tsne.png")), p_tsne)
  
  
  if (!is.null(labels)) {
    rf <- randomForest(x = X_features, y = as.factor(labels), importance = TRUE)
    importance_df <- data.frame(Feature = rownames(rf$importance),
                                Importance = rf$importance[, "MeanDecreaseGini"])
    importance_df <- importance_df[order(-importance_df$Importance), ]
    
    p_imp <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_col(fill = "darkorange") +
      coord_flip() +
      theme_minimal() +
      labs(title = "Feature Importance vs Labels", x = "Feature", y = "Importance (Gini)")
    ggsave(file.path(output_dir, paste0(prefix, "_feature_importance.png")), p_imp)
  }
  
  
  cat("??? EDA completed.\n")
  cat("Plots saved to:", normalizePath(output_dir), "\n")
}


find_bridges <- function(g) {
  bridges <- c()
  for (e in E(g)) {
    g_temp <- delete_edges(g, e)
    if (components(g_temp)$no > components(g)$no) {
      bridges <- c(bridges, e)
    }
  }
  return(E(g)[bridges])
}


eda_clustermetrics <- function(g, community = NULL, top_k_hubs = 5) {
  library(igraph)
  layout <- layout_with_fr(g)
  
  
  degree_vec <- degree(g)
  hubs <- order(degree_vec, decreasing = TRUE)[1:top_k_hubs]
  
  
  bridge_edges <- find_bridges(g)

  
  
  articulation_nodes <- articulation.points(g)
  
  
  triangle_vec <- count_triangles(g)
  

  clique_counts <- rep(0, vcount(g))
  cliques <- max_cliques(g, min = 3)
  for (cl in cliques) {
    clique_counts[cl] <- clique_counts[cl] + 1
  }
  

  edge_density_val <- edge_density(g)
  

  if (is.null(community)) {
    community <- cluster_louvain(g)
  }
  modularity_score <- modularity(community)
  
 
  compute_conductance <- function(g, membership) {
    conds <- numeric()
    for (comm in unique(membership)) {
      S <- which(membership == comm)
      cut_size <- sum(sapply(S, function(v) sum(!((neighbors(g, v) %in% S)))))
      vol <- sum(degree(g, S))
      conds <- c(conds, cut_size / max(vol, 1))
    }
    return(mean(conds))
  }
  conductance_val <- compute_conductance(g, membership(community))
  
 
  cluster_info <- data.frame(community = membership(community))
  cluster_sizes <- table(cluster_info$community)
  intra_edges <- sapply(unique(membership(community)), function(comm) {
    subg <- induced_subgraph(g, which(membership(community) == comm))
    ecount(subg)
  })
  intra_edge_ratio <- intra_edges / cluster_sizes
  
  
  par(mfrow = c(2, 3), mar = c(1, 1, 2, 1))
  
 
  V(g)$color <- ifelse(1:vcount(g) %in% hubs, "red", "lightgray")
  plot(g, layout = layout, vertex.label = NA, main = "Hubs (Top Degree)")
  
  
  E(g)$color <- ifelse(E(g) %in% bridge_edges, "red", "gray")
  V(g)$color <- "lightblue"
  plot(g, layout = layout, vertex.label = NA, main = "Bridges")
  
  
  V(g)$color <- ifelse(1:vcount(g) %in% articulation_nodes, "orange", "lightgray")
  plot(g, layout = layout, vertex.label = NA, main = "Articulation Points")
  
  
  V(g)$color <- rainbow(length(unique(membership(community))))[membership(community)]
  plot(g, layout = layout, vertex.label = NA, main = "Community Structure")
  
  
  V(g)$size <- triangle_vec + 3
  V(g)$color <- viridis::viridis(100)[rank(triangle_vec)]
  plot(g, layout = layout, vertex.label = NA, main = "Triangles per Node")
  
 
  V(g)$size <- clique_counts + 3
  V(g)$color <- viridis::magma(100)[rank(clique_counts)]
  plot(g, layout = layout, vertex.label = NA, main = "Clique Memberships")
  
  par(mfrow = c(1, 1))
  
 
  return(list(
    hubs = hubs,
    bridges = bridge_edges,
    articulation_points = articulation_nodes,
    triangles = triangle_vec,
    clique_memberships = clique_counts,
    edge_density = edge_density_val,
    modularity = modularity_score,
    conductance = conductance_val,
    intra_edge_ratio = intra_edge_ratio
  ))
}


generate_clustered_graph <- function(n_nodes = 100,
                                     n_clusters = 3,
                                     n_features = 8,
                                     seed = 42) {
  set.seed(seed)
  library(MASS)
  library(igraph)
  
 
  centers <- matrix(runif(n_clusters * n_features, -5, 5), ncol = n_features)
  
 
  nodes_per_cluster <- rep(n_nodes / n_clusters, n_clusters)
  X <- do.call(rbind, lapply(1:n_clusters, function(k) {
    mvrnorm(nodes_per_cluster[k], mu = centers[k, ], Sigma = diag(n_features))
  }))
  colnames(X) <- paste0("f", 1:n_features)
  
 
  labels <- rep(1:n_clusters, each = nodes_per_cluster[1])
  
 
  g <- make_empty_graph(n = n_nodes, directed = FALSE)
  V(g)$name <- as.character(1:n_nodes)
  V(g)$label <- labels
  
 
  for (j in 1:n_features) {
    g <- set_vertex_attr(g, name = paste0("f", j), value = X[, j])
  }
  
  return(list(
    graph = g,
    features = X,
    labels = labels
  ))
}



add_weighted_edges <- function(g, features, method = c("gaussian", "normal"), sigma = 1.0, k = NULL) {
  method <- match.arg(method)
  n <- vcount(g)
  dist_sq <- as.matrix(dist(features))^2
  
 
  if (method == "gaussian") {
    W <- exp(-dist_sq / (2 * sigma^2))
  } else if (method == "normal") {
    W <- 1 / (sqrt(dist_sq) + 1e-8)
  }
  
  diag(W) <- 0  
  
  
  if (!is.null(k)) {
    for (i in 1:n) {
      threshold <- sort(W[i, ], decreasing = TRUE)[k + 1]
      W[i, W[i, ] < threshold] <- 0
    }
  }
  

  W <- pmax(W, t(W))
  
  
  edge_idx <- which(W > 0, arr.ind = TRUE)
  edge_idx <- edge_idx[edge_idx[, 1] < edge_idx[, 2], ] 
  
 
  edge_vec <- as.vector(t(edge_idx)) 
  weights <- W[edge_idx]
  
 
  g <- add_edges(g, edge_vec, attr = list(weight = weights))
  return(g)
}


graph_from_df_custom <- function (d, directed = TRUE, vertices = NULL) {
  d <- as.data.frame(d)
  if (!is.null(vertices)) vertices <- as.data.frame(vertices)
  if (ncol(d) < 2) stop("the data frame should contain at least two columns")
  if (any(is.na(d[, 1:2]))) {
    cli::cli_warn("In {.code d}, {.code NA} elements were replaced with string {.str NA}.")
    d[, 1:2][is.na(d[, 1:2])] <- "NA"
  }
  if (!is.null(vertices) && any(is.na(vertices[, 1]))) {
    cli::cli_warn("In {.code vertices[,1]}, {.code NA} elements were replaced with string {.str NA}.")
    vertices[, 1][is.na(vertices[, 1])] <- "NA"
  }
  names <- unique(c(as.character(d[, 1]), as.character(d[, 2])))
  if (!is.null(vertices)) {
    names2 <- names
    vertices <- as.data.frame(vertices)
    if (ncol(vertices) < 1) stop("Vertex data frame contains no rows")
    names <- as.character(vertices[, 1])
    if (any(duplicated(names))) stop("Duplicate vertex names")
    if (any(!names2 %in% names)) stop("Some vertex names in edge list are not listed in vertex data frame")
  }
  g <- make_empty_graph(n = 0, directed = directed)
  attrs <- list(name = names)
  if (!is.null(vertices) && ncol(vertices) > 1) {
    for (i in 2:ncol(vertices)) {
      newval <- vertices[, i]
      if (inherits(newval, "factor")) newval <- as.character(newval)
      attrs[[names(vertices)[i]]] <- newval
    }
  }
  g <- add_vertices(g, length(names), attr = attrs)
  from <- as.character(d[, 1]); to <- as.character(d[, 2])
  edges <- rbind(match(from, names), match(to, names))
  attrs <- list()
  if (ncol(d) > 2) {
    for (i in 3:ncol(d)) {
      newval <- d[, i]
      if (inherits(newval, "factor")) newval <- as.character(newval)
      attrs[[names(d)[i]]] <- newval
    }
  }
  g <- add_edges(g, edges, attr = attrs)
  g
}

