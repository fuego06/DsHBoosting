suppressPackageStartupMessages({
  library(Matrix); library(igraph); library(keras)
})


read_three <- function(base_dir, name = "WikiCS", add_one_for_R = TRUE) {
  d <- file.path(base_dir, name)
  edges <- as.matrix(read.table(file.path(d, "edges.txt"), header = FALSE))
  storage.mode(edges) <- "integer"
  if (add_one_for_R) edges <- edges + 1L
  X <- as.matrix(read.csv(file.path(d, "features.csv"), header = FALSE, check.names = FALSE))
  storage.mode(X) <- "numeric"
  y <- as.integer(read.csv(file.path(d, "labels.csv"), header = FALSE)[,1])
  list(edges = edges, X = X, y = y)
}

make_Araw <- function(edges, N) {
  i <- edges[,1]; j <- edges[,2]
  A <- sparseMatrix(i = c(i, j), j = c(j, i), x = 1, dims = c(N, N))
  A <- sign(A); diag(A) <- 0; drop0(A)
}


graph_feats_all <- function(A) {
  g <- graph_from_adjacency_matrix(A, mode = "undirected", diag = FALSE)
  g <- simplify(g, remove.multiple = TRUE, remove.loops = TRUE)
  lv <- cluster_louvain(g); memb <- lv$membership
  deg <- degree(g)
  neigh <- adjacent_vertices(g, V(g))
  same_cnt <- mapply(function(v, nb) if (length(nb)) sum(memb[nb] == memb[v]) else 0L,
                     seq_len(gorder(g)), neigh)
  C <- ifelse(deg > 0, same_cnt / pmax(deg, 1L), 0)
  C[!is.finite(C)] <- 0; C <- pmin(pmax(C, 0), 1)
  ecc <- suppressWarnings(eccentricity(g))
  if (!all(is.finite(ecc))) { mx <- if (any(is.finite(ecc))) max(ecc[is.finite(ecc)]) else 0; ecc[!is.finite(ecc)] <- mx }
  D <- if (max(ecc) > min(ecc)) (ecc - min(ecc)) / (max(ecc) - min(ecc)) else rep(0, length(ecc))
  list(C = matrix(as.numeric(C), ncol = 1), D = matrix(as.numeric(D), ncol = 1))
}


row_norm <- function(A) { d <- pmax(rowSums(A), 1); Diagonal(x = 1/d) %*% A }


row_l2 <- function(X, eps = 1e-12){ rs <- sqrt(rowSums(X*X)); rs[rs < eps] <- 1; X/rs }
fit_feature_prep <- function(X_train, use_tfidf = TRUE, l2_rows = TRUE, pca_dim = 256L) {
  idf <- if (use_tfidf) { df <- colSums(X_train > 0); log((nrow(X_train)+1)/(df+1)) + 1 } else NULL
  Xw <- if (!is.null(idf)) sweep(X_train, 2L, idf, `*`) else X_train
  if (l2_rows) Xw <- row_l2(Xw)
  rnk <- max(1, min(as.integer(pca_dim), ncol(Xw), nrow(Xw) - 1))
  pca <- prcomp(Xw, center = TRUE, scale. = FALSE, rank. = rnk)
  list(idf = idf, l2 = l2_rows, pca = pca, dim = rnk, use_tfidf = !is.null(idf), ncol = ncol(X_train))
}
apply_feature_prep <- function(X_new, prep) {
  stopifnot(is.matrix(X_new), ncol(X_new) == prep$ncol)
  Xf <- if (isTRUE(prep$use_tfidf)) sweep(X_new, 2L, prep$idf, `*`) else X_new
  if (isTRUE(prep$l2)) Xf <- row_l2(Xf)
  Xp <- predict(prep$pca, Xf)
  Xp[, seq_len(prep$dim), drop = FALSE]
}


augment_AX <- function(Xp, A_feat, gamma2 = 0.25, post_l2 = TRUE) {
  AX <- A_feat %*% Xp
  A2X <- (A_feat %*% A_feat) %*% Xp
  Z <- cbind(Xp, AX, gamma2 * A2X)
  if (post_l2) { rn <- sqrt(rowSums(Z*Z)); rn[rn == 0] <- 1; Z <- Z / rn }
  Z
}

to_categorical <- function(y, K){ Y <- matrix(0, length(y), K); Y[cbind(seq_along(y), y + 1L)] <- 1; Y }
stratified_60_20_20 <- function(y, seed = 8L) {
  set.seed(seed)
  classes <- sort(unique(y))
  tr <- va <- te <- integer(0)
  for (c in classes) {
    idx <- which(y == c); idx <- sample(idx); n <- length(idx)
    n_tr <- max(1L, floor(0.6*n)); n_va <- max(1L, floor(0.2*n))
    tr <- c(tr, idx[seq_len(n_tr)])
    va <- c(va, idx[(n_tr+1):(n_tr+n_va)])
    te <- c(te, idx[(n_tr+n_va+1):n])
  }
  list(train = sort(tr), val = sort(va), test = sort(te))
}


read_webkb_three <- function(base_dir, name = "Texas", add_one_for_R = TRUE) {
  d_edges <- file.path(base_dir, name, "edges.txt")
  d_feat <- file.path(base_dir, name, "features.csv")
  d_lab <- file.path(base_dir, name, "labels.csv")
  
  if (!file.exists(d_edges) || !file.exists(d_feat) || !file.exists(d_lab)) {
    stop("One or more files are missing in: ", file.path(base_dir, name))
  }
  
 
  edges <- as.matrix(read.table(d_edges, header = FALSE))
  storage.mode(edges) <- "integer"
  if (add_one_for_R) edges <- edges + 1L 
  

  X <- as.matrix(read.csv(d_feat, header = FALSE, check.names = FALSE))
  storage.mode(X) <- "numeric"
  

  y <- as.integer(read.csv(d_lab, header = FALSE)[, 1])
  

  if (nrow(X) != length(y)) {
    warning("Row count of features (", nrow(X), ") != length of labels (", length(y), ")")
  }
  
  list(edges = edges, X = X, y = y)
}

graph_report <- function(ds, directed_input = TRUE) {
  library(igraph)
  
  g_d <- graph_from_edgelist(ds$edges, directed = directed_input)
  V(g_d)$class <- ds$y
  g_u <- as.undirected(simplify(g_d, remove.multiple = TRUE, remove.loops = TRUE), mode = "collapse")
  V(g_u)$class <- ds$y
  N <- gorder(g_u)
  
 
  deg <- degree(g_u); deg_sum <- sum(deg)
  gini <- function(x) {
    x <- sort(as.numeric(x))
    n <- length(x); if (n == 0 || sum(x) == 0) return(0)
    (2 * sum(seq_len(n) * x) / (n * sum(x))) - (n + 1) / n
  }
  deg_gini <- gini(deg)
  top_share <- function(k) sum(sort(deg, decreasing = TRUE)[seq_len(min(k, length(deg)))]) / pmax(deg_sum, 1)
  top1_share <- top_share(1); top5_share <- top_share(5)
  
 
  comp <- components(g_u)
  giant <- induced_subgraph(g_u, which(comp$membership == which.max(comp$csize)))
  
  
  apl <- suppressWarnings(average.path.length(giant, directed = FALSE))
  diam <- suppressWarnings(diameter(giant, directed = FALSE))

  L <- laplacian_matrix(giant, sparse = FALSE)
  eigL <- sort(eigen(L, symmetric = TRUE, only.values = TRUE)$values)
  lambda2 <- if (length(eigL) >= 2) eigL[2] else NA_real_
  
  
  tri_total <- sum(count_triangles(giant)) / 3
  clust_global <- suppressWarnings(transitivity(giant, type = "global"))
  clust_median <- median(transitivity(giant, type = "localundirected", isolates = "zero"), na.rm = TRUE)
  
  
  el <- as_edgelist(g_u, names = FALSE)
  y <- ds$y
  same <- y[el[,1]] == y[el[,2]]
  homophily <- mean(same)
  K <- length(unique(y))
  edge_mat <- matrix(0L, K, K)
  for (i in seq_len(nrow(el))) {
    a <- y[el[i,1]] + 1L; b <- y[el[i,2]] + 1L
    edge_mat[a, b] <- edge_mat[a, b] + 1L
    if (a != b) edge_mat[b, a] <- edge_mat[b, a] + 1L
  }
  rownames(edge_mat) <- colnames(edge_mat) <- as.character(0:(K-1))
  per_class_hom <- sapply(0:(K-1), function(c) {
    v <- which(y == c); e_inc <- incident_edges(g_u, v, mode = "all")
    if (length(e_inc) == 0) return(NA_real_)
    ends_same <- 0; ends_tot <- 0
    for (Eset in e_inc) {
      for (e in Eset) {
        ee <- ends(g_u, e)
        a <- as.integer(ee[1]); b <- as.integer(ee[2])
        if (a %in% v || b %in% v) {
          ends_tot <- ends_tot + 1
          if (y[a] == y[b]) ends_same <- ends_same + 1
        }
      }
    }
    ends_same / pmax(ends_tot, 1)
  })
  names(per_class_hom) <- as.character(0:(K-1))
  

  rec <- if (directed_input) reciprocity(g_d) else NA_real_
  self_loops <- sum(which_loop(g_d))
  has_mult <- any_multiple(g_d)
  mult_extra <- if (has_mult) sum(count_multiple(g_d) - 1L) else 0L
  
  
  med_deg <- median(deg)
  knn_k_hint <- max(8L, min(32L, as.integer(2*med_deg + 2))) 
  tau_hint <- if (homophily < 0.4) 0.05 else if (homophily < 0.6) 0.10 else 0.20
  alpha_hint <- if (homophily < 0.0) 0.5 else 0.6 
  onehop_same <- function(g, y) {
    el <- as_edgelist(g, names = FALSE)
    mean(y[el[,1]] == y[el[,2]])
  } 
  twohop_same <- function(g, y) {
    A <- as_adj(g, sparse = TRUE)
    A2 <- (A %*% A); diag(A2) <- 0

    A_bin <- sign(A); A2_bin <- sign(A2) - A_bin; A2_bin[A2_bin < 0] <- 0
    idx <- which(A2_bin != 0, arr.ind = TRUE)
    if (nrow(idx) == 0) return(NA_real_)
    mean(y[idx[,1]] == y[idx[,2]])
  }
  h1 <- onehop_same(g_u, ds$y)
  h2 <- twohop_same(g_u, ds$y)

  kc <- coreness(g_u)
  kcore_summary <- c(min = min(kc), median = median(kc), max = max(kc))
  
  
  
  Au <- as_adj(g_u, sparse = TRUE) 
  N <- nrow(Au)
  class_conductance <- sapply(sort(unique(ds$y)), function(c) {
    Vc <- which(ds$y == c)
    Vnc <- setdiff(seq_len(N), Vc)
    if (length(Vc) == 0) return(NA_real_)
    cut_edges <- sum(Au[Vc, Vnc]) 
    vol_S <- sum(rowSums(Au[Vc, ]))
    as.numeric(cut_edges / pmax(vol_S, 1)) 
  })
  names(class_conductance) <- as.character(sort(unique(ds$y)))
  
  
  deg <- degree(g_u)
  deg_pct <- quantile(deg, probs = c(.5,.75,.9))
  
  list(
    nodes = N,
    edges = gsize(g_u),
    features_per_node = ncol(ds$X),
    classes = K,
    class_counts = table(ds$y),
    density_undirected = edge_density(g_u),
    
    degree_summary = summary(deg),
    degree_gini = deg_gini,
    hub_share_top1 = top1_share,
    hub_share_top5 = top5_share,
    
    components = list(n = comp$no, giant_size = max(comp$csize)),
    avg_path_length = apl,
    diameter = diam,
    algebraic_connectivity = lambda2,
    
    clustering_global = clust_global,
    clustering_median = clust_median,
    triangles = tri_total,
    
    assortativity_by_class = suppressWarnings(assortativity_nominal(g_u, as.integer(factor(y)), directed = FALSE)),
    homophily_overall = homophily,
    homophily_by_class = per_class_hom,
    edge_label_matrix = edge_mat,
   
    self_loops = self_loops,
    reciprocity = rec,
    has_multiedges = has_mult,
    extra_parallel_edges = mult_extra,
    two_hop_homophily = h2,
    kcore_summary = kcore_summary,
    class_conductance = class_conductance,
    degree_percentiles = deg_pct,
    
    suggest = list(k_top = knn_k_hint, tau_cos = tau_hint, init_alpha_blend = alpha_hint)
  )
}


stratified_split_60_20_20 <- function(y, seed = 8L) {
  set.seed(seed)
  classes <- sort(unique(y))
  tr <- va <- te <- integer(0)
  for (c in classes) {
    idx <- which(y == c); idx <- sample(idx); n <- length(idx)
    n_tr <- max(1L, floor(0.6*n))
    n_va <- max(1L, floor(0.2*n))
    tr <- c(tr, idx[seq_len(n_tr)])
    va <- c(va, idx[(n_tr+1):(n_tr+n_va)])
    te <- c(te, idx[(n_tr+n_va+1):n])
  }
  list(train = sort(tr), val = sort(va), test = sort(te))
}


read_triplet_ds <- function(base_dir, name = "Actor", add_one_for_R = TRUE) {
  d_edges <- file.path(base_dir, name, "edges.txt")
  d_feat <- file.path(base_dir, name, "features.csv")
  d_lab <- file.path(base_dir, name, "labels.csv")
  stopifnot(file.exists(d_edges), file.exists(d_feat), file.exists(d_lab))
  edges <- as.matrix(read.table(d_edges, header = FALSE))
  storage.mode(edges) <- "integer"
  if (add_one_for_R) edges <- edges + 1L # PyG exports 0-based
  X <- as.matrix(read.csv(d_feat, header = FALSE, check.names = FALSE))
  storage.mode(X) <- "numeric"
  y <- as.integer(read.csv(d_lab, header = FALSE)[,1])
  list(edges = edges, X = X, y = y)
}

acc_at <- function(P, y, idx) { pr <- max.col(P[idx, , drop=FALSE]) - 1L; mean(pr == y[idx]) }
macro_f1_from_cm <- function(cm){
  prec <- diag(cm) / pmax(colSums(cm), 1)
  rec <- diag(cm) / pmax(rowSums(cm), 1)
  f1 <- ifelse(prec + rec > 0, 2*prec*rec/(prec+rec), 0)
  mean(f1)
}

get_probs <- function(model, inputs) {
  p <- predict(model, inputs, verbose = 0)
  if (is.list(p)) p <- p[[1]]
  
  
  p <- tryCatch(as.array(p), error = function(e) p)
  
  dims <- dim(p)
  
  if (is.null(dims)) {
    N <- dim(inputs$X_in)[2]
    K <- length(p) / N
    stopifnot(abs(K - round(K)) < 1e-8, K > 0)
    p <- array(p, dim = c(1L, N, as.integer(round(K))))
    return(p[1, , ]) 
  }
  
  if (length(dims) == 2L) {
    return(p)
  }
  
  if (length(dims) == 3L) {
    # 1 x N x K -> drop batch dim
    stopifnot(dims[1] == 1L)
    return(p[1, , ])
  }
  
  if (length(dims) == 4L && dims[1] == 1L) {
    N <- dims[2]
    K <- prod(dims[3:4])
    p <- array(p, dim = c(1L, N, K))
    return(p[1, , ])
  }
  
  stop(sprintf("Unexpected prediction shape: %s", paste(dims, collapse = "x")))
}
