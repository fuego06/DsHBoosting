evaluate_multiclass <- function(y_true, y_proba, plot_roc = TRUE) {
  stopifnot(is.matrix(y_proba) || is.data.frame(y_proba))
  y_proba <- as.matrix(y_proba)
  
  # ----- align levels & column names -----
  lev_vals  <- sort(unique(as.vector(y_true)))
  K         <- length(lev_vals)
  stopifnot(ncol(y_proba) == K)
  
  lab_names <- paste0("C", lev_vals)       # safe labels like C0, C1, ...
  y_fac     <- factor(y_true, levels = lev_vals, labels = lab_names)
  colnames(y_proba) <- lab_names           # make predictors match response levels
  
  # optional: row-normalize if not already probabilities
  rs <- rowSums(y_proba); rs[rs == 0] <- 1
  y_proba <- y_proba / rs
  
  # ----- hard predictions -----
  pred_idx  <- max.col(y_proba, ties.method = "first")
  y_predfac <- factor(lab_names[pred_idx], levels = lab_names)
  
  # ----- confusion matrix & metrics -----
  cm       <- table(True = y_fac, Pred = y_predfac)
  diag_c   <- diag(cm)
  rowsums  <- rowSums(cm)
  colsums  <- colSums(cm)
  n        <- sum(cm)
  
  precision <- ifelse(colsums > 0, diag_c / colsums, NA_real_)
  recall    <- ifelse(rowsums > 0, diag_c / rowsums, NA_real_)
  f1        <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), NA_real_)
  
  macroPrecision <- mean(precision, na.rm = TRUE)
  macroRecall    <- mean(recall,    na.rm = TRUE)
  macroF1        <- mean(f1,        na.rm = TRUE)
  accuracy       <- sum(diag_c) / n   # also micro-F1 for single-label multiclass
  
  per_class <- data.frame(
    class     = lab_names,
    precision = precision,
    recall    = recall,
    f1        = f1,
    support   = rowsums,
    row.names = NULL
  )
  
  # ----- ROC AUC (OvR) + optional plot -----
  suppressPackageStartupMessages(require(pROC))
  aucs <- sapply(seq_along(lab_names), function(k) {
    r <- pROC::roc(response = as.numeric(y_fac == lab_names[k]),
                   predictor = y_proba[, k], quiet = TRUE)
    as.numeric(r$auc)
  })
  names(aucs) <- lab_names
  
  if (plot_roc) {
    cols <- seq_along(lab_names)
    r1 <- pROC::roc(as.numeric(y_fac == lab_names[1]), y_proba[, 1], quiet = TRUE)
    plot(r1, lwd = 2, main = "One-vs-Rest ROC (per class)")
    for (k in 2:length(lab_names)) {
      rk <- pROC::roc(as.numeric(y_fac == lab_names[k]), y_proba[, k], quiet = TRUE)
      plot(rk, add = TRUE, col = cols[k], lwd = 2)
    }
    legend("bottomright",
           legend = paste0(lab_names, sprintf("  AUC=%.3f", aucs)),
           col = cols, lwd = 2, cex = 0.8, bg = "white")
  }
  
  list(
    confusion_matrix = cm,
    per_class        = per_class,
    macro            = c(macroPrecision = macroPrecision,
                         macroRecall = macroRecall,
                         macroF1 = macroF1),
    accuracy         = accuracy,
    auc_ovr          = aucs
  )
}


entropy_analysis <- function(y_proba,
                             y_true = NULL,              # optional (vector of ints/factors)
                             normalize = TRUE,           # divide by log(K) -> [0,1]
                             log_base = c("e","2"),
                             thresholds = c(0.6, 0.8),   # shown on histogram (if normalized)
                             make_plots = TRUE,
                             top_n = 20) {
  
  log_base <- match.arg(log_base)
  # --- safety & shapes ---
  y_proba <- as.matrix(y_proba)
  N <- nrow(y_proba); K <- ncol(y_proba)
  stopifnot(N > 0, K > 1)
  
  # row-normalize just in case
  rs <- rowSums(y_proba); rs[rs == 0] <- 1
  y_proba <- y_proba / rs
  
  # --- entropy & confidence ---
  eps <- 1e-12
  if (log_base == "e") {
    H <- -rowSums(y_proba * log(pmax(y_proba, eps)))
    Hmax <- log(K)
  } else {
    H <- -rowSums(y_proba * log2(pmax(y_proba, eps)))
    Hmax <- log2(K)
  }
  Hn <- if (normalize) H / Hmax else H
  conf <- if (normalize) 1 - Hn else NA_real_
  pmax_row <- apply(y_proba, 1, max)
  pred_idx  <- max.col(y_proba, ties.method = "first")
  
  df <- data.frame(
    node         = seq_len(N),
    entropy      = H,
    norm_entropy = if (normalize) Hn else NA_real_,
    max_prob     = pmax_row,
    pred_class   = pred_idx
  )
  
  if (!is.null(y_true)) {
    y_true <- as.vector(y_true)
    df$true_class <- y_true
    df$correct    <- as.integer(df$pred_class == df$true_class)
  }
  
  # --- summaries ---
  s <- list(
    K                 = K,
    mean_entropy      = mean(H),
    median_entropy    = median(H),
    sd_entropy        = sd(H),
    mean_norm_entropy = if (normalize) mean(Hn) else NA_real_,
    median_norm_entropy = if (normalize) median(Hn) else NA_real_,
    corr_maxprob_entropy = suppressWarnings(cor(pmax_row, if (normalize) Hn else H))
  )
  
  # fractions above thresholds (only if normalized)
  frac_above <- NULL
  if (normalize && length(thresholds)) {
    frac_above <- sapply(thresholds, function(t) mean(Hn >= t))
    names(frac_above) <- paste0("frac_norm_entropy>=", thresholds)
  }
  
  # per-class means (if labels available)
  by_true <- by_pred <- NULL
  if (!is.null(y_true)) {
    by_true <- aggregate(cbind(entropy = H, norm_entropy = if (normalize) Hn else NA_real_),
                         by = list(true = y_true), FUN = mean)
    by_pred <- aggregate(cbind(entropy = H, norm_entropy = if (normalize) Hn else NA_real_),
                         by = list(pred = pred_idx), FUN = mean)
  }
  
  # --- plots (entropy only) ---
  if (make_plots) {
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      library(ggplot2)
      if (normalize) {
        p1 <- ggplot(df, aes(norm_entropy)) +
          geom_histogram(bins = 40) +
          labs(title = "Normalized Entropy", x = "H / Hmax", y = "Count")
        if (length(thresholds)) {
          for (t in thresholds) p1 <- p1 + geom_vline(xintercept = t, linetype = 2)
        }
        print(p1)
      } else {
        print(ggplot(df, aes(entropy)) +
                geom_histogram(bins = 40) +
                labs(title = "Entropy", x = "H", y = "Count"))
      }
      
      # scatter: max prob vs entropy
      xent <- if (normalize) df$norm_entropy else df$entropy
      p2 <- ggplot(df, aes(x = max_prob, y = xent)) +
        geom_point(alpha = 0.35) +
        geom_smooth(se = FALSE) +
        labs(title = "Max softmax vs. entropy",
             x = "Max softmax probability",
             y = if (normalize) "Normalized entropy" else "Entropy") +
        coord_cartesian(xlim = c(0,1))
      print(p2)
      
      if (!is.null(y_true)) {
        # density split by correctness
        df$correct_f <- factor(ifelse(df$correct == 1, "correct", "wrong"),
                               levels = c("correct","wrong"))
        print(
          ggplot(df, aes(x = xent, fill = correct_f)) +
            geom_density(alpha = 0.4) +
            labs(title = "Entropy by correctness",
                 x = if (normalize) "Normalized entropy" else "Entropy",
                 fill = NULL)
        )
      }
    } else {
      par(mfrow = c(1,2))
      hist(if (normalize) Hn else H, breaks = 40,
           main = if (normalize) "Normalized entropy" else "Entropy",
           xlab = if (normalize) "H/Hmax" else "H")
      if (normalize && length(thresholds)) abline(v = thresholds, lty = 2, col = "red")
      plot(pmax_row, if (normalize) Hn else H,
           xlab = "Max softmax probability",
           ylab = if (normalize) "Normalized entropy" else "Entropy",
           main = "Max prob vs. entropy")
      abline(lm((if (normalize) Hn else H) ~ pmax_row), col = "blue")
      par(mfrow = c(1,1))
    }
  }
  
  list(
    summary         = s,
    fractions_above = frac_above,
    by_true_mean    = by_true,
    by_pred_mean    = by_pred,
    table           = df[order(-(if (normalize) df$norm_entropy else df$entropy)), ],
    top_uncertain   = head(df[order(-(if (normalize) df$norm_entropy else df$entropy)), ], top_n)
  )
}



evaluate_multiclass_mask <- function(y_true_all, y_proba_all, mask,
                                     levels_all = NULL, plot_roc = TRUE) {
  stopifnot(is.matrix(y_proba_all) || is.data.frame(y_proba_all))
  y_proba_all <- as.matrix(y_proba_all)
  stopifnot(length(y_true_all) == nrow(y_proba_all), length(mask) == length(y_true_all))
  
  # Subset rows by mask
  y_true <- y_true_all[mask]
  y_proba <- y_proba_all[mask, , drop = FALSE]
  
  # Fix the label space: use full set if provided, else from all labels (not subset!)
  if (is.null(levels_all)) levels_all <- sort(unique(y_true_all))
  K <- length(levels_all)
  stopifnot(ncol(y_proba) == K)
  
  lab_names <- paste0("C", levels_all)
  y_fac     <- factor(y_true, levels = levels_all, labels = lab_names)
  colnames(y_proba) <- lab_names
  
  # Row-normalize probabilities
  rs <- rowSums(y_proba); rs[rs == 0] <- 1
  y_proba <- y_proba / rs
  
  # Hard preds
  pred_idx  <- max.col(y_proba, ties.method = "first")
  y_predfac <- factor(lab_names[pred_idx], levels = lab_names)
  
  # Confusion matrix & metrics (allow missing classes)
  cm <- table(True = y_fac, Pred = y_predfac)
  # ensure full KxK even if some rows/cols missing
  cm_full <- matrix(0, nrow = K, ncol = K,
                    dimnames = list(True = lab_names, Pred = lab_names))
  cm_full[rownames(cm), colnames(cm)] <- cm
  cm <- cm_full
  
  diag_c  <- diag(cm); rowsums <- rowSums(cm); colsums <- colSums(cm); n <- sum(cm)
  precision <- ifelse(colsums > 0, diag_c/colsums, NA_real_)
  recall    <- ifelse(rowsums > 0, diag_c/rowsums, NA_real_)
  f1        <- ifelse(precision + recall > 0, 2*precision*recall/(precision+recall), NA_real_)
  macroPrecision <- mean(precision, na.rm = TRUE)
  macroRecall    <- mean(recall,    na.rm = TRUE)
  macroF1        <- mean(f1,        na.rm = TRUE)
  accuracy       <- sum(diag_c) / n
  
  per_class <- data.frame(
    class     = lab_names,
    precision = precision,
    recall    = recall,
    f1        = f1,
    support   = rowsums,
    row.names = NULL
  )
  
  # AUC OvR (NA if class absent in this split)
  suppressPackageStartupMessages(require(pROC))
  aucs <- sapply(seq_along(lab_names), function(k) {
    pos <- as.numeric(y_fac == lab_names[k])
    if (sum(pos == 1, na.rm = TRUE) == 0 || sum(pos == 0, na.rm = TRUE) == 0) return(NA_real_)
    as.numeric(pROC::roc(response = pos, predictor = y_proba[, k], quiet = TRUE)$auc)
  })
  names(aucs) <- lab_names
  
  if (plot_roc) {
    present <- which(!is.na(aucs))
    if (length(present) >= 1) {
      r1 <- pROC::roc(as.numeric(y_fac == lab_names[present[1]]), y_proba[, present[1]], quiet = TRUE)
      plot(r1, lwd = 2, main = "One-vs-Rest ROC (per class, masked split)")
      if (length(present) > 1) {
        for (idx in present[-1]) {
          rk <- pROC::roc(as.numeric(y_fac == lab_names[idx]), y_proba[, idx], quiet = TRUE)
          lines(rk, lwd = 2, col = idx)
        }
      }
      legend("bottomright",
             legend = paste0(lab_names, "  AUC=", sprintf("%.3f", aucs)),
             col = seq_along(lab_names), lwd = 2, cex = 0.8, bg = "white")
    } else {
      message("No classes present in this split for ROC plotting.")
    }
  }
  
  list(
    confusion_matrix = cm,
    per_class        = per_class,
    macro            = c(macroPrecision = macroPrecision,
                         macroRecall = macroRecall,
                         macroF1 = macroF1),
    accuracy         = accuracy,
    auc_ovr          = aucs
  )
}


node_degree <- function(A, undirected = TRUE) {
  stopifnot(is.matrix(A), nrow(A) == ncol(A))
  if (undirected) A <- (A + t(A)) > 0
  rowSums(A)
}

degree_bins <- function(deg) {
  cut(deg,
      breaks = c(-Inf, 0, 2, 5, Inf),
      labels = c("deg=0", "deg 1???2", "deg 3???5", ">5"),
      right = TRUE)
}

# ---------- Accuracy by degree ----------
# pNK, yNK: arrays with shape (1, N, K)
acc_by_bin <- function(pNK, yNK, bins) {
  stopifnot(length(dim(pNK)) == 3, length(dim(yNK)) == 3)
  pred <- max.col(pNK[1,, , drop = FALSE]) - 1L
  true <- max.col(yNK[1,, , drop = FALSE]) - 1L
  ok <- as.integer(pred == true)
  tibble(node = seq_along(ok), bin = bins, correct = ok) |>
    group_by(bin) |>
    summarise(n = n(), acc = mean(correct), .groups = "drop") |>
    arrange(bin)
}

# --- FIXED: robust embedding extractor (handles 2D, 3D, lists) ---
get_embeddings_default <- function(model = NULL,
                                   inputs = NULL,
                                   X_features = NULL,
                                   output_index = 1, # if predict() returns a list; can be name or index
                                   squeeze_batch = TRUE, # drop singleton batch/feature dims when possible
                                   N_expected = NULL) { # typically nrow(A)
  # 0) Fallback to features
  if (is.null(model) || is.null(inputs)) {
    Z <- as.matrix(X_features)
    if (!is.null(N_expected) && nrow(Z) != N_expected) {
      Z <- Z[seq_len(min(nrow(Z), N_expected)), , drop = FALSE]
      warning("X_features rows != expected; trimmed.")
    }
    return(Z)
  }
  
  # 1) Predict
  Zpred <- predict(model, inputs)
  
  # 2) If list, choose the correct head
  Z <- if (is.list(Zpred)) {
    if (is.character(output_index)) {
      if (!output_index %in% names(Zpred))
        stop(sprintf("output_index '%s' not in names: %s",
                     output_index, paste(names(Zpred), collapse = ", ")))
      Zpred[[output_index]]
    } else {
      Zpred[[output_index]]
    }
  } else {
    Zpred
  }
  
  # 3) Coerce to base array to ensure 'dim' exists
  Z <- as.array(Z)
  dims <- dim(Z)
  
  # 4) Normalize to (N, D)
  if (is.null(dims)) {
    # vector ??? (N,1)
    Z <- matrix(Z, ncol = 1)
  } else if (length(dims) == 1) {
    # length-N vector ??? (N,1)
    Z <- matrix(as.numeric(Z), ncol = 1)
  } else if (length(dims) == 2) {
    # already (N, D)
    # do nothing
  } else if (length(dims) == 3) {
    # (B, N, D) or (N, D, B)
    if (squeeze_batch && dims[1] == 1) {
      # (1, N, D) -> (N, D)
      Z <- Z[1,,] # NOTE: three subscripts
    } else if (squeeze_batch && dims[3] == 1) {
      # (N, D, 1) -> (N, D)
      Z <- Z[,,1] # NOTE: three subscripts
    } else {
      # conservative flatten (B*N, D)
      Z <- matrix(aperm(Z, c(2, 3, 1)), nrow = dims[1] * dims[2], ncol = dims[3])
      warning("Flattened 3D output into (B*N, D). If incorrect, pick a different output_index.")
    }
  } else if (length(dims) > 3) {
    # Try to drop any singleton dimensions first
    Zd <- drop(Z)
    if (is.null(dim(Zd))) {
      Z <- matrix(Zd, ncol = 1)
    } else if (length(dim(Zd)) == 2) {
      Z <- Zd
    } else if (length(dim(Zd)) == 3 && squeeze_batch) {
      d <- dim(Zd)
      if (d[1] == 1) Z <- Zd[1,,]
      else if (d[3] == 1) Z <- Zd[,,1]
      else {
        Z <- matrix(aperm(Zd, c(2, 3, 1)), nrow = d[1] * d[2], ncol = d[3])
        warning("Flattened squeezed 3D output into (B*N, D).")
      }
    } else {
      stop(sprintf("Unsupported tensor rank after drop(): %s", paste(dim(Zd), collapse = "x")))
    }
  }
  
  # 5) Ensure matrix & row-trim to expected N if provided
  Z <- as.matrix(Z)
  if (!is.null(N_expected) && nrow(Z) != N_expected) {
    Z <- Z[seq_len(min(nrow(Z), N_expected)), , drop = FALSE]
    warning(sprintf("Embeddings rows %d != expected %d; trimmed to min.",
                    nrow(Z), N_expected))
  }
  Z
}

# Optional: quick inspector for predict() shape (use ad-hoc, not required)
inspect_predict_shape <- function(model, inputs) {
  z <- predict(model, inputs)
  if (is.list(z)) {
    cat("predict() -> LIST:\n"); print(lapply(z, dim))
    if (!is.null(names(z))) cat("names:", paste(names(z), collapse = ", "), "\n")
  } else {
    cat("predict() -> TENSOR dim:\n"); print(dim(z))
  }
  invisible(z)
}

# ---------- Dimensionality reduction ----------
embed_2d <- function(Z, method = c("pca","tsne"), perplexity = 30, seed = 42) {
  method <- match.arg(method); set.seed(seed)
  if (method == "pca") {
    pcs <- prcomp(Z, center = TRUE, scale. = TRUE)
    Y <- pcs$x[, 1:2, drop = FALSE]
    colnames(Y) <- c("DR1","DR2")
    list(Y = Y, model = pcs)
  } else {
    ts <- Rtsne(as.matrix(Z), perplexity = perplexity,
                check_duplicates = FALSE, pca = TRUE)
    Y <- ts$Y
    colnames(Y) <- c("DR1","DR2")
    list(Y = Y, model = ts)
  }
}

# ---------- Outlier scoring (2D) ----------
lof_2d <- function(Y, k = 20) {
  Y=as.matrix(Y)
  n=nrow(Y)
  if(n<3) return(rep(0,n))
  k_use=max(2L,min(as.integer(k),n-1L))
  dbscan::lof(as.matrix(Y), minPts = k)
}
mahalanobis_2d <- function(Y) {
  Y <- as.matrix(Y)
  mu <- colMeans(Y)
  S <- cov(Y) + diag(1e-6, 2)
  mahalanobis(Y, center = mu, cov = S)
}
rank_outliers <- function(lof_scores, maha_dist, w_lof = 0.5) {
  z1 <- scale(lof_scores); z2 <- scale(maha_dist)
  as.numeric(w_lof * z1 + (1 - w_lof) * z2)
}

# ---------- Main figures ----------
plot_embedding_with_outliers <- function(
    A, X_features, node_ids = NULL,
    method = c("pca","tsne"),
    model = NULL, model_inputs = NULL,
    embedding_output_index = 1, # <- choose the output head
    perplexity = 30, # only used for t-SNE
    deg_as_color = TRUE,
    k_lof = 20, w_lof = 0.6, top_m = 25,
    seed = 42
) {
  method <- match.arg(method)
  N <- nrow(A)
  if (is.null(node_ids)) node_ids <- seq_len(N)
  
  deg <- node_degree(A, undirected = TRUE)
  bins <- degree_bins(deg)
  
  Z <- get_embeddings_default(model, model_inputs, X_features,
                              output_index = embedding_output_index,
                              squeeze_batch = TRUE,
                              N_expected = N)
  
  dr <- embed_2d(Z, method = method, perplexity = perplexity, seed = seed)
  Y2 <- as.data.frame(dr$Y)
  Y2$node_id <- node_ids
  Y2$deg <- deg
  Y2$bin <- bins
  
  lof_s <- lof_2d(Y2[, c("DR1","DR2")], k = k_lof)
  maha_s <- mahalanobis_2d(Y2[, c("DR1","DR2")])
  combo <- rank_outliers(lof_s, maha_s, w_lof = w_lof)
  
  Y2$lof <- lof_s
  Y2$maha <- maha_s
  Y2$score <- combo
  
  always <- which(Y2$deg == 0)
  top_ix <- order(Y2$score, decreasing = TRUE)
  top_ix <- unique(c(always, head(top_ix, top_m)))
  
  Y2$is_outlier <- FALSE
  Y2$is_outlier[top_ix] <- TRUE
  
  p <- ggplot(Y2, aes(DR1, DR2)) +
    geom_point(aes(color = if (deg_as_color) bin else NULL),
               alpha = 0.7, size = 1.6) +
    geom_point(data = Y2[Y2$is_outlier, ],
               shape = 21, stroke = 0.7, size = 3,
               fill = NA, color = "black") +
    ggrepel::geom_label_repel(
      data = Y2[Y2$is_outlier, ],
      aes(label = node_id),
      size = 3, label.padding = unit(0.15, "lines"),
      max.overlaps = Inf, seed = seed
    ) +
    labs(
      title = paste0("Node embedding (", toupper(method), ") with outliers labeled"),
      color = "Degree bin"
    ) +
    theme_minimal(base_size = 12)
  
  list(plot = p, table = Y2, outlier_nodes = Y2$node_id[top_ix])
}

# ---- reshape helpers ----
.as_NK <- function(arr, name = "tensor") {
  d <- dim(arr)
  if (is.null(d)) stop(sprintf("%s has no dim(); got a vector.", name))
  
  if (length(d) == 3L) {
    # Expect (B, N, K); usually B == 1
    B <- d[1]; N <- d[2]; K <- d[3]
    if (B != 1L) warning(sprintf("%s: batch size B = %d (expected 1). Using first batch.", name, B))
    # Proper 3D indexing to get an (N, K) matrix:
    M <- arr[1, , , drop = FALSE] # still 3D: (1, N, K)
    M <- array(M, dim = c(N, K)) # force to (N, K) without flattening to (N*K)
    return(M)
  } else if (length(d) == 2L) {
    # Already (N, K)
    return(arr)
  } else {
    stop(sprintf("%s has unsupported rank %s", name, paste(d, collapse="x")))
  }
}

# ---- accuracy by degree (robust) ----
acc_by_bin <- function(pNK, yNK, bins = NULL, A = NULL) {
  # pNK, yNK expected as (1,N,K) or (N,K). We will coerce to (N,K).
  P <- .as_NK(pNK, "pNK")
  Y <- .as_NK(yNK, "yNK")
  
  if (!all(dim(P) == dim(Y))) {
    stop(sprintf("pNK and yNK shapes differ: P=(%s) vs Y=(%s)",
                 paste(dim(P), collapse="x"), paste(dim(Y), collapse="x")))
  }
  
  N <- nrow(P)
  
  # bins: either provided, or computed from A
  if (is.null(bins)) {
    if (is.null(A)) stop("Provide either 'bins' or adjacency 'A' to compute degree bins.")
    deg <- rowSums((A + t(A)) > 0)
    bins <- cut(deg, breaks = c(-Inf, 0, 2, 5, Inf),
                labels = c("deg=0","deg 1???2","deg 3???5",">5"), right = TRUE)
  }
  
  if (length(bins) != N) {
    stop(sprintf("Length of bins (%d) must match N (%d).", length(bins), N))
  }
  
  pred <- max.col(P) - 1L
  true <- max.col(Y) - 1L
  ok <- as.integer(pred == true)
  
  dplyr::tibble(node = seq_len(N), bin = bins, correct = ok) |>
    dplyr::group_by(bin) |>
    dplyr::summarise(n = dplyr::n(), acc = mean(correct), .groups = "drop") |>
    dplyr::arrange(bin)
}

plot_accuracy_by_degree <- function(pNK, yNK, A) {
  # Compute bins from A inside, using the robust acc_by_bin()
  bybin <- acc_by_bin(pNK, yNK, bins = NULL, A = A)
  
  g <- ggplot2::ggplot(bybin, ggplot2::aes(x = bin, y = acc)) +
    ggplot2::geom_col(width = 0.7) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", acc)),
                       vjust = -0.2, size = 3.4) +
    ggplot2::ylim(0, 1) +
    ggplot2::labs(title = "Accuracy by degree bin", x = "Degree bin", y = "Accuracy") +
    ggplot2::theme_minimal(base_size = 12)
  
  list(plot = g, table = bybin)
}


get_softmax_probs <- function(model, inputs, output = 1) {
  out <- predict(model, inputs)
  p <- if (is.list(out)) {
    if (is.character(output)) out[[output]] else out[[output]]
  } else {
    out
  }
  # ensure (1, N, K)
  if (length(dim(p)) == 2L) {
    p <- array_reshape(p, c(1, nrow(p), ncol(p)))
  }
  p
}


predict_probs <- function(model, inputs_base) {
  p <- predict(model, inputs_base)
  if (is.list(p)) p <- p[[1]]
  if (length(dim(p)) == 2L) p <- array_reshape(p, c(1, nrow(p), ncol(p)))
  p[1,,]
}
node_accuracy <- function(pNK, yNK) {
  pred <- max.col(pNK) - 1L
  true <- max.col(yNK) - 1L
  mean(pred == true)
}


library(dplyr)
library(tidyr)
library(ggplot2)

plot_stage_losses_long <- function(hist_df) {
  # Make a global epoch index (across rounds)
  hist_df2 <- hist_df %>%
    arrange(round, epoch_in_round, metric, data) %>%
    mutate(global_epoch = row_number())
  
  loss_df <- hist_df2 %>%
    filter(grepl("loss", metric)) %>% # keep only loss metrics
    mutate(
      split = ifelse(data == "training", "train", "val")
    )
  
  if (nrow(loss_df) == 0) {
    stop("No rows with metric containing 'loss' in history.")
  }
  
  ggplot(loss_df, aes(x = global_epoch, y = value, colour = split)) +
    geom_line(alpha = 0.7) +
    facet_wrap(~ metric, scales = "free_y") +
    labs(
      x = "Global epoch (over all rounds & metrics)",
      y = "Loss",
      colour = "Split",
      title = "Per-metric loss curves (Final + Aux heads + total loss)"
    ) +
    theme_minimal(base_size = 12)
}

plot_mean_stage_loss_long <- function(hist_df) {
  hist_df2 <- hist_df %>%
    arrange(round, epoch_in_round, metric, data) %>%
    mutate(global_epoch = row_number())
  
  # Keep only stage head losses (FinalSoftmax, AuxSoftmax_*, etc.)
  stage_loss_df <- hist_df2 %>%
    filter(grepl("Softmax", metric), grepl("loss", metric))
  
  if (nrow(stage_loss_df) == 0) {
    stop("No 'Softmax' loss metrics found in history (e.g., 'FinalSoftmax_loss').")
  }
  
  mean_df <- stage_loss_df %>%
    mutate(split = ifelse(data == "training", "train", "val")) %>%
    group_by(round, epoch_in_round, split) %>%
    summarise(
      mean_stage_loss = mean(value, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(round, epoch_in_round, split) %>%
    mutate(global_epoch = row_number())
  
  ggplot(mean_df, aes(x = global_epoch, y = mean_stage_loss, colour = split)) +
    geom_line(size = 1) +
    labs(
      x = "Global epoch (over all rounds)",
      y = "Mean loss over Softmax heads",
      colour = "Split",
      title = "Mean stage loss (FinalSoftmax + AuxSoftmax heads)"
    ) +
    theme_minimal(base_size = 12)
}

plot_final_accuracy_long <- function(hist_df) {
  hist_df2 <- hist_df %>%
    arrange(round, epoch_in_round, metric, data) %>%
    mutate(global_epoch = row_number())
  
  # Which metrics are accuracy-type?
  acc_metrics <- hist_df2 %>%
    filter(grepl("acc", metric, ignore.case = TRUE)) %>%
    distinct(metric) %>%
    pull(metric)
  
  if (length(acc_metrics) == 0) {
    stop("No accuracy metrics found in history (metric column).")
  }
  
  # For simplicity, use the first accuracy metric as 'final'
  final_acc_metric <- acc_metrics[1]
  
  acc_df <- hist_df2 %>%
    filter(metric == final_acc_metric) %>%
    mutate(split = ifelse(data == "training", "train", "val"))
  
  ggplot(acc_df, aes(x = global_epoch, y = value, colour = split)) +
    geom_line(size = 1) +
    labs(
      x = "Global epoch",
      y = "Accuracy",
      colour = "Split",
      title = paste0("Accuracy over training (metric = ", final_acc_metric, ")")
    ) +
    theme_minimal(base_size = 12)
}



library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

plot_stage_head_losses <- function(hist_df) {
  hist_df2 <- hist_df %>%
    arrange(round, epoch_in_round, metric, data) %>%
    mutate(global_epoch = row_number())
  
  # Keep only loss metrics from Softmax heads
  stage_loss_df <- hist_df2 %>%
    filter(
      grepl("Softmax", metric), # FinalSoftmax / AuxSoftmax
      grepl("loss", metric)
    ) %>%
    mutate(
      split = ifelse(data == "training", "train", "val"),
      stage = dplyr::case_when(
        grepl("^FinalSoftmax", metric) ~ "Final",
        grepl("^AuxSoftmax_t[0-9]+", metric) ~ {
          # extract t index: AuxSoftmax_t1_loss -> "t1"
          t_idx <- str_match(metric, "AuxSoftmax_(t[0-9]+)")[, 2]
          paste0("Stage_", t_idx)
        },
        TRUE ~ metric # fallback
      )
    )
  
  if (nrow(stage_loss_df) == 0) {
    stop("No Softmax head loss metrics (like 'FinalSoftmax_loss') found in history.")
  }
  
  ggplot(stage_loss_df, aes(x = global_epoch, y = value, colour = split)) +
    geom_line(alpha = 0.8) +
    facet_wrap(~ stage, scales = "free_y") +
    labs(
      x = "Global epoch (over all rounds)",
      y = "Loss",
      colour = "Split",
      title = "Per-stage Softmax head losses (Final + Aux stages)"
    ) +
    theme_minimal(base_size = 12)
}


